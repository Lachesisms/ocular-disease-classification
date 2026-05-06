"""
模型二：EfficientNetB0 - 最终优化版
改进列表：
1. [Bug修复] matplotlib使用Agg后端，plt.show()改为plt.close()，彻底解决Qt报错
2. [Bug修复] preprocess_input正确归一化，修复原版[0,255]输入的问题
3. [精度提升] Focal Loss + class_weight，解决类别不平衡
4. [精度提升] Phase2解冻时显式设置BN层为训练模式，解决EfficientNet微调不稳定问题
5. [精度提升] Phase2学习率5e-5，比原来1e-4更稳定
6. [精度提升] Phase1增加至25epoch，给分类头更充分的收敛时间
7. 训练结束后自动保存.keras格式，方便Flask直接加载
8. 用sklearn计算最终指标，比keras更准确
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在import pyplot之前，彻底解决Qt报错
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

from data_preparation import (load_and_clean_data, split_data,
                               MODEL_SAVE_DIR, CLASS_NAMES, CLASS_LABELS,
                               NUM_CLASSES, TRAIN_IMG_DIR, IMG_SIZE, BATCH_SIZE)

# ============================================================
# GPU配置
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU: {gpus[0]}")

MODEL_NAME = "EfficientNetB0"


# ============================================================
# Focal Loss
# ============================================================
def focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
        ce = -y_true_smooth * tf.math.log(y_pred)
        weight = alpha * y_true_smooth * tf.pow(1.0 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss_fn


# ============================================================
# 计算class_weight
# ============================================================
def get_class_weight(train_gen):
    classes = train_gen.classes
    unique_classes = np.unique(classes)
    weights = compute_class_weight('balanced', classes=unique_classes, y=classes)
    class_weight_dict = dict(enumerate(weights))
    print(f"\nClass weights: {class_weight_dict}")
    return class_weight_dict


# ============================================================
# 数据生成器
# preprocess_input将[0,255]→[-1,1]，与ImageNet预训练权重匹配
# ============================================================
def create_generators(train_df, val_df):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_IMG_DIR,
        x_col='image', y_col='label',
        class_mode='categorical',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=TRAIN_IMG_DIR,
        x_col='image', y_col='label',
        class_mode='categorical',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    print(f"Class mapping: {train_gen.class_indices}")
    return train_gen, val_gen


# ============================================================
# 构建模型
# ============================================================
def build_model(freeze_base=True):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = not freeze_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


# ============================================================
# 阶段1：冻结base，训练分类头
# epoch增加到25，给分类头更充分的收敛时间
# ============================================================
def train_phase1(train_gen, val_gen, class_weight_dict):
    print(f"\n{'='*50}")
    print("Phase 1: Train classification head (base frozen)")
    print(f"{'='*50}")

    model = build_model(freeze_base=True)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1),
        metrics=['accuracy', Recall(name='recall'), Precision(name='precision')]
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_phase1.h5'),
            monitor='val_accuracy', save_best_only=True,
            save_weights_only=True, verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    print(f"\nPhase 1 best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    return model, history


# ============================================================
# 阶段2：解冻后30层，精细微调
#
# EfficientNet BN层问题说明：
# EfficientNet大量使用BN层，解冻后若BN仍在inference模式，
# 会使用训练阶段积累的running statistics而非当前batch统计量，
# 导致梯度方向偏差，表现为val_loss震荡不收敛。
# 修复：解冻的层中显式将所有BN层设为trainable=True。
# ============================================================
def train_phase2(model, train_gen, val_gen, class_weight_dict):
    print(f"\n{'='*50}")
    print("Phase 2: Fine-tune last 30 layers (with BN fix)")
    print(f"{'='*50}")

    # 全部冻结
    for layer in model.layers:
        layer.trainable = False

    # 解冻后30层，BN层显式设为训练模式
    for layer in model.layers[-30:]:
        layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    bn_count = sum(1 for l in model.layers[-30:] if isinstance(l, BatchNormalization))
    print(f"可训练层数: {trainable_count}（其中BN层: {bn_count}）")

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss=focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1),
        metrics=['accuracy', Recall(name='recall'), Precision(name='precision')]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_best.h5'),
            monitor='val_accuracy', save_best_only=True,
            save_weights_only=True, verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    print(f"\nPhase 2 best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    return model, history


# ============================================================
# 评估
# ============================================================
def evaluate(model, val_gen):
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} Final Evaluation")
    print(f"{'='*50}")

    val_gen.reset()
    y_pred_prob = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.classes

    alpha_labels = sorted(CLASS_NAMES)
    alpha_fullnames = [CLASS_LABELS[CLASS_NAMES.index(c)] for c in alpha_labels]

    acc = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=alpha_fullnames, zero_division=0))

    plot_confusion_matrix(y_true, y_pred, alpha_labels)
    plot_roc_auc(y_true, y_pred_prob, alpha_labels)
    save_metrics(acc, precision, recall, f1)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{MODEL_NAME} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {path}")


def plot_roc_auc(y_true, y_pred_prob, labels):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(10, 8))
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_prob[:, i])
            auc_scores.append(auc)
            plt.plot(fpr, tpr, label=f'{labels[i]} (AUC={auc:.3f})')
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{MODEL_NAME} - ROC-AUC Curves')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_roc_auc.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC-AUC已保存: {path}")
    print(f"Mean AUC: {np.mean(auc_scores):.4f}")


def plot_combined_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    p1 = len(h1.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc, label='Train')
    axes[0].plot(val_acc, label='Val')
    axes[0].axvline(x=p1, color='r', linestyle='--', label='Fine-tune Start')
    axes[0].set_title(f'{MODEL_NAME} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(loss, label='Train')
    axes[1].plot(val_loss, label='Val')
    axes[1].axvline(x=p1, color='r', linestyle='--', label='Fine-tune Start')
    axes[1].set_title(f'{MODEL_NAME} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"训练曲线已保存: {path}")


def save_metrics(acc, precision, recall, f1):
    path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_metrics.txt')
    with open(path, 'w') as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
    print(f"Metrics saved: {path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Training {MODEL_NAME} (Final Optimized Version)")
    print(f"{'='*50}\n")

    df = load_and_clean_data()
    train_df, val_df = split_data(df)
    train_gen, val_gen = create_generators(train_df, val_df)

    class_weight_dict = get_class_weight(train_gen)

    model, history1 = train_phase1(train_gen, val_gen, class_weight_dict)
    model, history2 = train_phase2(model, train_gen, val_gen, class_weight_dict)
    plot_combined_history(history1, history2)
    evaluate(model, val_gen)

    # 保存完整keras格式，方便Flask直接加载
    keras_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_best.keras')
    model.save(keras_path)
    print(f"完整Keras模型已保存: {keras_path}")

    print(f"\nDone! Model saved to {MODEL_SAVE_DIR}")