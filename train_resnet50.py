"""
模型一：ResNet50 - 改进版
核心改进：
1. Focal Loss 替换 categorical_crossentropy，解决类别不平衡
2. class_weight 双重平衡策略
3. Phase2 只解冻后50层（原来全解冻），保护ImageNet特征
4. Phase2 学习率从1e-5调整为5e-5，配合部分解冻加快收敛
5. 标签平滑 0.1，防止对Normal类过度自信
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

from data_preparation import (load_and_clean_data, split_data,
                               create_generators, MODEL_SAVE_DIR,
                               CLASS_NAMES, CLASS_LABELS, NUM_CLASSES)

# ============================================================
# GPU配置
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU: {gpus[0]}")

MODEL_NAME = "ResNet50"


# ============================================================
# Focal Loss
# gamma=2.0: 标准值，对难样本的聚焦程度
# alpha=0.25: 平衡正负样本权重
# label_smoothing=0.1: 防止模型对dominant class过度自信
# ============================================================
def focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 标签平滑
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

        # Focal Loss计算
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
# 构建模型
# ============================================================
def build_model(freeze_base=True):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = not freeze_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


# ============================================================
# 阶段1：冻结base，只训练分类头
# ============================================================
def train_phase1(train_gen, val_gen, class_weight_dict):
    print(f"\n{'='*50}")
    print("Phase 1: Train classification head (base frozen)")
    print(f"{'='*50}")

    model, _ = build_model(freeze_base=True)
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
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print(f"\nPhase 1 best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    return model, history


# ============================================================
# 阶段2：只解冻后50层进行微调
# 原版解冻全部175层，数据量不足容易破坏ImageNet特征
# 改为只解冻后50层（约含最后2个残差块），平衡迁移与适应
# ============================================================
def train_phase2(model, train_gen, val_gen, class_weight_dict):
    print(f"\n{'='*50}")
    print("Phase 2: Fine-tune last 50 layers")
    print(f"{'='*50}")

    # 全部先冻结，再解冻后50层
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-50:]:
        layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"可训练层数: {trainable_count}")

    # 5e-5: 比原来1e-5稍大，配合部分解冻加快收敛，又不至于破坏特征
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
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-8, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=30,
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

    # 用sklearn计算指标，比keras更准确（keras的recall/precision是per-batch累计）
    acc = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average='weighted')
    from sklearn.metrics import precision_score, recall_score
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{MODEL_NAME} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.show()
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
    plt.show()
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
    plt.show()


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
    print(f"Training {MODEL_NAME} (Improved: Focal Loss + Class Weight)")
    print(f"{'='*50}\n")

    df = load_and_clean_data()
    train_df, val_df = split_data(df)
    train_gen, val_gen = create_generators(train_df, val_df)

    # 计算class_weight
    class_weight_dict = get_class_weight(train_gen)

    # 阶段1
    model, history1 = train_phase1(train_gen, val_gen, class_weight_dict)

    # 阶段2
    model, history2 = train_phase2(model, train_gen, val_gen, class_weight_dict)

    # 绘图
    plot_combined_history(history1, history2)

    # 评估
    evaluate(model, val_gen)

    print(f"\nDone! Model saved to {MODEL_SAVE_DIR}")