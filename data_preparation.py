"""
数据预处理模块 - 简化版
直接使用原始图像，让ResNet50的预训练权重正常工作
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = r"D:\EyeDieaseProject\ODIR-5K\ODIR-5K"
CSV_PATH = os.path.join(BASE_DIR, "full_df.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Training Images")
MODEL_SAVE_DIR = r"D:\EyeDieaseProject\model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 类别配置
CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
CLASS_LABELS = [
    'Normal', 'Diabetes', 'Glaucoma', 'Cataract',
    'Age-related Macular Degeneration',
    'Hypertension', 'Pathological Myopia', 'Other'
]
NUM_CLASSES = 8
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # 增大batch size，训练更稳定


# ============================================================
# 数据加载
# ============================================================
def load_and_clean_data():
    df = pd.read_csv(CSV_PATH)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    left_df = pd.DataFrame({
        'image': df['Left-Fundus'].values,
        'label': _get_labels(df).values
    })
    right_df = pd.DataFrame({
        'image': df['Right-Fundus'].values,
        'label': _get_labels(df).values
    })

    combined_df = pd.concat([left_df, right_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=['image', 'label'])

    print(f"Total samples: {len(combined_df)}")
    print(f"\nClass distribution:")
    dist = combined_df['label'].value_counts()
    for cls, name in zip(CLASS_NAMES, CLASS_LABELS):
        count = dist.get(cls, 0)
        pct = count / len(combined_df) * 100
        print(f"  {cls} ({name:35s}): {count:5d} ({pct:.1f}%)")

    return combined_df


def _get_labels(df):
    """单标签优先级：N > D > G > C > A > H > M > O"""
    conditions = [
        (df['N'] == 1, 'N'),
        (df['D'] == 1, 'D'),
        (df['G'] == 1, 'G'),
        (df['C'] == 1, 'C'),
        (df['A'] == 1, 'A'),
        (df['H'] == 1, 'H'),
        (df['M'] == 1, 'M'),
    ]
    labels = pd.Series(['O'] * len(df), index=df.index)
    for mask, label in reversed(conditions):
        labels[mask] = label
    return labels


def split_data(df):
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ============================================================
# 关键修复：使用ResNet专用预处理函数替代rescale
# preprocess_input会做正确的均值/方差归一化
# 与ImageNet预训练权重完全匹配
# ============================================================
def create_generators(train_df, val_df):
    # 训练集：有数据增强
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ResNet专用预处理
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.1,
        fill_mode='nearest'
    )

    # 验证集：只做ResNet预处理，不做增强
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_IMG_DIR,
        x_col='image',
        y_col='label',
        class_mode='categorical',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=TRAIN_IMG_DIR,
        x_col='image',
        y_col='label',
        class_mode='categorical',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"\nClass mapping: {train_generator.class_indices}")
    return train_generator, val_generator


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    df = load_and_clean_data()
    train_df, val_df = split_data(df)
    train_gen, val_gen = create_generators(train_df, val_df)
    print("\nData preparation complete!")