import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = r"D:\EyeDieaseProject\ODIR-5K\ODIR-5K"
CSV_PATH = os.path.join(BASE_DIR, "full_df.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Training Images")

df = pd.read_csv(CSV_PATH)

# 随机取5张图，显示图片+对应标签
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
samples = df.sample(5, random_state=42)

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = os.path.join(TRAIN_IMG_DIR, row['Left-Fundus'])
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"N={row['N']} D={row['D']} G={row['G']}\n"
                      f"C={row['C']} A={row['A']} H={row['H']}\n"
                      f"M={row['M']} O={row['O']}", fontsize=8)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(r"D:\EyeDieaseProject\check_samples.png", dpi=150)
plt.show()
print("检查图片已保存！")

# 打印前10行的标签
print("\n前10行标签:")
print(df[['Left-Fundus','N','D','G','C','A','H','M','O']].head(10))