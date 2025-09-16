import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms

# データ読み込み
data = torch.load("data/EMNIST/emnist_noise_0.2/train_data.pt")
imgs = data['x_train'][:16]       # 最初の16画像
labels = data['y_train'][:16]
noise_flags = data['noise_info'][:16]



# normalize = transforms.Normalize((0.1307,), (0.3081,))  # 1ch想定
# imgs = normalize(imgs)  # shape: [B, 1, H, W]

# 1ch → 3ch に変換（RGB 表示対応）
imgs_rgb = imgs.repeat(1, 3, 1, 1)

# make_grid で画像まとめる
grid = make_grid(imgs_rgb, nrow=4, padding=2)
npimg = grid.permute(1, 2, 0).numpy()


# 表示
plt.figure(figsize=(8, 8))
plt.imshow(npimg)
plt.axis('off')
titles = [f"{l.item()}{'*' if n.item() else ''}" for l, n in zip(labels, noise_flags)]
plt.title("Labels ( * = noisy ): " + " | ".join(titles), fontsize=10)
plt.savefig("npy.png")
print("[Info] 画像を npy.png に保存しました。")
