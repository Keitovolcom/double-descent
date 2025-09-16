import numpy as np
import torch
import os

# ----- ユーザ設定 -----
npy_dir = "data/EMNIST/EMNIST_0.2"  # NumPyファイルが置かれているディレクトリ
out_path = "data/EMNIST/emnist_noise_0.2/train_data_from_npy.pt"  # 保存先
# ----------------------

# 1. NumPyファイルをロード
npy_images = np.load(os.path.join(npy_dir, "train_images.npy"))   # (240000, 1, 32, 32)
npy_labels = np.load(os.path.join(npy_dir, "train_labels.npy"))   # (240000,)
npy_noise  = np.load(os.path.join(npy_dir, "train_noise_info.npy"))  # (240000,)

print("[NumPy] images:", npy_images.shape, npy_images.dtype)
print("[NumPy] labels:", npy_labels.shape, npy_labels.dtype)
print("[NumPy] noise :", npy_noise.shape,  npy_noise.dtype)

# 2. torch.Tensor に変換
x_train = torch.tensor(npy_images, dtype=torch.float32)
y_train = torch.tensor(npy_labels, dtype=torch.int64)
noise_info = torch.tensor(npy_noise, dtype=torch.int32)

# 3. 辞書形式にして保存
data_dict = {
    "x_train": x_train,
    "y_train": y_train,
    "noise_info": noise_info
}

torch.save(data_dict, out_path)
print(f"✅ Saved as: {out_path}")
