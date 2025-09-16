import numpy as np
import torch, hashlib, os, sys

# ---------- ユーザ設定 ----------
npy_dir = "data/EMNIST/EMNIST_0.2"                    # *.npy があるディレクトリ
pt_path = "data/EMNIST/emnist_noise_0.2/train_data_merged.pt"
# -------------------------------

# 1) NumPy 側をロード
npy_images = np.load(os.path.join(npy_dir, "train_images.npy"))
npy_labels = np.load(os.path.join(npy_dir, "train_labels.npy"))
npy_noise  = np.load(os.path.join(npy_dir, "train_noise_info.npy"))
print("[NumPy] images:", npy_images.shape, npy_images.dtype)
print("[NumPy] labels:", npy_labels.shape, npy_labels.dtype)
print("[NumPy] noise :", npy_noise.shape,  npy_noise.dtype)

# 2) .pt 側をロード
obj = torch.load(pt_path, map_location="cpu")

# --- 明示的に型を表示して確認 ---
print("\n[Info] .pt type:", type(obj))

# 3) 形式判定（dict or tuple）
if isinstance(obj, (list, tuple)):
    print("[Info] Detected tuple/list format")
    pt_images, pt_labels = obj[:2]
    pt_noise = obj[2] if len(obj) > 2 else None

elif isinstance(obj, dict):
    print("[Info] Detected dict format")
    print("[Debug] Keys in .pt dict:", list(obj.keys()))

    # 候補を拡充（辞書形式に対応）
    img_keys = ["x_train", "images", "data", "x"]
    lbl_keys = ["y_train", "labels", "targets", "y"]
    noise_keys = ["noise_info", "noisy", "noise"]

    def find_key(candidates):
        for k in candidates:
            if k in obj:
                return obj[k]
        raise KeyError(f"No matching key found in {list(obj.keys())}")

    pt_images = find_key(img_keys)
    pt_labels = find_key(lbl_keys)

    pt_noise = None
    for k in noise_keys:
        if k in obj:
            pt_noise = obj[k]
            break

else:
    raise ValueError("Unsupported format in train_data.pt")

# 4) NumPy array に変換
pt_images_np = pt_images.numpy() if torch.is_tensor(pt_images) else np.asarray(pt_images)
pt_labels_np = pt_labels.numpy() if torch.is_tensor(pt_labels) else np.asarray(pt_labels)
pt_noise_np  = pt_noise.numpy()  if pt_noise is not None and torch.is_tensor(pt_noise) else (
               np.asarray(pt_noise) if pt_noise is not None else None)

print("[.pt] images:", pt_images_np.shape, pt_images_np.dtype)
print("[.pt] labels:", pt_labels_np.shape, pt_labels_np.dtype)
print("[.pt] noise :", None if pt_noise_np is None else (pt_noise_np.shape, pt_noise_np.dtype))

# 5) 軽いハッシュ比較
def sha256(a):
    return hashlib.sha256(a.tobytes()).hexdigest()

print("\n--- quick hash check ---")
print("images hash numpy vs pt:", sha256(npy_images), sha256(pt_images_np))
print("labels hash numpy vs pt:", sha256(npy_labels), sha256(pt_labels_np))
if pt_noise_np is not None:
    print("noise  hash numpy vs pt:", sha256(npy_noise),  sha256(pt_noise_np))

# 6) 完全一致チェック
same_images = np.array_equal(npy_images, pt_images_np)
same_labels = np.array_equal(npy_labels, pt_labels_np)
same_noise  = pt_noise_np is None or np.array_equal(npy_noise, pt_noise_np)

print("\n--- element-wise equality ---")
print("images identical:", same_images)
print("labels identical:", same_labels)
print("noise  identical:", same_noise)
