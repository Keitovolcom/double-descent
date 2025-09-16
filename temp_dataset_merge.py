import torch

# パス設定
src_path = "data/EMNIST/emnist_noise_0.2/train_data.pt"           # y_train, noise_info が正しいファイル
dst_path = "data/EMNIST/emnist_noise_0.2/train_data_temp.pt"      # x_train が正しいファイル
out_path = "data/EMNIST/emnist_noise_0.2/train_data_merged.pt"    # 出力ファイル名（新規）

# 1. y_train, noise_info を取得
src = torch.load(src_path, map_location="cpu")
y_train = src["y_train"]
noise_info = src["noise_info"]

# 2. x_train を取得
dst = torch.load(dst_path, map_location="cpu")
x_train = dst["x_train"]

# 3. 新しい dict を構築
merged = {
    "x_train": x_train,
    "y_train": y_train,
    "noise_info": noise_info
}

# 4. 保存
torch.save(merged, out_path)

print(f"✅ 新しいファイルに保存しました: {out_path}")
