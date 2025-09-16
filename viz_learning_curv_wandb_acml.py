import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib


import matplotlib.font_manager as fm

import matplotlib, pathlib, sys
print(pathlib.Path(matplotlib.get_data_path()) / "fonts/ttf")


# rebuild_if_missing=True で見つからなければ自動でキャッシュ再構築


# ===== 設定 =====
entity = "dsml-kernel24"
project = "kobayashi_emnist"
target_run_name = "cnn_5layers_width8_emnist_digits_lr0.01_batch_size128_epoch1000_LabelNoiseRate0.2_Optimsgd_momentum0.0"
output_base = "/workspace/vizualize/ACML/emnist_learning_curve2"

# ===== W&B Run一覧から目的のrun.idを検索 =====
api = wandb.Api()
run_id = None
for run in api.runs(f"{entity}/{project}"):
    if run.name == target_run_name:
        run_id = run.id
        break

if run_id is None:
    raise ValueError(f"[Error] Run name '{target_run_name}' が見つかりませんでした。")

# ===== run.idでログ取得 =====
run = api.run(f"{entity}/{project}/{run_id}")
history = run.history(keys=[
    "epoch",
    "test_error",
    "train_error_total",
    "train_error_clean",
    "train_error_noisy", # カンマが抜けていたのを修正
    "test_loss",
    "train_loss"
])
df = pd.DataFrame(history)
# "epoch"列にNaNが含まれる行を削除してからソート
df = df.dropna(subset=['epoch']).sort_values("epoch")


# ===== データ準備 =====
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30

epochs = df["epoch"]
# NOTE: 全てのエラー率を100で割り、0.0-1.0の割合に変換します。
test_error = df["test_error"] / 100
train_error = df["train_error_total"] / 100
train_error_noisy = df["train_error_noisy"] / 100
train_error_clean = df["train_error_clean"] / 100
# lossデータを抽出
train_loss = df["train_loss"]
test_loss = df["test_loss"]


vertical_epochs = [1,27,55,120]  # 縦線を引きたいエポック（必要に応じて）

### 変更点: グラフを3段に変更し、サイズを調整 ###
fig, (ax1, ax2,) = plt.subplots(2, 1, figsize=(16, 16), constrained_layout=True)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13), constrained_layout=True)

# ===== 上段: Test vs Train error =====
ax1.plot(epochs, test_error, label="test", color="red", linewidth=5)
ax1.plot(epochs, train_error, label="train", color="blue", linewidth=5)
for epoch in vertical_epochs:
    ax1.axvline(x=epoch, color="black", linestyle="-", linewidth=2, zorder=0)
ax1.set_yticks([0.0, 0.1, 0.2, 0.3])
ax1.set_xscale("log")
ax1.set_ylabel("error", fontsize=55)
# Y軸の範囲を割合表記に合わせます (-2% から 52%)
ax1.set_ylim(-0.02, 0.32)
ax1.set_xlim(0.9, 1000)
ax1.legend(loc="upper right", fontsize=45)
# ax1.yaxis.grid(True, linestyle="--", linewidth=0.3)
# ax1.xaxis.grid(True, linestyle="--", linewidth=0.3)
ax1.tick_params(axis="both", labelsize=45, labelbottom=False)


# ===== 中段: Noisy vs Clean Train error (2軸使用) =====

# --- 左Y軸 (Noisy) ---
ax2.plot(epochs, train_error_noisy, label="train noisy", color="blue", linewidth=5,linestyle='--')
ax2.set_ylabel("train noisy error", fontsize=55,)
# Y軸の範囲を割合表記 (-2% から 102%) に設定
ax2.set_ylim(-0.02, 1.02)
ax2.tick_params(axis='y', labelsize=45)
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# --- 右Y軸 (Clean) ---
ax3 = ax2.twinx()
ax3.plot(epochs, train_error_clean, label="train clean", color="blue", linewidth=5)
ax3.set_ylabel("train clean error", fontsize=50)
# NOTE: CleanのY軸範囲を0-4%に対応する割合 (-0.2% から 4.2%) に修正
ax3.set_ylim(-0.00235, 0.11) 
ax3.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])

ax3.tick_params(axis='y', labelsize=45)

# --- 中段の共通設定 ---
ax2.set_xscale("log")
### 変更点: 中段なのでX軸ラベルは非表示に ###
# ax2.set_xlabel("Epoch", fontsize=55)
ax2.set_xlim(0.9, 1000)
# ax2.xaxis.grid(True, linestyle="--", linewidth=0.3)
### 変更点: 中段なのでX軸のティックラベルを非表示に ###
ax2.tick_params(axis="x", labelsize=45)
# 凡例を結合して表示
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=45)
ax2.set_xlabel("epoch", fontsize=55)
ax2.set_xlim(0.9, 1000)
# 縦線を追加
for epoch in vertical_epochs:
    ax2.axvline(x=epoch, color="black", linestyle="-", linewidth=2, zorder=0)


# ### 追加: 下段: Train vs Test Loss =====
# ax4.plot(epochs, test_loss, label="test", color="red", linewidth=3)

# ax4.plot(epochs, train_loss, label="train", color="blue", linewidth=3)

# # 縦線を追加
# for epoch in vertical_epochs:
#     ax4.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

# # 軸の設定
# ax4.set_xscale("log")
# # ax4.set_yscale("log") # Lossは対数スケールの方が見やすいことが多いです
# ax4.set_ylabel("loss", fontsize=55)
# ax4.set_xlabel("epoch", fontsize=55)
# ax4.set_xlim(0.9, 1000)
# ax4.legend(loc="upper right", fontsize=55)
# ax4.yaxis.grid(True, linestyle="--", linewidth=0.3)
# ax4.xaxis.grid(True, linestyle="--", linewidth=0.3)
# ax4.tick_params(axis="both", labelsize=45)
# ax4.set_yticks([0.0, 0.4,0.8, 1.2])


# ===== 保存 =====
plt.savefig(f"{output_base}_ratio.svg", format='svg')
plt.savefig(f"{output_base}_ratio.pdf", format='pdf')
print(f"[✓] Saved: {output_base}_ratio.pdf and .svg")