import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ======== W&B からデータ取得（ここは変更なし） ========
entity = "dsml-kernel24"
project = "kobayashi_emnist"
target_run_name = "cnn_5layers_width8_emnist_digits_lr0.01_batch_size128_epoch1000_LabelNoiseRate0.2_Optimsgd_momentum0.0"
output_base = "/workspace/vizualize/miru_oral/ratio_single"

api = wandb.Api()
run_id = next(r.id for r in api.runs(f"{entity}/{project}") if r.name == target_run_name)
run     = api.run(f"{entity}/{project}/{run_id}")

history = run.history(keys=[
    "epoch",
    "test_error",
    "train_error_total",
    "train_error_clean",
    "train_error_noisy",
    "test_loss",
    "train_loss"
])

df = (pd.DataFrame(history)
        .dropna(subset=["epoch"])
        .sort_values("epoch"))

# ======== データ整形 ========
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]   = 20

epochs            = df["epoch"]
test_error        = df["test_error"]        / 100
train_error       = df["train_error_total"] / 100
train_error_noisy = df["train_error_noisy"] / 100
train_error_clean = df["train_error_clean"] / 100
train_loss        = df["train_loss"]
test_loss         = df["test_loss"]

vertical_epochs = [1, 27, 55, 120]

# ======== 図と軸を“手動”で生成（subplot を使わない） ========
fig = plt.figure(figsize=(10, 8), constrained_layout=True)
# add_axes([left, bottom, width, height]) ですべて 0–1 の比率で指定
ax1 = fig.add_axes([0.12, 0.12, 0.80, 0.80])   # メイン軸
ax2 = ax1.twinx()                               # 右 Y 軸

# ----- Error 曲線 -----
ax1.plot(epochs, test_error,  label="test",          color="red",  linewidth=3)
ax1.plot(epochs, train_error, label="train",   color="blue", linewidth=3)

# ----- Noisy / Clean -----
ax1.plot(epochs, train_error_noisy, label="train noisy", color="blue",
         linewidth=3, linestyle="--")
ax2.plot(epochs, train_error_clean, label="train clean", color="blue",
         linewidth=3)

# ----- 軸設定 -----
ax1.set_xscale("log")
ax1.set_xlim(0.9, 1000)
ax1.set_ylim(-0.02, 1.02)
ax2.set_ylim(-0.00235, 0.11)

ax1.set_xlabel("epoch", fontsize=30)
ax1.set_ylabel("error", fontsize=30)
ax2.set_ylabel("train clean error", fontsize=30)

ax1.set_xticks([1, 10, 100, 1000])
ax1.tick_params(axis="both", labelsize=30)
ax2.tick_params(axis="y",    labelsize=30)

# ----- 縦線 -----
for ep in vertical_epochs:
    ax1.axvline(x=ep, color="black", linestyle="-", linewidth=0.9, zorder=0)

# ----- 凡例 (左右軸を結合) -----
lines,  labels  = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2,
           loc="upper right", fontsize=30, frameon=False)

# ======== 保存 ========
fig.savefig(f"{output_base}.svg")
fig.savefig(f"{output_base}.pdf")
print(f"[✓] Saved {output_base}.svg / .pdf")
