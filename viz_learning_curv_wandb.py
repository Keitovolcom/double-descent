import wandb
import pandas as pd
import matplotlib.pyplot as plt

# ===== 設定 =====
entity = "dsml-kernel24"
project = "kobayashi_emnist"
target_run_name = "save_model/emnist_digits/noise_0.2/use_mixup_True_alpha8.0_test_seed_42width64_cnn_5layers_cus_emnist_digits_variance0_combined_lr0.01_batch128_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/csv/training_metrics.csv"
output_base = "/workspace/vizualize/ACML/emnist_learning_curve"

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
    "train_error_noisy"
])
df = pd.DataFrame(history)
df = df.sort_values("epoch")

# ===== Plot設定 =====
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 20

epochs = df["epoch"]
test_error = df["test_error"]/100
train_error = df["train_error_total"]/100
train_error_noisy = df["train_error_noisy"]/100
train_error_clean = df["train_error_clean"]/100

vertical_epochs = []  # 縦線を引きたいエポック（必要に応じて）

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)

# ===== 上段: Test vs Train error =====
ax1.plot(epochs, test_error, label="Test", color="red", linewidth=3)
ax1.plot(epochs, train_error, label="Train", color="blue", linewidth=3)
for epoch in vertical_epochs:
    ax1.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

ax1.set_xscale("log")
ax1.set_ylabel("Error (%)", fontsize=30)
ax1.set_ylim(-0.1, 103)
ax1.set_xlim(0.9, 2000)
ax1.legend(loc="upper right", fontsize=30)
ax1.yaxis.grid(True, linestyle="--", linewidth=0.3)
ax1.xaxis.grid(True, linestyle="--", linewidth=0.3)
ax1.tick_params(axis="both", labelsize=30, labelbottom=False)

# ===== 下段: Noisy vs Clean Train error =====
ax2.plot(epochs, train_error_noisy, label="Noisy", color="darkblue", linewidth=3)
ax2.plot(epochs, train_error_clean, label="Clean", color="cyan", linewidth=3)
for epoch in vertical_epochs:
    ax2.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

ax2.set_xscale("log")
ax2.set_xlabel("Epoch", fontsize=30)
ax2.set_ylabel("Train Error (%)", fontsize=30)
ax2.set_ylim(-0.01, 1.0)
ax2.set_xlim(0.9, 1000)
ax2.legend(loc="upper right", fontsize=30)
ax2.yaxis.grid(True, linestyle="--", linewidth=0.3)
ax2.xaxis.grid(True, linestyle="--", linewidth=0.3)
ax2.tick_params(axis="both", labelsize=30)

# ===== 保存 =====
plt.savefig(f"{output_base}.svg", format='svg')
plt.savefig(f"{output_base}.pdf", format='pdf')
print(f"[✓] Saved: {output_base}.pdf and .svg")
