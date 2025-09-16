import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import os

api = wandb.Api()

# Replace 'your_entity' and 'your_project' with your actual entity and project names
runs = api.runs('dsml-kernel24/gucci_color_mnist_100_classification_not_balanced_batch_sampler')

# ランの名前で取得するためのユーティリティ関数
def get_run_by_name(runs, label):
    try:
        return next(run for run in runs if run.name == label)
    except StopIteration:
        print(f"Run with label '{label}' not found.")
        return None

# ランを取得してデータを処理するための関数
def get_processed_data(run):
    # 履歴データを取得
    history_df = run.history(samples=1000, x_axis="epoch")
    
    # 'accuracy' を含む列があれば 'error' (100 - accuracy) を生成
    for col in history_df.columns:
        if 'accuracy' in col:
            error_col = col.replace('accuracy', 'error')
            history_df[error_col] = 100 - pd.to_numeric(history_df[col], errors='coerce')
    
    return history_df

# (noise, var) ごとにランを取得して格納する辞書
label_noises={0}
variances={0,10000}
widths={4}
runs_dict = {}
for noise in label_noises:
  for var in variances:
    for width in widths:
      print(noise, var, width)
      label = f'cnn_5layers_distribution_colored_emnist_combined_lr0.01_batch256_epoch1000_LabelNoiseRate{noise}_Optimsgd_cleanw1.0_noisew1.0_variance{var}_width{width}_seed42'
      runs_dict[(noise, var, width)] = get_run_by_name(runs, label)

# データを処理した結果を格納する辞書
processed_data = {}
for (noise, var, width), run in runs_dict.items():
    if run is not None:
        processed_data[(noise, var, width)] = get_processed_data(run)

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.size'] = 16
plt.rcParams['figure.facecolor'] = 'white' # 背景色を白に

fig, axes = plt.subplots(3, 4, figsize=(1.618*4*2.4, 4*2), dpi=200)
variances = [0, 1000, 3612, 10000]

for i, ax in enumerate(axes[0]):
    data = processed_data.get((0.0, variances[i], 8))
    data = data.iloc[1:]
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.plot(data['epoch'], data["test_error"], linewidth=1.5, color='g', label="Color and Digit")
    ax.plot(data['epoch'], data["test_error_color_total"], linewidth=1.5, color='r', label="Color")
    ax.plot(data['epoch'], data["test_error_digit_total"], linewidth=1.5, color='b', label="Digit")
    ax.grid(axis='x', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xscale('log')
    variance = "10^3"
    variance = "0" if i == 0 else variance
    variance = "10^{3.5}" if i == 2 else variance
    variance = "10^4" if i == 3 else variance
    ax.set_title(f"$\sigma^2={variance}$")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks(np.linspace(0, 100, 5))
    # ax.set_xlim(-2, 300)
    # ax.set_yscale('symlog')
    ax.set_ylim(0, 100)
    # ax.minorticks_off(axis='y')
    # ax.minorticks_on()
    if i == 0:
        ax.set_ylabel("Test Error (%)")
    ax.tick_params(labelbottom=False, labelleft=i==0)

for i, ax in enumerate(axes[1]):
    data = processed_data.get((0.0, variances[i], 8))
    data = data.iloc[1:]
    ax.plot(data['epoch'], data["train_error"], linewidth=1.5, color='g', label="Color and Digit")
    ax.plot(data['epoch'], data["train_error_color_total"], linewidth=1.5, color='r', label="Color")
    ax.plot(data['epoch'], data["train_error_digit_total"], linewidth=1.5, color='b', label="Digit")
    # ax.minorticks_off()
    # ax.minorticks_on()
    ax.grid(axis='x', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xscale('log')
    # ax.set_yscale('symlog')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks(np.linspace(0, 100, 5))
    # ax.set_xlim(-2, 300)
    ax.set_ylim(0, 100)
    if i == 0:
        ax.set_ylabel("Train Error (%)")
    ax.tick_params(labelbottom=False, labelleft=i==0)

for i, ax in enumerate(axes[2]):
    data = processed_data.get((0.0, variances[i], 8))
    data = data.iloc[1:]
    ax.plot(data['epoch'], data["train_loss"], linewidth=1.5, label="Train Loss", color="b")
    ax.plot(data['epoch'], data["test_loss"], linewidth=1.5, label="Test Loss", color="r")
    # ax.minorticks_off()
    # ax.minorticks_on()
    ax.grid(axis='x', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks(np.linspace(0, 20, 3))
    # ax.set_xlim(-2, 300)
    ax.set_ylim(0, 20)
    if i == 0:
        ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")  # 最下段のみX軸ラベルを表示
    ax.tick_params(labelleft=i==0)

axes[0][0].legend(loc='upper right', fontsize=12)
axes[1][0].legend(loc='upper right', fontsize=12)
axes[2][0].legend(loc='upper right', fontsize=12)

plt.subplots_adjust(hspace=0.15, wspace=0.1)  # 余白を調整
# plt.tight_layout()
plt.savefig("train_test_error_every_variance_width8.pdf", bbox_inches='tight', format='pdf', metadata={})
plt.show()