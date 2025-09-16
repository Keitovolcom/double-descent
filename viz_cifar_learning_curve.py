import os
import pandas as pd
import matplotlib.pyplot as plt
import re
plt.rcParams["font.size"] = 23
plt.rcParams["figure.figsize"] = [12, 9]
plt.rcParams["figure.dpi"] = 400
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_epoch4000_summary(target_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import re

    # 結果を格納するリスト
    summary_data = []

    for subdir in os.listdir(target_dir):
        full_path = os.path.join(target_dir, subdir)
        if not os.path.isdir(full_path):
            continue
        if "epoch4000" not in subdir:
            continue

        # widthを取得
        match = re.search(r"width(\d+)", subdir)
        if not match:
            continue
        width = int(match.group(1))

        # CSVファイル読み込み
        csv_path = os.path.join(full_path, "csv")
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        if not csv_files:
            continue
        df = pd.read_csv(os.path.join(csv_path, csv_files[0]))

        # epoch == 4000 の行のみ抽出
        row = df[df["epoch"] == 4000]
        if row.empty:
            continue

        # メトリクスを収集
        summary_data.append({
            "width": width,
            "test_error": row["test_error"].values[0]/100,
            "train_error_total": row["train_error_total"].values[0]/100,
            "train_error_noisy": row["train_error_noisy"].values[0]/100,
            "train_error_clean": row["train_error_clean"].values[0]/100,
            "train_loss": row["train_loss"].values[0],
            "test_loss": row["test_loss"].values[0]
        })

    # データがなければ終了
    if not summary_data:
        print("No data found for epoch=4000")
        return

    # DataFrameに変換 & widthでソート
    df_summary = pd.DataFrame(summary_data).sort_values("width")

    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1段目：test_error
    axes[0].plot(df_summary["width"], df_summary["test_error"], marker='o', color="red", label="test error",linewidth=2.5)
    axes[0].plot(df_summary["width"], df_summary["train_error_total"], marker='o', color="blue", label="train error",linewidth=2.5)
    axes[0].set_ylabel("error")
    axes[0].set_ylim(-0.01, 1.01)
    axes[0].set_xscale("log")

    axes[0].grid(True)
    axes[0].legend()

    # 2段目：train_error系（青系）
    axes[1].plot(df_summary["width"], df_summary["train_error_noisy"], marker='o', color="purple", label="noisy",linewidth=2.5,linestyle="--")
    axes[1].plot(df_summary["width"], df_summary["train_error_clean"], marker='o', color="green", label="clean",linewidth=2.5)
    axes[1].set_ylabel("train error")
    axes[1].set_ylim(-0.01, 1.01)
    axes[1].set_xscale("log")

    axes[1].grid(True)
    axes[1].legend()

    # 3段目：loss（train = 青, test = 赤）
    axes[2].plot(df_summary["width"], df_summary["train_loss"], marker='o', color="blue", label="train loss",linewidth=2.5)
    axes[2].plot(df_summary["width"], df_summary["test_loss"], marker='o', color="red", label="test loss",linewidth=2.5)
    axes[2].set_xlabel("width")
    axes[2].set_ylabel("loss")
    axes[2].set_ylim(-0.01, 5.01)
    axes[2].set_xscale("log")
    
    axes[2].grid(True)
    axes[2].legend()
    # x軸の目盛りを指定
    xticks = [1, 2, 4, 8, 10, 12, 16, 32, 64]
    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)  # 明示的にラベルを指定
        plt.setp(ax.get_xticklabels(), fontsize=16)  # フォントサイズのみ変更（斜めなし）
    plt.tight_layout()
    output_path = os.path.join(target_dir, "fig", "summary_epoch4000.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved summary figure: {output_path}")
# メイン処理
# 対象ルートディレクトリ
root_dir = "./save_model/cifar10"
target_dir = os.path.join(root_dir, "noise_0.2")
fig_dir = os.path.join(target_dir, "fig")
os.makedirs(fig_dir, exist_ok=True)
plot_epoch4000_summary(target_dir)

# ディレクトリを走査
# for subdir in os.listdir(target_dir):
#     full_path = os.path.join(target_dir, subdir)
#     if not os.path.isdir(full_path):
#         continue
#     if "epoch4000" not in subdir:
#         continue

#     # width を抽出
#     match = re.search(r"width(\d+)", subdir)
#     if not match:
#         continue
#     width = match.group(1)

#     # csvファイルの読み込み
#     csv_path = os.path.join(full_path, "csv")
#     csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
#     if not csv_files:
#         print(f"No CSV found in {csv_path}")
#         continue
#     csv_file_path = os.path.join(csv_path, csv_files[0])

#     df = pd.read_csv(csv_file_path)

#     # プロット
#     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

#     # 1段目：test_error
#     axes[0].plot(df["epoch"], df["test_error"], label="test_error", color="red")
#     axes[0].set_ylabel("Test Error")
#     axes[0].legend()
#     axes[0].grid(True)
#     axes[0].set_xscale("log")
#     axes[0].set_ylim(0, 100)

#     # 2段目：train_error系（青系）
#     axes[1].plot(df["epoch"], df["train_error_total"], label="train_error_total", color="blue")
#     axes[1].plot(df["epoch"], df["train_error_noisy"], label="train_error_noisy", color="skyblue")
#     axes[1].plot(df["epoch"], df["train_error_clean"], label="train_error_clean", color="navy")
#     axes[1].set_ylabel("Train Error")
#     axes[1].legend()
#     axes[1].grid(True)
#     axes[1].set_xscale("log")
#     axes[1].set_ylim(0, 100)

#     # 3段目：loss（train = 青, test = 赤）
#     axes[2].plot(df["epoch"], df["train_loss"], label="train_loss", color="blue")
#     axes[2].plot(df["epoch"], df["test_loss"], label="test_loss", color="red")
#     axes[2].set_xlabel("Epoch (log scale)")
#     axes[2].set_ylabel("Loss")
#     axes[2].legend()
#     axes[2].grid(True)
#     axes[2].set_xscale("log")
#     axes[2].set_ylim(0, 2.5)

#     plt.tight_layout()
#     output_path = os.path.join(fig_dir, f"width{width}.png")
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Saved: {output_path}")
