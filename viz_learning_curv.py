import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# !!!注意!!! 以下のファイルパスを、環境に合わせて修正してください
file_path = "save_model/Colored_EMSNIT/noise_rate=0_sigma=0/seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.0_Optimsgd_Momentum0.0/csv/training_metrics.csv"
file_path="save_model/emnist_digits/noise_0.2/7_14_use_mixup_False_alpha0.0_test_seed_42width64_cnn_5layers_cus_emnist_digits_variance0_combined_lr0.01_batch128_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/csv/training_metrics.csv"
# --- 以下は修正不要です ---

try:
    df = pd.read_csv(file_path)

    # フォント設定
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 20

    # データの抽出
    epochs = df["epoch"]
    # test_error=(100 - df["test_accuracy_total"])/100
    # train_error = (100 - df["train_accuracy_total"])/100
    # train_error_noisy = (1 - df["train_accuracy_noisy"])/100
    # train_error_clean = (1 - df["train_accuracy_clean"])/100
    train_loss = df["train_loss"]
    test_loss = df["test_loss"]
    test_error = df["test_error"]/100
    train_error = df["train_error_total"]/100
    train_error_noisy = df["train_error_noisy"]/100
    train_error_clean = df["train_error_clean"]/100
    # train_loss = df["avg_loss_noisy"]
    # test_loss = df["avg_loss_clean"]

    # 縦線を引くエポックリスト
    vertical_epochs = [30,53,140]

    # プロット作成（3行1列）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 19), constrained_layout=True)

    # 【上段】Test / Total Train Error
    ax1.plot(epochs, test_error, label="test", color="red", linewidth=3)
    ax1.plot(epochs, train_error, label="train", color="blue", linewidth=3)
    for epoch in vertical_epochs:
        ax1.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

    ax1.legend(loc="upper right", fontsize=30)
    ax1.set_xscale("log")
    ax1.set_ylabel("error", fontsize=30)
    ax1.set_ylim(-0.01, 0.41)
    ax1.set_xlim(1, 1000)
    # ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(axis="y", labelsize=30)
    ax1.tick_params(axis="x", labelbottom=False) # x軸のラベルを非表示に

    # 【中段】Train Clean / Noisy Error
    p2, = ax2.plot(epochs, train_error_noisy, label="noisy", color="purple", linewidth=3, linestyle='--')
    ax2.set_ylim(-0.05, 1.01)
    ax2.set_ylabel("train noisy error", fontsize=30,)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.tick_params(axis='y',labelsize=30)

    ax2_twin = ax2.twinx()
    p1, = ax2_twin.plot(epochs, train_error_clean, color="green",label="clean",linewidth=3)
    ax2_twin.set_ylim(-0.01, 0.151)
    ax2_twin.set_ylabel("train clean error", fontsize=30)
    ax2_twin.tick_params(axis='y', labelsize=30)

    for epoch in vertical_epochs:
        ax2.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

    ax2.set_xscale("log")
    ax2.set_xlim(1, 1000)
    ax2.tick_params(axis="x", labelbottom=False) # x軸のラベルを非表示に

    lines = [p1, p2]
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=30)

    # 【下段】Loss (Train / Test)
    ax3.plot(epochs, test_loss, label="noisy loss", color="red", linewidth=3)
    ax3.plot(epochs, train_loss, label="clean loss", color="blue", linewidth=3)
    for epoch in vertical_epochs:
        ax3.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

    ax3.legend(loc="upper right", fontsize=30)
    ax3.set_xscale("log")
    ax3.set_ylabel("loss", fontsize=30)
    ax3.set_xlabel("epoch", fontsize=30)
    ax3.set_xlim(1, 1000)
    ax3.tick_params(axis="both", labelsize=30)

    # 保存
    output_base = "vizualize/7_19/clean_noise_02_emnist_learning_curve"
    plt.savefig(f"{output_base}.svg", format='svg')
    plt.savefig(f"{output_base}.pdf", format='pdf')
    print(f"[✓] Saved: {output_base}.pdf and {output_base}.svg")

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。\n指定されたパス '{file_path}' を確認してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")