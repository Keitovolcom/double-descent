import pandas as pd
import matplotlib.pyplot as plt

# 日本語フォントの設定
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 20

# データセットの読み込み
df = pd.read_csv('vizualize/ACML/EMNIST/irekawari_no/matchno_noise.csv')

# ───── other_label を計算 ─────
# 端数誤差で 0 未満になるのを防ぐため、clip(0, 1) を掛けています
# df['other_label'] = (1.0 - df['noisy_label'] - df['clean_label2']).clip(lower=0)
df['other_label'] = (1.0- df['clean_label2']).clip(lower=0)

# グラフの作成
plt.figure(figsize=(10, 7))

# clean ラベル
plt.plot(df['epoch'], df['clean_label2'],
         c='blue', label='clean label', alpha=1.0)
# plt.fill_between(df['epoch'],
#                  df['clean_label2'] - df['clean_std'],
#                  df['clean_label2'] + df['clean_std'],
#                  color='blue', alpha=0.3)

# other ラベル
plt.plot(df['epoch'], df['other_label'],
         c='green', label='other label', alpha=1.0)

# 必要なら noisy も表示したい場合はコメントを外してください
# plt.plot(df['epoch'], df['noisy_label'],
#          c='red', label='noisy label', alpha=1.0)

# 目印となる縦線
for epoch in [27, 55, 120]:
    plt.axvline(x=epoch, color='k', linestyle='-', linewidth=0.9)

# 軸設定
plt.xscale("log")
plt.ylim(-0.04, 1.04)
plt.xlabel('epoch')
plt.ylabel('match ratio')

# 凡例・保存
plt.legend()
plt.tight_layout()
plt.savefig('vizualize/7_11_temporal/match_ratio/7_21random_match_no_noise_vlines.png')

print("Data has been visualized and saved as "
      "'vizualize/7_11_temporal/match_ratio/7_21random_match_no_noise_vlines.png'.")
