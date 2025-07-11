import pandas as pd
import matplotlib.pyplot as plt

# 日本語フォントの設定
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 20

# データセットの読み込み
df = pd.read_csv('vizualize/ACML/EMNIST/irekawari/matchnoise.csv')

# グラフの作成
plt.figure(figsize=(10, 7))

# cleanデータの平均と標準偏差をプロット
plt.plot(df['epoch'], df['clean_label2'], c='blue', label='clean label', alpha=1.0)
plt.fill_between(df['epoch'],
                 df['clean_label2'] - df['clean_std'],
                 df['clean_label2'] + df['clean_std'],
                 color='blue', alpha=0.3)

# noisyデータの平均と標準偏差をプロット
plt.plot(df['epoch'], df['noisy_label'], c='red', label='noise label', alpha=1.0)
plt.fill_between(df['epoch'],
                 df['noisy_label'] - df['noisy_std'],
                 df['noisy_label'] + df['noisy_std'],
                 color='red', alpha=0.3)

# 指定されたepochに縦線を追加
epochs_to_mark = [27, 55, 120]
for epoch in epochs_to_mark:
    plt.axvline(x=epoch, color='k', linestyle='-', linewidth=0.9)


# X軸を対数スケールに設定
plt.xscale("log")
plt.ylim(-0.05,1.05)
# タイトルとラベルの設定
plt.xlabel('epoch')
plt.ylabel('match ratio')

# 凡例の表示
plt.legend()

# グラフを画像として保存
plt.savefig('matchnoise_noise_vlines.png')

print("Data has been visualized and saved as 'matchnoise_noise_vlines.png'.")