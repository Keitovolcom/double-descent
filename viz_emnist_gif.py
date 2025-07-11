import os
import glob
from typing import List
from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------------------------------------------
# 提供された関数
# ------------------------------------------------------------------

def get_highlight_labels_from_path(data_dir):
    """
    data_dir からハイライトすべき2つのラベルを取得する。
    例: "alpha_test/cifar10/0.2/64/noise/pair0001/7_3/csv" -> (7, 3)
    """
    label_dir = os.path.basename(os.path.dirname(data_dir))  # "7_3"
    label1, label2 = map(int, label_dir.split("_"))
    # return label1, label2
    return label2,label1

def get_sample_dirs(base_dir: str) -> List[str]:
    """
    base_dir 以下の 2 階層下で "fig_and_log" を含むディレクトリを返す。
    例: base_dir = '.../noise'
        -> ['.../noise/pair0001/7_3', '.../noise/pair0002/1_8', ...]
    """
    sample_dirs = []
    if not os.path.exists(base_dir):
        print(f"[!] ベースディレクトリが見つかりません: {base_dir}")
        return []
    
    # 1階層目 (例: pair0001)
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if not os.path.isdir(d_path):
            continue
        
        # 2階層目 (例: 7_3)
        for sub in os.listdir(d_path):
            sub_path = os.path.join(d_path, sub)
            # このディレクトリ内に 'fig_and_log' があるかチェック
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "fig_and_log")):
                sample_dirs.append(sub_path)
    return sample_dirs

def generate_alpha_probabilities_gif(data_dir, output_path, epoch_stride=5,
                                     start_epoch=1, end_epoch=300):
    """
    指定した data_dir 内の epoch_*.csv を読み込み、alpha と予測確率の推移を GIF 保存する。
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "epoch_*.csv")))
    if not csv_files:
        print(f"[!] CSVファイルが見つかりません: {data_dir}")
        return

    data = {}
    for f in csv_files:
        try:
            epoch = int(os.path.basename(f).split("_")[1].split(".")[0])
            data[epoch] = pd.read_csv(f)
        except (IndexError, ValueError):
            print(f"[!] エポック番号をファイル名から取得できませんでした: {f}")
            continue
    
    if not data:
        print(f"[!] 有効なエポックデータが読み込めませんでした: {data_dir}")
        return

    all_epochs = sorted(data.keys())
    filtered_epochs = [e for e in all_epochs if start_epoch <= e <= end_epoch]
    epochs = filtered_epochs[::epoch_stride]

    if not epochs:
        print("[!] 指定された範囲と間隔に一致するエポックがありません。")
        return

    alpha_values = data[epochs[0]]['alpha']
    label1, label2 = get_highlight_labels_from_path(data_dir)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cmap = plt.get_cmap('tab20')
    other_colors = cycle([cmap(i) for i in range(cmap.N)])
    lines = []
    df_first = data[epochs[0]]

    for t in range(100):  # prob_0 から prob_99 までを想定
        col = f'prob_{t}'
        if col in df_first.columns:
            if t == label1:
                color, lw, zorder, alpha = 'blue', 2.5, 10, 1.0
            elif t == label2:
                color, lw, zorder, alpha = 'red', 2.5, 10, 1.0
            else:
                color, lw, zorder, alpha = next(other_colors), 0.8, 1, 0.5
            line, = ax.plot(alpha_values, df_first[col], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
            lines.append((t, line))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(-0.5, 1.6, 0.5))
    # ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"Alpha Interpolation (Epoch {epochs[0]})")

    def update(epoch):
        ax.set_title(f"Alpha Interpolation (Epoch {epoch})")
        df = data[epoch]
        for t, line in lines:
            line.set_ydata(df[f'prob_{t}'])
        return [line for _, line in lines]

    anim = FuncAnimation(fig, update, frames=epochs, interval=200, blit=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"[✓] GIFを保存しました: {output_path}")

# ------------------------------------------------------------------
# メイン関数の実装
# ------------------------------------------------------------------

def main():
    """
    メイン関数。
    設定項目で指定されたベースディレクトリ内の各サンプルに対してGIF画像を生成します。
    """
    # --- ⚙️ 設定項目 ---
    # ここで処理したいベースディレクトリのパスを指定してください
    BASE_DIR = './alpha_test/emnist_digits/0.2/8/noise'
    
    # GIF生成のオプション
    EPOCH_STRIDE = 1    # GIFに含めるエポックの間隔
    START_EPOCH = 1     # GIF生成の開始エポック
    END_EPOCH = 150     # GIF生成の終了エポック
    # --- 設定はここまで ---

    print(f"[*] サンプルディレクトリを検索中: {BASE_DIR}")
    sample_dirs = get_sample_dirs(base_dir=BASE_DIR)
    
    if not sample_dirs:
        print(f"[!] 対象ディレクトリが見つかりませんでした: {BASE_DIR}")
        return

    print(f"[*] {len(sample_dirs)}個のサンプルディレクトリが見つかりました。")

    for sample_dir in sample_dirs:
        csv_dir = os.path.join(sample_dir, 'csv')
        gif_path = os.path.join(sample_dir, 'fig_and_log', 'alpha_plot.gif')
        
        print(f"\n--- 処理開始: {sample_dir} ---")
        
        if not os.path.exists(csv_dir):
            print(f"[!] CSVディレクトリが見つからないためスキップします: {csv_dir}")
            continue
            
        generate_alpha_probabilities_gif(
            data_dir=csv_dir, 
            output_path=gif_path,
            epoch_stride=EPOCH_STRIDE,
            start_epoch=START_EPOCH,
            end_epoch=END_EPOCH
        )

if __name__ == '__main__':
    main()