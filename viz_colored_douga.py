import matplotlib
matplotlib.use('Agg')  # 非表示バックエンド（Agg）を利用して描画負荷を低減

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from itertools import cycle
from matplotlib.lines import Line2D  # ダミー凡例作成用
import numpy as np
import cv2  # OpenCVを利用
from multiprocessing import Process

# フォントや図サイズの設定
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 200

def load_data(data_dir, targets='digit'):
    csv_dir = os.path.join(data_dir, "csv")
    if targets == 'combined':
        files = sorted(glob.glob(os.path.join(csv_dir, "raw_probabilities_epoch_*.csv")))
    else:
        files = sorted(glob.glob(os.path.join(csv_dir, "alpha_log_epoch_*.csv")))
    
    data = {}
    for file in files:
        epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
        data[epoch] = pd.read_csv(file)
    return data

def extract_digit_for_digit_mode(name):
    if len(name) == 1:
        return 0
    else:
        return int(name[0])

def extract_digit_for_color_mode(name):
    return int(name[-1])  # ディレクトリ名の末尾の数字を使用

def get_highlight_targets(data_dir, targets):
    current_dir_name = os.path.basename(data_dir)
    parent_dir_name = os.path.basename(os.path.dirname(data_dir))
    if targets == 'combined':
        parent_val = int(parent_dir_name)
        current_val = int(current_dir_name)
    elif targets == 'digit':
        parent_val = extract_digit_for_digit_mode(parent_dir_name)
        current_val = extract_digit_for_digit_mode(current_dir_name)
    elif targets == 'color':
        parent_val = extract_digit_for_color_mode(parent_dir_name)
        current_val = extract_digit_for_color_mode(current_dir_name)
    else:
        raise ValueError("targets must be 'digit', 'color', or 'combined'.")
    return [parent_val, current_val]

def target_plot_probabilities(
    data_dir, 
    targets='digit', 
    video_output="output.mp4",   # 出力ファイル名の基本部分
    save_dir=None,               # 動画の保存先ディレクトリ
    use_opencv=True,             # OpenCV を使って動画出力するか
    show_legend=True,
    epoch_start=None, 
    epoch_end=None, 
    epoch_step=None
):
    """
    指定した data_dir 内の CSV ファイルからデータを読み込み、各エポックごとにグラフを更新し、
    Blitting および Agg バックエンドを利用して動画ファイルとして出力します。
    
    ラインは blitting で高速更新し、タイトルは毎回通常描画で更新することで、以前のタイトルが重なる問題を防ぎます。
    """
    data = load_data(data_dir, targets)
    epochs = sorted(data.keys())
    if epoch_start is not None:
        epochs = [e for e in epochs if e >= epoch_start]
    if epoch_end is not None:
        epochs = [e for e in epochs if e <= epoch_end]
    if epoch_step is not None and epoch_step > 1:
        epochs = epochs[::epoch_step]
    if len(epochs) == 0:
        print(f"No CSV files found in {data_dir} within the specified epoch range. Skipping...")
        return

    initial_epoch = epochs[0]
    df_first = data[initial_epoch]
    
    if save_dir is None:
        save_dir = os.path.join(data_dir, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)
    parent_name = os.path.basename(os.path.dirname(data_dir))
    current_name = os.path.basename(data_dir)
    video_file_name = f"{parent_name}_{current_name}_{video_output}"
    video_output_path = os.path.join(save_dir, video_file_name)
    
    # --- 初回 epoch での predicted_label と label_match を取得 ---
    initial_pred  = int(df_first['predicted_label'].iloc[0])
    initial_match = bool(df_first['label_match'].iloc[0])

    # グラフ作成
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    lines = []
    orig_styles = []
    other_colors = cycle([
        'green', 'orange', 'purple', 'brown', 
        'gray', 'pink', 'olive', 'cyan', 'lime', 'navy'
    ])

    alpha_values = df_first['alpha']
    # prob_0～prob_9 のループ
    for t in range(10):
        column_name = f'prob_{t}'
        if column_name not in df_first.columns:
            continue

        if t == initial_pred:
            # 正解なら青、誤りなら赤
            color     = 'blue' if initial_match else 'red'
            linestyle = '-'
            linewidth = 2.0
            label     = 'Correct Prediction' if initial_match else 'Incorrect Prediction'
        else:
            color     = next(other_colors)
            linestyle = '--'
            linewidth = 1.0
            label     = None

        line, = ax.plot(
            alpha_values, df_first[column_name],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            alpha=1.0
        )
        lines.append(line)
        orig_styles.append((color, linestyle, linewidth, 1.0))

    ax.set_ylabel('Probability', fontsize=22)
    ax.set_title(f'Probability for Epoch {initial_epoch}', fontsize=22)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r'$x_0 \in X_C$', r'$x_1 \in X_C$'], fontsize=30)
    if show_legend:
        correct_line = Line2D([0], [0], color='blue', linewidth=2.0, label='Correct Prediction')
        wrong_line   = Line2D([0], [0], color='red',  linewidth=2.0, label='Incorrect Prediction')
        other_line   = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Other Classes')
        ax.legend(
            handles=[correct_line, wrong_line, other_line],
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0, fontsize=16
        )
        fig.subplots_adjust(right=0.8)

    # update 関数：各 epoch ごとに predicted_label／label_match を読み出してラインスタイルを更新
    def update(epoch):
        df_epoch = data[epoch]
        pred  = int(df_epoch['predicted_label'].iloc[0])
        match = bool(df_epoch['label_match'].iloc[0])

        for t, line in enumerate(lines):
            column_name = f'prob_{t}'
            # Y データは常に更新
            line.set_ydata(df_epoch[column_name])

            # ハイライト更新
            if t == pred:
                line.set_color('blue' if match else 'red')
                line.set_linestyle('-')
                line.set_linewidth(2.0)
                line.set_alpha(1.0)
            else:
                orig_color, orig_ls, orig_lw, orig_alpha = orig_styles[t]
                line.set_color(orig_color)
                line.set_linestyle(orig_ls)
                line.set_linewidth(orig_lw)
                line.set_alpha(orig_alpha)

        return lines

    # Blitting の準備
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    width, height = fig.canvas.get_width_height()
    new_width, new_height = width // 2, height // 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (new_width, new_height))

    for epoch in epochs:
        update(epoch)
        fig.canvas.restore_region(background)
        for line in lines:
            ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

        ax.set_title(f'Probability for Epoch {epoch}', fontsize=22)
        fig.canvas.draw_idle()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        video_writer.write(image_resized)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)

    video_writer.release()
    print(f"Video saved as {video_output_path}")
    plt.close(fig)

def run_in_subprocess(d, save_dir):
    """
    各ディレクトリごとに target_plot_probabilities を実行し、
    処理終了後にメモリ解放のためサブプロセスを利用。
    """
    target_plot_probabilities(
        data_dir=d,
        targets="combined",
        video_output="output.mp4",
        save_dir=save_dir,
        use_opencv=True,
        show_legend=True,
        epoch_start=1,
        epoch_end=1000,
        epoch_step=1
    )

def main(base_dir, save_dir=None):
    """
    base_dir 以下の各サブディレクトリ（それぞれに 'csv' フォルダがあるもの）を
    全て処理して動画を生成します。
    save_dir を指定すると、そこにまとめて出力。指定なければ各 data_dir/fig_and_log に出力。
    """
    # 必要モジュールはファイルの先頭でインポート済み：os, glob
    data_dirs = [
        d for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "csv"))
    ]

    for data_dir in sorted(data_dirs):
        print(f"▶ Processing {data_dir}")
        run_in_subprocess(data_dir, save_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <base_dir> [save_dir]")
        sys.exit(1)

    base_dir = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) >= 3 else None

    main(base_dir, save_dir)
