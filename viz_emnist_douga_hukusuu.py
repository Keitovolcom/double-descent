import matplotlib
matplotlib.use('Agg')  # 非表示バックエンド（Agg）を利用して描画負荷を低減
import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from itertools import cycle
from matplotlib.lines import Line2D  # ダミー凡例作成用
import numpy as np
import cv2  # OpenCVを利用

# フォントや図サイズの設定
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 200

def get_highlight_labels_from_path(data_dir):
    parts = re.split(r"[\\/]", data_dir)
    for part in reversed(parts):
        if re.fullmatch(r"\d+_\d+", part):
            a, b = map(int, part.split("_"))
            return a, b
    raise ValueError(f"ハイライト用のラベルがパスから見つかりませんでした: {data_dir}")

def load_data(data_dir, targets='digit'):
    csv_dir = os.path.join(data_dir, "csv")
    files = sorted(glob.glob(os.path.join(csv_dir, "epoch_*.csv")))
    data = {}
    for file in files:
        epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
        data[epoch] = pd.read_csv(file)
    return data

# 並列表示用関数
def target_plot_probabilities_parallel(
    data_dirs,                     # [左ディレクトリ, 右ディレクトリ]
    targets='digit',
    video_output="output_parallel.mp4",
    save_dir=None,
    show_legend=True,
    epoch_start=None,
    epoch_end=None,
    epoch_step=None,
    modes=None                   # ['no_noise' or 'noise' for each side]
):
    # データ読み込み
    datas = [load_data(d, targets) for d in data_dirs]
    epochs_list = [sorted(d.keys()) for d in datas]
    # 共通エポック
    epochs = sorted(set(epochs_list[0]) & set(epochs_list[1]))
    if epoch_start is not None:
        epochs = [e for e in epochs if e >= epoch_start]
    if epoch_end is not None:
        epochs = [e for e in epochs if e <= epoch_end]
    if epoch_step and epoch_step > 1:
        epochs = epochs[::epoch_step]
    if not epochs:
        print(f"No CSV files found in specified ranges. Skipping...")
        return

    initial_epoch = epochs[0]
    # ハイライトラベル取得
    hl_labels = [get_highlight_labels_from_path(d) for d in data_dirs]
    dfs_first = [datas[i][initial_epoch] for i in range(2)]

    # 保存先設定
    if save_dir is None:
        save_dir = os.path.join(data_dirs[0], "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)
    parts = [os.path.basename(os.path.dirname(d)) + "_" + os.path.basename(d) for d in data_dirs]
    video_name = f"{parts[0]}__{parts[1]}_{video_output}"
    video_path = os.path.join(save_dir, video_name)

    # Figure と 2 軸作成
    fig, axes = plt.subplots(1, 2, figsize=(plt.rcParams["figure.figsize"][0]*2, plt.rcParams["figure.figsize"][1]))
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    lines = [[], []]
    orig_styles = [[], []]
    # 各軸にラインをセットアップ
    for i, ax in enumerate(axes):
        other_colors = cycle(['green','orange','purple','brown','gray','pink','olive','cyan','lime','navy'])
        alpha_values = dfs_first[i]['alpha']
        hl1, hl2 = hl_labels[i]
        for t in range(10):
            col = f'prob_{t}'
            if col not in dfs_first[i].columns:
                continue
            if t == hl1:
                color, ls, lw, lbl = 'blue', '-', 2.0, 'Clean Label'
            elif t == hl2:
                color, ls, lw, lbl = 'red', '-', 2.0, 'Noisy Label'
            else:
                color, ls, lw, lbl = next(other_colors), '--', 1.0, None
            line, = ax.plot(
                alpha_values,
                dfs_first[i][col],
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=lbl,
                alpha=1.0
            )
            lines[i].append(line)
            orig_styles[i].append((color, ls, lw, 1.0))

        # 軸の設定
        ax.set_ylabel('probability', fontsize=22)
        ax.set_title(f'Epoch {initial_epoch}', fontsize=22)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0.0, 1.0])
        if modes and modes[i] == 'no_noise':
            ax.plot([0.0, 1.0], [0, 0], 'o', color='blue', markersize=10, zorder=5)
        else:
            ax.plot(0, 0, 'o', color='blue', markersize=10, zorder=5)
            ax.plot(1, 0, 'o', color='red', markersize=10, zorder=5)
        ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=30)

        if show_legend:
            clean = Line2D([0], [0], color='blue', linewidth=2.0, label='clean Label')
            noisy = Line2D([0], [0], color='red', linewidth=2.0, label='noisy Label')
            other = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='other labels')
            ax.legend(
                handles=[clean, noisy, other],
                loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0,
                fontsize=16
            )

    # VideoWriter セットアップ
    canvas = fig.canvas
    canvas.draw()
    bgs = [canvas.copy_from_bbox(ax.get_window_extent()) for ax in axes]
    width, height = canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # フレーム更新ループ
    for epoch in epochs:
        for i, ax in enumerate(axes):
            df_e = datas[i][epoch]
            hl1, hl2 = hl_labels[i]
            for t, line in enumerate(lines[i]):
                col = f'prob_{t}'
                line.set_ydata(df_e[col])
                if t == hl1:
                    line.set_color('blue'); line.set_linestyle('-'); line.set_linewidth(2.0); line.set_alpha(1.0)
                elif t == hl2:
                    line.set_color('red'); line.set_linestyle('-'); line.set_linewidth(2.0); line.set_alpha(1.0)
                else:
                    oc, ols, olw, oa = orig_styles[i][t]
                    line.set_color(oc); line.set_linestyle(ols); line.set_linewidth(olw); line.set_alpha(oa)
            ax.set_title(f'Epoch {epoch}', fontsize=22)
            canvas.restore_region(bgs[i])
            for line in lines[i]:
                ax.draw_artist(line)
            canvas.blit(ax.get_window_extent())

        canvas.flush_events()
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)
        canvas.draw()
        bgs = [canvas.copy_from_bbox(ax.get_window_extent()) for ax in axes]

    writer.release()
    print(f"Parallel video saved as {video_path}")
    plt.close(fig)

if __name__ == "__main__":
    left_dir = "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/noise/pair1/9_8"
    right_dir = "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/no_noise/pair2/7_3"
    save = "vizualize/miru_oral"
    # 左は no_noise (青青)、右は noise (青赤)
    modes = ['no_noise', 'noise']
    target_plot_probabilities_parallel(
        [left_dir, right_dir],
        targets="combined",
        video_output="output_test.mp4",
        save_dir=save,
        show_legend=True,
        epoch_start=1,
        epoch_end=10,
        epoch_step=1,
        modes=modes
    )
