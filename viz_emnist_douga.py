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
from multiprocessing import Process

# フォントや図サイズの設定
plt.rcParams["font.size"] = 30
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 200
plt.rcParams['font.family'] = 'Times New Roman'


def get_highlight_labels_from_path(data_dir):
    """
    data_dir: 例 →
      alpha_test/cifar10/0.2/64/noise/pair0001/7_3/csv
      のような任意の深さのパスから、
      ディレクトリ名が「数字_数字」の形式になっているものを探し、
      (label1, label2) を返す。
    """
    # パスの区切り文字を OS に依存せず正規化
    parts = re.split(r"[\\/]", data_dir)
    # 後ろから「数字_数字」の部分を探す
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
    epoch_step=None,
    mode = None
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
    if not epochs:
        print(f"No CSV files found in {data_dir} within the specified epoch range. Skipping...")
        return

    initial_epoch = epochs[0]
    df_first = data[initial_epoch]

    # --- ハイライト用ラベルをパスから取得 ---
    hl_label1, hl_label2 = get_highlight_labels_from_path(data_dir)

    if save_dir is None:
        save_dir = os.path.join(data_dir, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)
    parent_name = os.path.basename(os.path.dirname(data_dir))
    current_name = os.path.basename(data_dir)
    video_file_name = f"{parent_name}_{current_name}_{video_output}"
    video_output_path = os.path.join(save_dir, video_file_name)

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

        if t == hl_label1:
            color, linestyle, linewidth, label = 'blue', '-', 2.0, 'clean label'
        elif t == hl_label2:
            color, linestyle, linewidth, label = 'red', '-', 2.0, 'noisy label'
        else:
            color, linestyle, linewidth, label = next(other_colors), '--', 1.0, None

        line, = ax.plot(
            alpha_values,
            df_first[column_name],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            alpha=1.0
        )
        lines.append(line)
        orig_styles.append((color, linestyle, linewidth, 1.0))

    ax.set_ylabel('probability', fontsize=22)
    ax.set_title(f'probability for Epoch {initial_epoch}', fontsize=22)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 1.0])
    if mode == "no_noise":
        ax.plot([0.0, 1.0], [0.0, 0.0], "o",
                    color="blue", markersize=20, zorder=5)
    else:  # mode == "noise"
        ax.plot(0.0, 0.0, "o", color="blue",
                    markersize=20, zorder=5)
        ax.plot(1.0, 0.0, "o", color="red",
                    markersize=20, zorder=5)



    ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=30)
    if show_legend:
        clean_line = Line2D([0], [0], color='blue', linewidth=2.0, label='clean Label')
        noisy_line = Line2D([0], [0], color='red', linewidth=2.0, label='noisy Label')
        other_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='other labels')
        ax.legend(
            handles=[clean_line, noisy_line, other_line],
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0, fontsize=16
        )
        fig.subplots_adjust(right=0.8)

    # update 関数：各 epoch でもハイライトはパス由来のまま、Y データだけ更新
    def update(epoch):
        df_epoch = data[epoch]
        for t, line in enumerate(lines):
            column_name = f'prob_{t}'
            line.set_ydata(df_epoch[column_name])
            # スタイルは常にオリジナル or ハイライトのまま
            if t == hl_label1:
                line.set_color('blue')
                line.set_linestyle('-')
                line.set_linewidth(2.0)
                line.set_alpha(1.0)
            elif t == hl_label2:
                line.set_color('red')
                line.set_linestyle('-')
                line.set_linewidth(2.0)
                line.set_alpha(1.0)
            else:
                oc, ols, olw, oa = orig_styles[t]
                line.set_color(oc)
                line.set_linestyle(ols)
                line.set_linewidth(olw)
                line.set_alpha(oa)
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


def run_in_subprocess(d, save_dir,mode):
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
        epoch_end=10,
        epoch_step=1
    )


def target_plot_probabilities_parallel(
    data_dirs,                  # [left_dir, right_dir]
    targets='combined',
    video_output="parallel_output.mp4",
    save_dir=None,
    use_opencv=True,
    show_legend=True,
    epoch_start=None,
    epoch_end=None,
    epoch_step=None,
    modes=('no_noise', 'noise'),  # モードもリストで渡す
    fps=5
):
    """
    data_dirs: リスト（長さ2）で、左右に並べたい data_dir を指定。
    modes: 左右それぞれの mode ("no_noise" or "noise") を同じ長さで指定。
    """
    # --- データ読込み ---
    for i, d in enumerate([left_dir, right_dir]):
        epochs = sorted(load_data(d).keys())
        print(f"dir[{i}] = {d!r}, found epochs: {epochs[:5]} ... total {len(epochs)}")
    datas = [load_data(d, targets) for d in data_dirs]
    # 共通のエポックだけ使う
    epochs_sets = [set(d.keys()) for d in datas]
    common_epochs = sorted(epochs_sets[0].intersection(epochs_sets[1]))
    # 範囲・ステップ制限
    if epoch_start is not None:
        common_epochs = [e for e in common_epochs if e >= epoch_start]
    if epoch_end   is not None:
        common_epochs = [e for e in common_epochs if e <= epoch_end]
    if epoch_step and epoch_step > 1:
        common_epochs = common_epochs[::epoch_step]
    if not common_epochs:
        print("No common epochs to process. Skipping.")
        return

    # --- ハイライトラベル取得 ---
    highlights = [get_highlight_labels_from_path(d) for d in data_dirs]

    # --- 保存先・ファイル名設定 ---
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    base_names = [os.path.basename(d) for d in data_dirs]
    video_path = os.path.join(save_dir, f"{base_names[0]}__{base_names[1]}_{video_output}")

    # --- Figure・Axes 準備 ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.1, bottom=0.2)

    all_lines = []        # [[lines for left], [lines for right]]
    all_styles = []       # [[orig_styles for left], [orig_styles for right]]

    # 各サブプロット初期描画
    for idx, ax in enumerate(axes):
        initial = common_epochs[0]
        df0 = datas[idx][initial]
        alpha_vals = df0['alpha']
        hl1, hl2 = highlights[idx]
        lines, styles = [], []
        other_colors = cycle(['green','orange','purple','brown','gray','pink','olive','cyan','lime','navy'])
        for t in range(10):
            col = f'prob_{t}'
            if col not in df0.columns: continue
            if t == hl1:
                c, ls, lw, lbl = 'blue', '-', 6.0, 'clean label'
            elif t == hl2:
                c, ls, lw, lbl = 'red', '-', 6.0, 'noisy label'
            else:
                c, ls, lw, lbl = next(other_colors), '--', 3.0, None
            ln, = ax.plot(alpha_vals, df0[col],
                          color=c, linestyle=ls, linewidth=lw, label=lbl)
            lines.append(ln)
            styles.append((c, ls, lw, 1.0))
        ax.set_title(f"{base_names[idx]} (epoch {initial})", fontsize=80)
        ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0.0,1.0]); ax.set_xticklabels([r'$x_0$',r'$x_1$'])
        if idx ==0:
            ax.set_ylabel("probability",fontsize=50)
            ax.set_yticks([0.0, 0.5, 1.0],labelsize=50)
        else:
            # 右プロットは y 軸目盛の「文字だけ」消す
            ax.tick_params(axis='y', labelleft=False)

        # モードに応じたマーキング
        if modes[idx]=="no_noise":
            ax.plot([0,1],[0,0],"o",color="blue",ms=30)
        else:
            ax.plot(0,0,"o",color="blue",ms=30)
            ax.plot(1,0,"o",color="red",ms=30)
        # if show_legend and idx==1:
        #     correct_line = Line2D([0], [0], color='blue', linewidth=2.0, label='correct label')
        #     wrong_line   = Line2D([0], [0], color='red',  linewidth=2.0, label='noisy label')
        #     other_line   = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='other label')
        #     ax.legend(
        #         handles=[correct_line, wrong_line, other_line],
        #         loc="center left", bbox_to_anchor=(1.01, 0.5),
        #         borderaxespad=0, fontsize=30
        #     )
        fig.subplots_adjust(right=0.8)
        all_lines.append(lines)
        all_styles.append(styles)

    # 動画ライター準備
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # OpenCV VideoWriter は (width, height) の順
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    # 更新関数
    def update_subplot(idx, epoch):
        df = datas[idx][epoch]
        hl1, hl2 = highlights[idx]
        for t, ln in enumerate(all_lines[idx]):
            col = f'prob_{t}'
            ln.set_ydata(df[col])
            # スタイル固定
            if t==hl1:
                ln.set_color('blue'); ln.set_linestyle('-'); ln.set_linewidth(6)
            elif t==hl2:
                ln.set_color('red'); ln.set_linestyle('-'); ln.set_linewidth(6)
            else:
                oc, ols, olw, oa = all_styles[idx][t]
                ln.set_color(oc); ln.set_linestyle(ols); ln.set_linewidth(olw)
        axes[idx].set_title(f"epoch {epoch}", fontsize=60)

    # 各エポックでフレームを生成
    for epoch in common_epochs:
        for i in (0,1):
            update_subplot(i, epoch)
        fig.canvas.draw()
        # バッファ取得 → OpenCV 書き込み
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    video_writer.release()
    plt.close(fig)
    print(f"Parallel video saved to {video_path}")

if __name__ == "__main__":
    # data = "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/noise/pair1/9_8"
    # save="vizualize/miru_oral"
    # mode = "noise"
    # run_in_subprocess(d=data,save_dir=save,mode="noise")

    right_dir = "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/noise/pair20/2_6"
    left_dir = "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/no_noise/pair21/0_0"
    target_plot_probabilities_parallel(
        [left_dir, right_dir],
        video_output="phase3_ver2.mp4",
        save_dir="vizualize/miru_oral",
        show_legend=True,
        epoch_start=55,
        epoch_end=300,
        epoch_step=1,
        modes=('no_noise','noise'),
        fps=5
    )
