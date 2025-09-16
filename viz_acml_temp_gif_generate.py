#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate alpha–probability GIFs for every pair in a given directory tree.

Directory layout (example) :
root/
└── no_noise/
    ├── pair1/5_5/csv/epoch_*.csv
    ├── pair1/5_5/fig_and_log/      <-- GIF is saved here
    ├── pair2/4_4/csv/epoch_*.csv
    └── …

Usage:
    python generate_all_alpha_gifs.py /path/to/no_noise \
        --epoch-stride 5 --start-epoch 1 --end-epoch 150 --workers 4
"""
import argparse
import glob
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.animation import FuncAnimation, PillowWriter


# ----------------------------------------------
# 既存コード（必要なら import でも OK）
# ----------------------------------------------
def get_highlight_labels_from_path(data_dir):
    """
    data_dir: 例 → alpha_test/cifar10/0.2/64/noise/pair0001/7_3/csv
    """
    label_dir = os.path.basename(os.path.dirname(data_dir))  # "7_3"
    label1, label2 = map(int, label_dir.split("_"))
    return label1, label2


def generate_alpha_probabilities_gif(data_dir, output_path, targets='combined', epoch_stride=1,
                                     start_epoch=1, end_epoch=300):
    """
    指定した data_dir 内の epoch_*.csv を読み込み、alpha と予測確率の推移を GIF 保存する。
    ハイライトはディレクトリ名にあるラベル2つ（青・赤）に基づく。
    
    Parameters:
    - data_dir: CSVが入っているディレクトリ
    - output_path: 出力するGIFのパス
    - targets: ラベル指定（未使用）
    - epoch_stride: 何エポックごとに描画するか（例：5なら5エポックごと）
    - start_epoch: 開始エポック（Noneなら最初のエポック）
    - end_epoch: 終了エポック（Noneなら最後のエポック）
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "epoch_*.csv")))
    if not csv_files:
        print(f"[!] No CSV files in {data_dir}")
        return

    data = {}
    for f in csv_files:
        epoch = int(os.path.basename(f).split("_")[1].split(".")[0])
        data[epoch] = pd.read_csv(f)

    all_epochs = sorted(data.keys())

    # 指定された範囲でエポックをフィルター
    filtered_epochs = [e for e in all_epochs
                       if (start_epoch is None or e >= start_epoch) and
                          (end_epoch is None or e <= end_epoch)]

    epochs = filtered_epochs[::epoch_stride]
    if not epochs:
        print("[!] No epochs match the given range and stride.")
        return

    alpha_values = data[epochs[0]]['alpha']
    label1, label2 = get_highlight_labels_from_path(data_dir)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    other_colors = cycle(['gray', 'purple', 'green', 'orange', 'cyan', 'brown'])
    lines = []

    df_first = data[epochs[0]]

    for t in range(100):
        col = f'prob_{t}'
        if col in df_first.columns:
            if t == label1:
                color = 'blue'
                lw = 2.5
                alpha = 1.0
            elif t == label2:
                color = 'red'
                lw = 2.5
                alpha = 1.0
            else:
                color = next(other_colors)
                lw = 0.8
                alpha = 0.5
            line, = ax.plot(alpha_values, df_first[col], color=color, linewidth=lw, alpha=alpha)
            lines.append((t, line))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(-0.5, 1.6, 0.5))
    ax.set_title(f"Alpha Interpolation (Epoch {epochs[0]})")

    def update(epoch):
        ax.set_title(f"Alpha Interpolation (Epoch {epoch})")
        df = data[epoch]
        for t, line in lines:
            col = f'prob_{t}'
            if col in df.columns:
                line.set_ydata(df[col])
        return [line for _, line in lines]

    anim = FuncAnimation(fig, update, frames=epochs, interval=200, blit=True)
    anim.save(output_path, writer=PillowWriter(fps=8))
    plt.close()
    print(f"[✓] Saved alpha probability GIF to {output_path}")


# ----------------------------------------------
# 新規: すべての pair を走査して GIF を生成
# ----------------------------------------------
def process_pair(
    csv_dir,
    epoch_stride,
    start_epoch,
    end_epoch,
    overwrite,
):
    # csv_dir は …/pairX/<l1>_<l2>/csv
    base_dir = os.path.dirname(csv_dir)           # …/pairX/<l1>_<l2>
    fig_dir = os.path.join(base_dir, "fig_and_log")
    gif_path = os.path.join(fig_dir, "alpha_plot.gif")

    if not overwrite and os.path.exists(gif_path):
        print(f"[skip] {gif_path} already exists")
        return

    try:
        generate_alpha_probabilities_gif(
            data_dir=csv_dir,
            output_path=gif_path,
            epoch_stride=epoch_stride,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
        )
    except Exception as e:
        print(f"[!] Failed for {csv_dir}: {e}")


def find_all_csv_dirs(root_dir):
    """
    root_dir 以下で *pair*/<label>_<label>/csv をすべて探す
    """
    pattern = os.path.join(root_dir, "pair*", "*_*", "csv")
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Generate alpha-probability GIFs for every pair under a directory"
    )
    parser.add_argument("root_dir", help="pair ディレクトリを束ねるルート (no_noise など)")
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=150)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--overwrite", action="store_true", help="既存 GIF を作り直す")
    args = parser.parse_args()

    csv_dirs = find_all_csv_dirs(args.root_dir)
    if not csv_dirs:
        print("[!] No csv/ directories found")
        return

    print(f"[Info] Found {len(csv_dirs)} csv directories")

    # 並列実行
    with mp.Pool(processes=args.workers) as pool:
        pool.map(
            partial(
                process_pair,
                epoch_stride=args.epoch_stride,
                start_epoch=args.start_epoch,
                end_epoch=args.end_epoch,
                overwrite=args.overwrite,
            ),
            csv_dirs,
        )


if __name__ == "__main__":
    main()

# python viz_acml_temp_gif_generate.py /workspace/alpha_test/emnist_digits/0.2/8/noise \
#   --epoch-stride 1 --start-epoch 1 --end-epoch 150 --workers 4 --overwrite