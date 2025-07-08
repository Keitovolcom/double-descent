# ==============================================================================
# EMNIST 専用コード
# 既存のロジックや描画設定を変更せず、EMNIST用に新たに作成したコードです。
# ==============================================================================

# 必要なライブラリ (元のコードと共通)
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.lines import Line2D
import numpy as np

# グラフの基本設定 (元のコードと共通)
plt.rcParams["font.size"] = 25
plt.rcParams["figure.figsize"] = (26, 18)
plt.rcParams["figure.dpi"] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_data_emnist(data_dir):
    """EMNIST専用: csvディレクトリからepoch_*.csvファイルを読み込む"""
    csv_dir = os.path.join(data_dir, "csv")
    data = {}
    files = sorted(glob.glob(os.path.join(csv_dir, "epoch_*.csv")))
    if not files:
        print(f"警告: CSVファイルがディレクトリに見つかりません: {csv_dir}")
    for file in files:
        try:
            epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
            data[epoch] = pd.read_csv(file)
        except (IndexError, ValueError):
            print(f"警告: ファイル名からエポックを抽出できませんでした: {os.path.basename(file)}")
    return data

def get_highlight_targets_emnist(data_dir):
    """EMNIST専用: 'number1_number2'形式のディレクトリ名からハイライト対象を取得"""
    dir_name = os.path.basename(data_dir)
    try:
        parts = dir_name.split('_')
        if len(parts) != 2:
            raise ValueError("ディレクトリ名に '_' が1つではありません。")
        number1 = int(parts[0])
        number2 = int(parts[1])
        print(f"ハイライト対象: 元ラベル={number1}(青), ノイズ後ラベル={number2}(赤)")
        return [number1, number2]
    except (IndexError, ValueError) as e:
        raise ValueError(f"ディレクトリ名'{dir_name}'を'number1_number2'形式として解析できませんでした。エラー: {e}")

def target_plot_probabilities_single_epoch_emnist(
    data_dir,
    epoch,
    savefig=True,
    show_legend=True,
    show_ylabel=True,
    show_yticklabels=True,
    show_xlabel=True,
    show_xticks=True,
    ax=None
):
    """EMNIST専用: 単一エポックの確率をプロットする"""
    data = load_data_emnist(data_dir)
    if epoch not in data:
        print(f"Epoch {epoch} のデータが {data_dir} に存在しません。")
        return

    df = data[epoch]
    fig_and_log_dir = os.path.join(data_dir, "fig_and_log")
    os.makedirs(fig_and_log_dir, exist_ok=True)

    highlight_targets = get_highlight_targets_emnist(data_dir)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    alpha_values = df['alpha']
    other_colors = ['green','orange','purple','brown','gray','pink','olive','cyan','lime','navy']
    dotted_color_cycle = cycle(other_colors)

    # EMNISTのプロットロジック (prob_0, prob_1, ... を描画)
    for target_val in range(10): # クラスは10個と仮定
        column_name = f'prob_{target_val}'
        if column_name not in df.columns:
            print(f"警告: Column '{column_name}' が存在しません。スキップします。")
            continue

        if target_val == highlight_targets[0]:
            color = 'blue'
            line_style = '-'
            line_label = 'Original label'
            line_alpha = 1.0
            line_width = 2.0
        elif target_val == highlight_targets[1]:
            color = 'red'
            line_style = '-'
            line_label = 'Label after label noise'
            line_alpha = 1.0
            line_width = 2.0
        else:
            color = next(dotted_color_cycle)
            line_style = '--'
            line_label = None
            line_alpha = 1.0
            line_width = 1.0

        ax.plot(
            alpha_values,
            df[column_name],
            line_style,
            color=color,
            alpha=line_alpha,
            linewidth=line_width,
            label=line_label
        )

    # --- 以下、元のコードから流用した描画設定 (変更なし) ---
    ax.axvline(x=0.0, color='black', linestyle='-', linewidth=1.0, label=None)
    ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.0, label=None)

    if show_xlabel:
        ax.set_xticks([0.0, 1.0], [r'$x_0$', r'$x_1$'])
        ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=28)
    else:
        ax.set_xlabel('')

    if show_xticks:
        ax.set_xticks([0.0, 1.0], [r'$x_0$', r'$x_1$'])
        ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=30)
    else:
        ax.set_xticks([])

    if show_ylabel:
        ax.set_ylabel('probability', fontsize=28)
    if not show_yticklabels:
        ax.tick_params(axis='y', which='both', labelleft=False)

    ax.set_ylim(-0.1, 1.1)
    
    # マーカーのプロット (ディレクトリ名にnoiseが含まれるかで判断)
    marker_size = 10
    # この部分は元のロジックを簡略化し、常に両方のラベルポイントを示すようにします。
    # 必要に応じて元の `determine_vertical_lines` のような複雑なロジックをここに実装してください。
    ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
    ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)


    if show_legend and ax is not None:
        handles, labels = ax.get_legend_handles_labels()
        others_line = Line2D([0],[0], color='black', linestyle='--', label='others', linewidth=1.0, alpha=1.0)
        handles.append(others_line)
        labels.append("others")
        unique = list(dict(zip(labels, handles)).items())
        if unique:
            unique_labels, unique_handles = zip(*unique)
            ax.legend(unique_handles, unique_labels, loc="upper right")

    if ax is None:
        if savefig:
            save_path = os.path.join(fig_and_log_dir, f"emnist_epoch{epoch}.png")
            plt.savefig(save_path, format='svg', dpi=100, bbox_inches='tight', transparent=False)
            print(f"画像を保存しました: {save_path}")
            plt.close(fig)
        else:
            plt.show()


def plot_multiple_epochs_grid_emnist(
    data_dir,
    epochs,
    savefig=True,
    show_legend=True,
    show_ylabel=True,
    show_yticklabels=True,
    save_filename='emnist_combined_plot.png',
    columns=4
):
    """EMNIST専用: 複数エポックをグリッド形式でプロットする"""
    num_epochs = len(epochs)
    rows = (num_epochs + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(20, 8), dpi=100)
    
    if rows == 1 and columns == 1:
        axes = np.array([[axes]])
    elif rows == 1 or columns == 1:
        axes = np.reshape(axes, (rows, columns))
    else:
        axes = np.array(axes).reshape(rows, columns)

    last_row = rows - 1
    # タイトル用の文字リスト (元のコードから流用)
    title_chars = ["H","A","B","C","D","E","F","G"]

    for idx, epoch in enumerate(epochs):
        row = idx // columns
        col = idx % columns
        ax = axes[row, col]

        current_show_xlabel = (row == last_row)
        current_show_xticks = (row == last_row)
        current_show_ylabel = (col == 0 and show_ylabel)
        current_show_yticklabels = (col == 0 and show_yticklabels)

        target_plot_probabilities_single_epoch_emnist(
            data_dir=data_dir,
            epoch=epoch,
            savefig=False,
            show_legend=False,
            show_ylabel=current_show_ylabel,
            show_yticklabels=current_show_yticklabels,
            show_xlabel=current_show_xlabel,
            show_xticks=current_show_xticks,
            ax=ax
        )
        if idx < len(title_chars):
            ax.set_title(f'epoch{epoch} ({title_chars[idx]})')
        else:
            ax.set_title(f'epoch{epoch}')

    # 余分な軸を削除
    total_subplots = rows * columns
    for idx in range(num_epochs, total_subplots):
        row = idx // columns
        col = idx % columns
        fig.delaxes(axes[row, col])

    # 凡例を最初のグラフに表示
    if show_legend and axes.size > 0:
        # axは最後のループのものを指しているため、最初のaxから凡例情報を取得
        first_ax = fig.axes[0]
        handles, labels = first_ax.get_legend_handles_labels()
        others_line = Line2D([0],[0], color='black', linestyle='--', label='others', linewidth=1.0, alpha=1.0)
        handles.append(others_line)
        labels.append("others")
        unique = list(dict(zip(labels, handles)).items())
        if unique:
            unique_labels, unique_handles = zip(*unique)
            first_ax.legend(unique_handles, unique_labels, loc="upper right")

    if savefig:
        # 保存先は実行ディレクトリからの相対パスなどに適宜変更してください
        os.makedirs("./output_emnist", exist_ok=True)
        save_path = os.path.join("./output_emnist", save_filename)
        plt.savefig(save_path, format='pdf', dpi=100, bbox_inches='tight', transparent=False)
        print(f"結合した画像を保存しました: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def main_emnist_single():
    """EMNIST専用: 単一のペアディレクトリを指定して実行するメイン関数"""
    
    # --- 実行する対象ディレクトリを指定してください ---
    # 例: root_dir = "/path/to/your/data/pair1/9_8"
    root_dir = "/workspace/data/emnist_pairs/pair1/9_8" # このパスはご自身の環境に合わせてください
    
    # --- プロットしたいエポックのリストを指定してください ---
    epochs_to_plot = [1, 5, 30, 50, 100, 200, 500, 1000]

    # ファイル名をディレクトリ名から自動生成
    pair_name = os.path.basename(os.path.dirname(root_dir))
    labels_name = os.path.basename(root_dir)
    output_filename = f"{pair_name}_{labels_name}.pdf"

    print(f"処理中: {root_dir}")
    plot_multiple_epochs_grid_emnist(
        data_dir=root_dir,
        epochs=epochs_to_plot,
        savefig=True,
        show_legend=True, # 凡例を表示する場合はTrue
        save_filename=output_filename,
        columns=4
    )
    print("完了。")

def main_emnist_all_pairs():
    """EMNIST専用: 指定したベースディレクトリ以下の全てのペアを処理する関数"""
    
    # --- 全てのペアが格納されているベースディレクトリを指定してください ---
    # 例: base_dir = "/path/to/your/data/" 
    # この下に pair1, pair2, ... がある想定
    base_dir = "/workspace/alpha_test/emnist_digits/0.2/kyu_ver2_8/noise" # このパスはご自身の環境に合わせてください

    # --- プロットしたいエポックのリストを指定してください ---
    epochs_to_plot = [1, 5, 30, 50, 100, 200, 500, 1000]

    # 'pair'で始まるディレクトリを検索
    pair_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('pair')])

    for pair_dir in pair_dirs:
        pair_path = os.path.join(base_dir, pair_dir)
        # 'number1_number2' 形式のサブディレクトリを検索
        label_dirs = [d for d in os.listdir(pair_path) if os.path.isdir(os.path.join(pair_path, d))]
        
        for label_dir in label_dirs:
            target_dir = os.path.join(pair_path, label_dir)
            
            output_filename = f"{pair_dir}_{label_dir}.pdf"
            
            print(f"処理中: {target_dir}")
            plot_multiple_epochs_grid_emnist(
                data_dir=target_dir,
                epochs=epochs_to_plot,
                savefig=True,
                show_legend=True,
                save_filename=output_filename,
                columns=4
            )
    print("全ての処理が完了しました。")


# --- 実行 ---
if __name__ == "__main__":
    # どちらか一方のコメントを解除して実行してください

    # 1. 単一のペアディレクトリを指定して実行する場合
    main_emnist_single()

    # 2. 指定したディレクトリ以下の全てのペアをまとめて実行する場合
    # main_emnist_all_pairs()