import os
import re
from typing import List, Tuple, Dict, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import os, glob, concurrent.futures
plt.rcParams["font.size"] = 35
plt.rcParams["figure.figsize"] = [13, 8]
plt.rcParams["figure.dpi"] = 400
plt.rcParams['font.family'] = 'Times New Roman'
def parse_args_model_save_separate():
    parser = argparse.ArgumentParser(description='PyTorch Training with Model Saving and Data Grouping')

    # --- 基本設定 ---
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to use (e.g., cifar10, distribution_colored_emnist)')
    parser.add_argument('--target', type=str, default='digits',
                        help='Target for the dataset (e.g., digits, colors, combined)')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture')
    parser.add_argument('--model_width', type=int, default=64,
                        help='Width of the model (e.g., for ResNet)')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Optimizer to use (sgd or adam)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--fix_seed', type=int, default=42,
                        help='Seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for DataLoader')

    # --- ノイズ設定 ---
    parser.add_argument('--label_noise_rate', type=float, default=0.0,
                        help='Rate of label noise')
    parser.add_argument('--variance', type=float, default=0.0,
                        help='Variance for certain datasets')
    parser.add_argument('--gray_scale', action='store_true',
                        help='Convert images to grayscale')
    parser.add_argument('--weight_noisy', type=float, default=1.0,
                        help='Weight for noisy samples in loss calculation')
    parser.add_argument('--weight_clean', type=float, default=1.0,
                        help='Weight for clean samples in loss calculation')

    # --- ログ設定 ---
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='pytorch-training',
                        help='W&B project name')

    # --- グループ分け機能の引数 ---
    parser.add_argument('--group_data', action='store_true',
                        help='If specified, run data grouping logic instead of training.')
    parser.add_argument('--epoch_a', type=int, default=10,
                        help='Reference epoch "a" for initial correctness check.')
    parser.add_argument('--epoch_t', type=int, default=11,
                        help='Start epoch "t" for the evaluation window.')
    parser.add_argument('--epoch_k', type=int, default=50,
                        help='End epoch "k" for the evaluation window.')

    args = parser.parse_args()
    return args

def get_sample_dirs(base_dir: str) -> List[str]:
    """
    base_dir 以下の 2 階層下で "fig_and_log" を含むディレクトリを返す。
    """
    sample_dirs = []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if not os.path.isdir(d_path):
            continue
        for sub in os.listdir(d_path):
            sub_path = os.path.join(d_path, sub)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "fig_and_log")):
                sample_dirs.append(sub_path)
    return sample_dirs

def get_sample_dirs_one_level(base_dir: str) -> List[str]:
    """
    base_dir 直下のディレクトリで、"fig_and_log" を含むものを返す。
    例: base_dir/sample1/fig_and_log が存在する場合、base_dir/sample1 を返す。
    """
    sample_dirs = []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if os.path.isdir(d_path) and os.path.exists(os.path.join(d_path, "fig_and_log")):
            sample_dirs.append(d_path)
    return sample_dirs

def list_epoch_files(
    csv_dir: str,
    start: Optional[int] = None,
    end:   Optional[int] = None
) -> List[Tuple[int, str]]:
    """
    csv_dir 内の epoch_{n}.csv を (epoch, filepath) のリストで返す。
    範囲指定 (start,end) があればフィルタ。
    """
    files = []
    for fname in os.listdir(csv_dir):
        match = re.match(r'epoch_(\d+)\.csv$', fname)
        if not match:
            continue
        epoch = int(match.group(1))
        if start is not None and epoch < start:
            continue
        if end   is not None and epoch > end:
            continue
        files.append((epoch, os.path.join(csv_dir, fname)))
    files.sort(key=lambda x: x[0])
    return files


def compute_spatial_instability(
    df: pd.DataFrame,
    y_scale: Literal['ratio', 'percent', 'raw'] = 'ratio'
) -> float:
    """
    各 epoch 内での predicted_label の変化回数を計算。
    'ratio'/'percent'/'raw' に対応。
    """
    preds = df['predicted_label']
    n = len(preds) - 1
    if n <= 0 or preds.nunique() <= 1:
        return 0.0

    changes = int((preds != preds.shift()).sum()) - 1
    if y_scale == 'ratio':
        return changes / n
    if y_scale == 'percent':
        return (changes / n) * 100
    return float(changes)


def compute_temporal_instability(
    dfs: List[pd.DataFrame],
    y_scale: Literal['ratio', 'percent', 'raw'] = 'raw'
) -> Dict[float, float]:
    """
    複数 epoch の df リストを受け取り、各 alpha ごとに変化回数を計算。
    'ratio'/'percent'/'raw' に対応。

    Returns
    -------
    { alpha_value: score }
    """
    if len(dfs) < 2:
        return {}
    alphas = dfs[0]['alpha'].to_numpy()
    M = len(alphas)
    counts = np.zeros(M, dtype=int)
    # 各 epoch 間の変化を集計
    for i in range(len(dfs) - 1):
        cur = dfs[i]['predicted_label'].to_numpy()
        nxt = dfs[i+1]['predicted_label'].to_numpy()
        counts += (cur != nxt).astype(int)
    # スケール
    if y_scale == 'ratio':
        scores = counts / (len(dfs) - 1)
    elif y_scale == 'percent':
        scores = (counts / (len(dfs) - 1)) * 100
    else:
        scores = counts.astype(float)
    return {alpha: float(score) for alpha, score in zip(alphas, scores)}


def save_scores_csv(
    df: pd.DataFrame,
    path: str
) -> None:
    """
    DataFrame を CSV に保存し、ディレクトリも自動作成。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Scores saved to: {path}")


def load_score_csv(path: str) -> pd.DataFrame:
    """
    CSV を読み込んで DataFrame で返す。
    """
    return pd.read_csv(path)


def plot_instability_curve(
    x:       np.ndarray,
    y:       np.ndarray,
    std:     Optional[np.ndarray],
    xlabel:  str,
    ylabel:  str,
    save_path: str,
    log_scale_x: bool = False,
    y_lim:   Optional[Tuple[float,float]] = None
) -> None:
    """
    mean (y) ± std を帯域表示した折れ線図を保存。
    """
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    ax.plot(x, y, linewidth=2, zorder=3,color="blue")
    if std is not None:
        ax.fill_between(x, y - std, y + std, alpha=0.2, zorder=2,color="blue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale('symlog')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Plot saved to: {save_path}")


def evaluate_label_changes(
    pair_csv_dir: str,
    output_dir:   str,
    mode:         Literal['alpha','epoch'] = 'alpha',
    y_scale:      Literal['ratio','percent','raw'] = 'raw',
    epoch_start:  Optional[int] = None,
    epoch_end:    Optional[int] = None,
    plot:         bool = True
) -> Dict[float,float]:
    """
    単一サンプル (pair_csv_dir) に対して指定モードの不安定性を計算し、
    scores を CSV/プロットとして output_dir に保存。
    """
    files = list_epoch_files(pair_csv_dir, epoch_start, epoch_end)
    if not files:
        raise ValueError("No epoch CSV files found in directory.")
    suffix = ''
    if epoch_start is not None or epoch_end is not None:
        s = str(epoch_start) if epoch_start is not None else 'start'
        e = str(epoch_end)   if epoch_end   is not None else 'end'
        suffix = f"_epoch_{s}_to_{e}"

    if mode == 'alpha':
        scores = {}
        for ep, fp in files:
            df = pd.read_csv(fp)
            score = compute_spatial_instability(df, y_scale)
            scores[ep] = score
        df_out = pd.DataFrame({'epoch': list(scores.keys()), 'label_change': list(scores.values())})
        csv_path = os.path.join(output_dir, f'label_change_scores_alpha{suffix}.csv')
        save_scores_csv(df_out, csv_path)
        if plot:
            xs = df_out['epoch'].to_numpy()
            ys = df_out['label_change'].to_numpy()
            plot_instability_curve(
                xs, ys, None,
                xlabel='Epoch',
                ylabel=f"Spatial Instability ({y_scale})",
                save_path=os.path.join(output_dir, f'label_change_scores_alpha{suffix}.svg'),
                log_scale_x=True
            )
        return scores

    # mode == 'epoch'
    print(y_scale)
    dfs = [pd.read_csv(fp) for _, fp in files]
    scores = compute_temporal_instability(dfs, y_scale)
    df_out = pd.DataFrame({'alpha': list(scores.keys()), 'unsmoothed_scores': list(scores.values())})
    csv_path = os.path.join(output_dir, f'epoch_unsmoothed_scores{suffix}.csv')
    save_scores_csv(df_out, csv_path)
    if plot:
        xs = np.array(list(scores.keys()))
        ys = np.array(list(scores.values()))
        plot_instability_curve(
            xs, ys, None,
            xlabel='Alpha',
            ylabel=f"Temporal Instability ({y_scale})",
            save_path=os.path.join(output_dir, f'epoch_unsmoothed_scores{suffix}.svg')
        )
    return scores


def evaluate_label_changes_all(
    sample_dirs: List[str],
    **kwargs
) -> Dict[str, Dict[float,float]]:
    """
    複数サンプルで evaluate_label_changes を一括実行。
    """
    all_scores = {}
    for d in sample_dirs:
        try:
            print(f"[Info] Processing {d}")
            sc = evaluate_label_changes(d, **kwargs)
            all_scores[d] = sc
        except Exception as ex:
            print(f"[Error] {d}: {ex}")
    return all_scores


from pathlib import Path

def aggregate_instability_across_samples(
    sample_dirs:  List[str],
    target:       str,
    mode:         Literal['alpha', 'epoch'],
    y_scale:      Literal['ratio', 'percent', 'raw'],
    epoch_range:  Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    各サンプルのスコア CSV を読み込み、x_value ごとの mean/std を返す DataFrame。
    なければ mode==epoch のときに evaluate_label_changes で CSV を生成する。
    """
    # suffix
    suffix = ''
    if epoch_range:
        s, e = epoch_range
        suffix = f"_epoch_{s}_to_{e}"

    rows = []
    for d in sample_dirs:
        base = os.path.join(d, 'fig_and_log')

        if mode == 'alpha':
            fname = f'label_change_scores_alpha.csv'   # suffix は alpha 側では使っていなかった
            x_col, y_col = 'epoch', 'label_change'
        elif mode == 'epoch':
            # epoch_unsmoothed_scores_combined_epoch_1_to_1000
            fname = f'epoch_unsmoothed_scores{suffix}.csv'
            x_col, y_col = 'alpha', 'unsmoothed_scores'
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        fpath = os.path.join(base, fname)

        # ========== フォールバック生成 ==========
        if mode == 'epoch' and not os.path.exists(fpath):
            print(f"[Info] {fpath} が存在しないため自動生成します…")
            try:
                pair_csv_dir = os.path.join(d, 'csv')
                evaluate_label_changes(
                    pair_csv_dir=pair_csv_dir,
                    output_dir=base,
                    mode='epoch',
                    y_scale=y_scale,
                    epoch_start=epoch_range[0] if epoch_range else None,
                    epoch_end=epoch_range[1] if epoch_range else None,
                    plot=False     # ここでプロット不要なら False
                )
            except Exception as e:
                print(f"[Warn] 生成に失敗 ({e}) : {pair_csv_dir}")
                continue  # このサンプルは飛ばす

        # ---------- 読み込み ----------
        if not os.path.exists(fpath):
            print(f"[Warn] missing {fpath}")
            continue

        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            rows.append((row[x_col], row[y_col]))

    # ---------- 集計 ----------
    if not rows:
        print("[Error] No valid instability data found.")
        return pd.DataFrame()

    df_all = pd.DataFrame(rows, columns=['x', 'score'])
    stats = (
        df_all
        .groupby('x')['score']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'x': 'x_value', 'mean': 'mean_score', 'std': 'std_score'})
    )
    return stats
def plot_aggregate_temporal_instability(
    stats_df:    pd.DataFrame,
    xlabel:      str,
    ylabel:      str,
    save_path:   str,
    *,
    mode:        Literal["no_noise", "noise"] = "no_noise",
    epoch_range: Optional[Tuple[int, int]]     = None,
    highlight:   Optional[List[float]]         = None,
    log_scale_x: bool                          = False,
    y_lim:       Optional[Tuple[float, float]] = None,
    marker_size: int                           = 15,
    highlight_epochs=None,
    abc=None # このパラメータを使用して動的な凡例を生成します
) -> None:
    """
    aggregate_instability の結果をプロットして保存する。

    Parameters
    ----------
    mode : {"no_noise", "noise"}
        α=0,1 のマーカー色を切り替えるフラグ
        "no_noise": α=0,1 とも青 / "noise": α=0 は青, α=1 は赤
    epoch_range : (start, end), optional
        xlabel == "alpha" の場合に y, std を
        (end - start) で割って正規化するための区間
    abc : list of tuples, optional
        凡例に表示するTのサブスクリプトペア (例: [("A","C"), ("C","E")])
        指定しない場合、または不正な形式の場合はデフォルトの"AC"が使用されます。
    """

    # ---------- main curve ----------
    x   = stats_df["x_value"].to_numpy()

    y   = stats_df["mean_score"].to_numpy()
    std = stats_df["std_score"].to_numpy()

    # xlabel = alpha のときだけ epoch_range で正規化
    if xlabel.lower() == "alpha" and epoch_range is not None:
        denom = max(epoch_range[1] - epoch_range[0], 1)  # ゼロ割防止
        print(f"Normalization denominator: {denom}")
        y   = y   / denom
        std = std / denom
    else:
        y   = y / 200
        std = std  / 100.0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    if highlight:
        for v in highlight:
            ax.axvline(v, color="black", linestyle="-")

    ax.plot(x, y, linewidth=2, zorder=3,
             label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color="blue")
    ax.fill_between(x, y - std, y + std,
                     alpha=0.2, zorder=2, color="blue")
    if highlight_epochs:
        for ep in highlight_epochs:
            ax.axvline(x=ep, color='black', linestyle='-', linewidth=1.5, zorder=1)

    # --- α = 0,1 の補助線 & マーカー、および動的なTラベル ---
    if xlabel.lower() == "alpha":
        # abcが提供されており、正しい形式のリストであることを確認
        if abc and isinstance(abc, list) and all(isinstance(t, tuple) and len(t) == 2 for t in abc):
            for p1, p2 in abc:
                # epoch_rangeが指定されている場合のみ凡例に表示
                if epoch_range is not None:
                    ax.plot([], [], label=rf'$\mathcal{{T}}_{{{p1}{p2}}}$=[{epoch_range[0]},{epoch_range[1]}]', linewidth=0)
                else:
                    # epoch_rangeがない場合はシンプルな形式で表示
                    ax.plot([], [], label=rf'$\mathcal{{T}}_{{{p1}{p2}}}$', linewidth=0)
        else:
            # abcが提供されていない、または不正な形式の場合のフォールバック
            if epoch_range is not None:
                # ax.plot([], [], label=rf'$\mathcal{{T}}_{{AC}}$=[{epoch_range[0]},{epoch_range[1]}]', linewidth=0)
                ax.plot([], [], label=rf'$\mathcal{{T}}$=[{epoch_range[0]},{epoch_range[1]}]', linewidth=0)

            else:
                ax.plot([], [], label=rf'$\mathcal{{T}}$', linewidth=0)


        ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)

        if mode == "no_noise":
            ax.plot([0.0, 1.0], [0.0, 0.0], "o",
                     color="blue", markersize=marker_size, zorder=5)
        else:  # mode == "noise"
            ax.plot(0.0, 0.0, "o", color="blue",
                     markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, "o", color="red",
                     markersize=marker_size, zorder=5)

    # ---------- aesthetics ----------
    ax.set_ylabel(ylabel) # オリジナルでコメントアウトされていました
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r'$x_{0}$', r'$x_{1}$'], fontsize=40)

    if y_lim:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale("log")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # ---------- save ----------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Aggregate plot saved to: {save_path}")
def plot_aggregate_instability(
    stats_df:   pd.DataFrame,
    xlabel:     str,
    ylabel:     str,
    save_path:  str,
    *,
    mode:          Literal["no_noise", "noise"] = "no_noise",
    epoch_range:   Optional[Tuple[int, int]]     = None,          # ← 追加
    highlight:     Optional[List[float]]         = None,
    log_scale_x:   bool                         = False,
    y_lim:         Optional[Tuple[float, float]] = None,
    marker_size:   int                          = 10,
    highlight_epochs=None,
) -> None:
    """
    aggregate_instability の結果をプロットして保存する。

    Parameters
    ----------
    mode : {"no_noise", "noise"}
        α=0,1 のマーカー色を切り替えるフラグ
        "no_noise": α=0,1 とも青 / "noise": α=0 は青, α=1 は赤
    epoch_range : (start, end), optional
        xlabel == "alpha" の場合に y, std を
        (end - start) で割って正規化するための区間
    """

    # ---------- main curve ----------
    x   = stats_df["x_value"].to_numpy()

    y   = stats_df["mean_score"].to_numpy()
    std = stats_df["std_score"].to_numpy()

    # xlabel = alpha のときだけ epoch_range で正規化
    if xlabel.lower() == "alpha" and epoch_range is not None:
        denom = max(epoch_range[1] - epoch_range[0], 1)  # ゼロ割防止
        print(denom)
        y   = y   / denom
        std = std / denom
    else:
        # y   = y 
        # std = std  
        y   = y / 200
        std = std /200
    # fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    #miruの発表要
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    if highlight:
        for v in highlight:
            ax.axvline(v, color="black", linestyle="-")

    ax.plot(x, y, linewidth=2, zorder=3,
            label="$INST_s(\\chi,t)$", color="blue")
    ax.fill_between(x, y - std, y + std,
                    alpha=0.2, zorder=2, color="blue")
    if highlight_epochs:
        for ep in highlight_epochs:
            ax.axvline(x=ep, color='black', linestyle='-', linewidth=1.5, zorder=1)
    # ---------- α = 0,1 の補助線 & マーカー ----------
    if xlabel.lower() == "alpha":
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)

        if mode == "no_noise":
            ax.plot([0.0, 1.0], [0.0, 0.0], "o",
                    color="blue", markersize=marker_size, zorder=5)
        else:  # mode == "noise"
            ax.plot(0.0, 0.0, "o", color="blue",
                    markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, "o", color="red",
                    markersize=marker_size, zorder=5)

    # ---------- aesthetics ----------
    ax.autoscale(False)   # 以後どんなアーティストを追加しても軸範囲は固定

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(y_lim)
    ax.set_xlim(0.9,1000)
    ax.set_yticks([0.0, 0.01])
    if y_lim:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale("log")
    ax.set_ylim(-0.0001, 0.011)
    ax.set_yticks([0.0, 0.01])
    # ax.set_yticks([0.0, 0.02,0.04, 0.06, 0.08, 0.1])


    
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # ---------- save ----------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Aggregate plot saved to: {save_path}")
def append_cross_entropy_to_csv(
    csv_dir:      str,
    true_label:   int,
    epoch_start:  Optional[int] = None,
    epoch_end:    Optional[int] = None
) -> None:
    """
    各 epoch_{n}.csv に対して、行ごとの -log(prob_{true_label}) を計算し
    'cross_entropy' 列として追加して上書き保存する。
    """
    files = list_epoch_files(csv_dir, epoch_start, epoch_end)
    if not files:
        raise ValueError(f"No epoch CSV files found in {csv_dir}")

    prob_col = f'prob_{true_label}'
    for ep, fp in files:
        df = pd.read_csv(fp)
        if prob_col not in df.columns:
            raise ValueError(f"File {fp} does not contain column '{prob_col}'")
        probs = df[prob_col].astype(float).to_numpy()
        # log(0) 対策
        probs = np.clip(probs, 1e-12, 1.0)
        df['cross_entropy'] = -np.log(probs)
        # 上書き保存
        df.to_csv(fp, index=False)
        print(f"[✓] Appended 'cross_entropy' to: {fp}")

def save_label_change_to_csv_with_sample_dirs(base_root, widths, target_epoch, save_path):
    data = []

    for width in widths:
        for noise_type in ['noise', 'no_noise']:
            base_dir = os.path.join(base_root, str(width), noise_type)
            sample_dirs = get_sample_dirs(base_dir)  # get_sample_dirsを使用
            scores = []

            for sdir in sample_dirs:
                score_file = os.path.join(sdir, 'fig_and_log', 'label_change_scores_alpha.csv')
                if not os.path.exists(score_file):
                    print(f"[Warn] Missing score file: {score_file}")
                    continue
                try:
                    df = pd.read_csv(score_file)
                    row = df[df['epoch'] == target_epoch]
                    if not row.empty:
                        score = float(row['label_change'].values[0])
                        scores.append(score)
                    else:
                        print(f"[Warn] No data at epoch={target_epoch} in {score_file}")
                except Exception as e:
                    print(f"[Error] Failed reading {score_file}: {e}")

            if scores:
                mean_score = np.mean(scores)
                data.append({'width': width, 'noise': mean_score if noise_type == 'noise' else None, 'no_noise': mean_score if noise_type == 'no_noise' else None})

    # データフレームに変換してCSVに保存
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"[✓] CSV saved to: {save_path}")

def plot_mean_std_match_rates_per_epoch(
    base_dir: str,
    plot_save_path: str,
    csv_save_path: str,
    *,
    mode: Literal["noise", "no_noise"] = "noise"
):
    """
    指定フォルダ以下の "epoch_*.csv" を探索し、一致率をエポックごとに集計。
    全ペアの平均と標準偏差を計算し、プロットおよびデータ保存を行う。

    Parameters
    ----------
    base_dir : str
        ペアごとのCSVファイルを格納しているベースディレクトリ。
        (例: base_dir/pairXXX/num1_num2/csv/epoch_0.csv)
    plot_save_path : str
        プロット画像の保存先パス。
    csv_save_path : str
        プロット用データを保存するCSVのパス。
    mode : {"noise", "no_noise"}, default "noise"
        - "noise": num1 (noisy) と num2 (clean) の両方を処理する。
        - "no_noise": num2 (clean) のみ処理する。
    """
    print(f"処理モード: {mode}")
    print("データファイルの検索と集計を開始します...")
    
    search_pattern = os.path.join(base_dir, '**', 'epoch_*.csv')
    file_paths = glob.glob(search_pattern, recursive=True)

    results_list = []
    path_pattern = re.compile(r'pair\d+[/\\](\d+)_(\d+)[/\\]csv[/\\]epoch_(\d+)\.csv')

    for path in file_paths:
        match = path_pattern.search(path)
        if not match:
            continue
        
        num1, num2, epoch = map(int, match.groups())

        try:
            df = pd.read_csv(path, usecols=['predicted_label'])
        except (ValueError, FileNotFoundError) as e:
            print(f"警告: {path} の読み込みに失敗しました。スキップします。エラー: {e}")
            continue
            
        if df.empty:
            continue

        total_samples = len(df)
        
        # num2 (clean label) は常に計算
        rate_num2 = (df['predicted_label'] == num2).sum() / total_samples
        record = {'epoch': epoch, 'rate_num2': rate_num2}

        # "noise"モードの場合のみnum1 (noisy label) を計算
        if mode == "noise":
            rate_num1 = (df['predicted_label'] == num1).sum() / total_samples
            record['rate_num1'] = rate_num1
        
        results_list.append(record)
        
    if not results_list:
        print("警告: 対象となるデータが見つかりませんでした。処理を終了します。")
        return

    results_df = pd.DataFrame(results_list)
    
    # modeに応じて集計する内容を定義
    agg_dict = {
        'mean_rate2': ('rate_num2', 'mean'),
        'std_rate2': ('rate_num2', 'std'),
    }
    if mode == "noise":
        agg_dict.update({
            'mean_rate1': ('rate_num1', 'mean'),
            'std_rate1': ('rate_num1', 'std'),
        })
        
    stats_df = results_df.groupby('epoch').agg(**agg_dict).reset_index()
    stats_df = stats_df.fillna(0)
    
    # 💾 CSV 保存
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    stats_df.to_csv(csv_save_path, index=False)
    print(f"✅ 統計データをCSVに保存しました: {csv_save_path}")

    # 📊 プロット
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # number2 (clean label) のプロット (常に行う)
    ax.plot(stats_df['epoch'], stats_df['mean_rate2'], label='clean label (mean)', color='royalblue')
    ax.fill_between(
        stats_df['epoch'],
        stats_df['mean_rate2'] - stats_df['std_rate2'],
        stats_df['mean_rate2'] + stats_df['std_rate2'],
        alpha=0.2, color='royalblue', label='clean label (std dev)'
    )

    # "noise"モードの場合のみ number1 (noisy label) をプロット
    if mode == "noise":
        ax.plot(stats_df['epoch'], stats_df['mean_rate1'], label='noisy label (mean)', color='darkorange')
        ax.fill_between(
            stats_df['epoch'],
            stats_df['mean_rate1'] - stats_df['std_rate1'],
            stats_df['mean_rate1'] + stats_df['std_rate1'],
            alpha=0.2, color='darkorange', label='noisy label (std dev)'
        )
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Match Rate')
    title = 'Epochごとの平均一致率（全ペア平均 ± 標準偏差）' if mode == "noise" else 'Epochごとの一致率 (Clean Label)'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    # 🖼️ 画像を保存
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.close(fig)
    print(f"✅ 図を保存しました: {plot_save_path}")
def analyze_all_temporal_instability(base_root, widths, output_dir, target_row=None):
    """
    すべてのalphaに対して temporal instability を計算し、
    width ごとの CSV ファイルを生成する。
    さらに、各 alpha に対して width vs instability のプロットを生成する。

    Parameters
    ----------
    base_root : str
        ベースディレクトリのパス
    widths : list
        調査対象の width のリスト
    output_dir : str
        結果を保存するディレクトリ
    target_row : int, optional
        特定の行のみを解析する場合の行番号（1-indexed）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 最初のファイルからalpha値のリストを取得
    first_sample = None
    for width in widths:
        base_dir = os.path.join(base_root, str(width), 'noise')
        samples = get_sample_dirs(base_dir)
        if samples:
            first_sample = samples[0]
            break
    
    if not first_sample:
        raise ValueError("No sample directories found")
    
    first_file = os.path.join(first_sample, 'fig_and_log', 'epoch_unsmoothed_scores.csv')
    df_first = pd.read_csv(first_file)
    alpha_values = df_first['alpha'].tolist()
    
    if target_row is not None:
        alpha_values = [alpha_values[target_row - 1]]

    # 各widthに対してデータを収集
    width_data = {}
    for width in widths:
        print(f"[Info] Processing width {width}...")
        results = []
        
        for alpha in alpha_values:
            scores_noise = []
            scores_no_noise = []
            
            # noise データの収集
            for noise_type in ['noise', 'no_noise']:
                base_dir = os.path.join(base_root, str(width), noise_type)
                sample_dirs = get_sample_dirs(base_dir)
                scores = []
                
                for sdir in sample_dirs:
                    score_file = os.path.join(sdir, 'fig_and_log', 'epoch_unsmoothed_scores.csv')
                    if not os.path.exists(score_file):
                        continue
                    try:
                        df = pd.read_csv(score_file)
                        closest_alpha_idx = (df['alpha'] - alpha).abs().idxmin()
                        score = float(df.iloc[closest_alpha_idx]['unsmoothed_scores'])
                        if noise_type == 'noise':
                            scores_noise.append(score)
                        else:
                            scores_no_noise.append(score)
                    except Exception as e:
                        print(f"[Error] Failed reading {score_file}: {e}")
            
            # 結果を記録
            results.append({
                'alpha': alpha,
                'noise_mean': np.mean(scores_noise) if scores_noise else np.nan,
                'noise_std': np.std(scores_noise) if scores_noise else np.nan,
                'no_noise_mean': np.mean(scores_no_noise) if scores_no_noise else np.nan,
                'no_noise_std': np.std(scores_no_noise) if scores_no_noise else np.nan,
                'n_samples_noise': len(scores_noise),
                'n_samples_no_noise': len(scores_no_noise)
            })
        
        # widthごとのCSVファイルを保存
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f'temporal_instability_width_{width}.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"[✓] Saved results for width {width} to {csv_path}")
        
        width_data[width] = df_results

    # 各alphaに対してプロットを生成
    for i, alpha in enumerate(alpha_values):
        plt.figure(figsize=(10, 6))
        
        # noise と no_noise のデータを収集
        noise_means = []
        noise_stds = []
        no_noise_means = []
        no_noise_stds = []
        
        for width in widths:
            df = width_data[width]
            row = df[df['alpha'] == alpha].iloc[0]
            noise_means.append(row['noise_mean'])
            noise_stds.append(row['noise_std'])
            no_noise_means.append(row['no_noise_mean'])
            no_noise_stds.append(row['no_noise_std'])
        
        # プロット
        plt.errorbar(widths, noise_means, yerr=noise_stds, 
                    label='noise', marker='o', capsize=5)
        plt.errorbar(widths, no_noise_means, yerr=no_noise_stds, 
                    label='no_noise', marker='s', capsize=5)
        
        plt.xlabel('Width')
        plt.ylabel('Temporal Instability')
        plt.title(f'Temporal Instability vs Width (alpha={alpha:.3f})')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'temporal_instability_alpha_{alpha:.3f}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"[✓] Saved plot for alpha={alpha:.3f}")

def save_spatial_instability_epoch_summary(
    base_root: str,
    widths: List[int],
    y_scale: str = 'ratio',
    target_epoch: int = 4000,
    output_path: str = './spatial_instability_epoch_summary.csv'
):
    """
    各 width に対して、target_epoch における Spatial Instability の
    平均と標準偏差（noise / no_noise）を1つのCSVに記録。

    出力列：width, noise_mean, noise_std, no_noise_mean, no_noise_std
    """
    records = []
    
    for width in widths:
        print(f"[Info] Processing width={width}...")

        row_data = {'width': width}

        for noise_type in ['noise', 'no_noise']:
            base_dir = os.path.join(base_root, str(width), noise_type)
            sample_dirs = get_sample_dirs(base_dir)

            stats_df = aggregate_instability_across_samples(
                sample_dirs=sample_dirs,
                target="combined",
                mode="alpha",
                y_scale=y_scale,
                epoch_range=(target_epoch, target_epoch)
            )

            row = stats_df[stats_df['x_value'] == target_epoch]
            if row.empty:
                print(f"[Warn] No data at epoch={target_epoch} for width={width}, type={noise_type}")
                row_data[f"{noise_type}_mean"] = None
                row_data[f"{noise_type}_std"] = None
                continue

            row_data[f"{noise_type}_mean"] = float(row['mean_score'].values[0])
            row_data[f"{noise_type}_std"] = float(row['std_score'].values[0])

        records.append(row_data)

    # CSV保存
    df_out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[✓] Summary CSV saved to: {output_path}")
# 代わりに `ax` を引数で受け取るように変更。
def draw_on_ax_temporal_instability(
    ax: plt.Axes,  # ★★★ 描画対象のaxを引数で受け取る ★★★
    stats_df: pd.DataFrame,
    xlabel: str,
    *,
    mode: Literal["no_noise", "noise"] = "no_noise",
    epoch_range: Optional[Tuple[int, int]] = None,
    log_scale_x: bool = False,
    y_lim: Optional[Tuple[float, float]] = None,
    marker_size: int = 15,
    abc=None
) -> None:
    """与えられたaxオブジェクト上に、集計されたinstabilityの結果をプロットする。"""
    
    # --- データ準備と正規化 ---
    x = stats_df["x_value"].to_numpy()
    y = stats_df["mean_score"].to_numpy()
    std = stats_df["std_score"].to_numpy()

    if xlabel.lower() == "alpha" and epoch_range is not None:
        denom = max(epoch_range[1] - epoch_range[0], 1)
        y = y / denom
        std = std / denom
    else:
        y = y / 100.0
        std = std / 100.0

    # --- 描画処理 (axに対して行う) ---
    ax.plot(x, y, linewidth=2, zorder=3,
            # label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color="blue")
            label=r'$INST_{T}(T,x)$', color="blue")
    ax.fill_between(x, y - std, y + std,
                    alpha=0.2, zorder=2, color="blue")

    if xlabel.lower() == "alpha":
        # 動的凡例の生成
        if abc and isinstance(abc, list) and all(isinstance(t, tuple) and len(t) == 2 for t in abc):
            for p1, p2 in abc:
                if epoch_range is not None:
                    ax.plot([], [], label=rf'$\mathcal{{T}}$=[{epoch_range[0]},{epoch_range[1]}]', linewidth=0)
        # 補助線とマーカー
        ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)

        if mode == "no_noise":
            ax.plot([0.0, 1.0], [0.0, 0.0], "o",
                    color="blue", markersize=marker_size, zorder=5)
        else:
            ax.plot(0.0, 0.0, "o", color="blue",
                    markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, "o", color="red",
                    markersize=marker_size, zorder=5)

    # --- 見た目の調整 (axに対して行う) ---
    # ★★★ Y軸ラベルの設定は、呼び出し元で制御するためここでは行わない ★★★
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r'$x_{0}$', r'$x_{1}$'], fontsize=40) # フォントサイズを調整

    if y_lim:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale("log")
    # ax.legend()
    ax.grid(True)
    # ★★★ fig.tight_layout() や savefig, close はここでは行わない ★★★


# ===== STEP 2: メインスクリプト =====

# def main():
#     # --- 初期設定 ---
#     mode      = "noise"
#     width     = 8
#     base_root = f"alpha_test/cifar10/0.2/64/{mode}"
#     save_dir  = "/workspace/vizualize/ACML/cifar"
#     os.makedirs(save_dir, exist_ok=True)

#     plot_configs = [
#         {"epoch_range": (1, 30), "abc": [("A", "C")]},
#         # {"epoch_range": (27, 42), "abc": [("C", "D")]},
#         {"epoch_range": (30, 80), "abc": [("C", "E")]},
#         {"epoch_range": (80, 4000), "abc": [("E", "G")]},
#         # {"epoch_range": (120, 1000), "abc": [("G", "H")]},

#     ]
#     num_plots = len(plot_configs)
#     sample_dirs = get_sample_dirs(base_root)

#     # --- 連結グラフの準備 ---
#     # Y軸を共有(`sharey=True`)してサブプロットを作成
#     fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True, dpi=300)

#     # --- 各設定でループ処理し、サブプロットに描画 ---
#     for i, config in enumerate(plot_configs):
#         ax = axes[i]  # 描画対象のサブプロットを取得

#         epoch_s, epoch_end = config["epoch_range"]
#         current_abc = config["abc"]

#         print(f"[Info] Processing plot {i+1}/{num_plots}: Epoch range {epoch_s}-{epoch_end}")

#         # データの集計
#         stats_df = aggregate_instability_across_samples(
#             sample_dirs = sample_dirs,
#             target      = "combined",
#             mode        = "epoch",
#             y_scale     = "raw",
#             epoch_range = (epoch_s, epoch_end),
#         )

#         # ★★★ STEP 1で作成した関数を呼び出し、現在のaxに描画 ★★★
#         draw_on_ax_temporal_instability(
#             ax=ax,
#             stats_df=stats_df,
#             xlabel="alpha",
#             mode=mode,
#             epoch_range=(epoch_s, epoch_end),
#             y_lim=(-0.01, 0.63),
#             abc=current_abc,
#         )

#         # サブプロットごとのタイトルを設定

#         # ★★★ 一番左のプロット(i=0)にのみY軸ラベルを表示 ★★★
#         if i == 0:
#             ax.set_ylabel("Temporal Instability")

#     # --- 全体のレイアウトを調整して保存 ---
#     fig.tight_layout()  # プロット間の重なりを自動調整

#     fname_base = f"temporal_instability_w{width}_{mode}_combined_horizontal"
#     svg_path   = os.path.join(save_dir, f"{fname_base}.svg")
#     pdf_path   = os.path.join(save_dir, f"{fname_base}.pdf")

#     print(f"\nSaving combined plot to: {svg_path}")
#     fig.savefig(svg_path)
#     print(f"Saving combined plot to: {pdf_path}")
#     fig.savefig(pdf_path)

#     plt.close(fig) # メモリを解放
#     print("\n[✓] Combined plot has been generated successfully.")


# if __name__ == '__main__':
#     main()




# if __name__ == '__main__':


#     widths = [1, 2, 4, 8, 10, 12, 16, 32, 64]

#     save_spatial_instability_epoch_summary(
#     base_root='alpha_test/cifar10/0.2',
#     widths=[1, 2, 4, 8, 10, 12, 16, 32, 64],
#     y_scale='ratio',
#     target_epoch=4000,  # ここで任意のepochを指定
#     output_path='alpha_test/cifar10/0.2/figure/spatial_inst_epoch4000_summary.csv'
# )
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

def plot_temporal_instability_grid(
    base_roots:   List[str],
    plot_configs: List[Dict[str, object]],
    *,
    # --- 行ごとの mode --------------------------------------------------
    default_mode: str                = "noise",
    row_modes:    Optional[List[str]] = None,
    # --- 描画オプション --------------------------------------------------
    y_scale:    str              = "raw",
    y_lim:      Optional[Tuple[float, float]] = (-0.01, 0.83),
    marker_size:int              = 15,
    col_space:  float            = 0.15,   # ★ 横方向の余白
    row_space:  float            = 0.15,   # ★ 縦方向の余白
    # --- ラベル・保存先 --------------------------------------------------
    ylabel:     str              = "Temporal Instability",
    xlabel:     str              = "alpha",
    save_path:  str              = "./temporal_instability_grid.svg",
) -> None:
    """行=base_roots, 列=plot_configs のグリッド図を生成。"""

    n_rows, n_cols = len(base_roots), len(plot_configs)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("base_roots と plot_configs は空にできません")

    # 行ごとの mode を決定
    if row_modes is None:
        row_modes = [default_mode] * n_rows
    if len(row_modes) != n_rows:
        raise ValueError("row_modes の長さが base_roots と一致しません")

    # --------- Figure / Axes 生成（余白を gridspec_kw で指定） ----------
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8 * n_cols, 5 * n_rows),
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": col_space, "hspace": row_space},  # ★ここ★
    )
    # axes を 2 次元配列化
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # --------- 各セルを描画 --------------------------------------------
    for r, (base_root, mode) in enumerate(zip(base_roots, row_modes)):
        sample_dirs = get_sample_dirs(base_root)
        if not sample_dirs:
            print(f"[Warn] No samples under {base_root}");  continue

        for c, cfg in enumerate(plot_configs):
            ax          = axes[r][c]
            epoch_s, epoch_e = cfg["epoch_range"]
            abc_cfg     = cfg.get("abc")

            stats_df = aggregate_instability_across_samples(
                sample_dirs=sample_dirs,
                target     ="combined",
                mode       ="epoch",
                y_scale    =y_scale,
                epoch_range=(epoch_s, epoch_e),
            )

            draw_on_ax_temporal_instability(
                ax          = ax,
                stats_df    = stats_df,
                xlabel      = xlabel,
                mode        = mode,
                epoch_range = (epoch_s, epoch_e),
                y_lim       = y_lim,
                marker_size = marker_size,
                abc         = abc_cfg,
            )

            if c == 0:                # 左端にだけ Y ラベル
                ax.set_ylabel(ylabel, fontsize=35)

    # --------- 保存 ------------------------------------------------------
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Grid figure saved to: {save_path}")




# --- 設定 --------------------------------------------------------------
def temporal_gattai():
    base_roots = [
    "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/no_noise",
    "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/noise"
    ]

    row_modes = [
        "no_noise",  # 1 行目の mode
        "noise",     # 2 行目の mode
    ]

    plot_configs = [
        {"epoch_range": (1, 27),  "abc": None},
        {"epoch_range": (27, 55), "abc": None},
        {"epoch_range": (55,120), "abc": None},
    ]

    plot_temporal_instability_grid(
        base_roots    = base_roots,
        row_modes     = row_modes,      # 行ごとの mode
        plot_configs  = plot_configs,
        y_scale       = "raw",
        y_lim         = (-0.01, 0.43),
        marker_size=20,
        col_space    = 0.05,   # 横をギュッと
        row_space    = 0.2,   # 縦もギュッと
        save_path     = "vizualize/miru_oral/temporal_instability_grid_2.svg",
    )


def temporal():    
    mode      = "no_noise"          # "noise" or "no_noise"
    width     = 64        # ディレクトリ階層で使う幅
    # base_root = f"alpha_test/emnist_digits/0.2/kyu_ver2_8/{mode}"
    # base_root = f"alpha_test/emnist_digits/0.2/{width}/{mode}"
    save_dir  = "/workspace/vizualize/EMNIST_nosise0_2_64"
    base_root = "alpha_test/emnist_digits/0.2/64/noiseis_correct_0_top100"
    base_root = "alpha_test/emnist_digits/0.2/kyu_ver2_8_random/no_noise"
    base_root = "alpha_test/emnist_digits/0.0/64/no_noise"
    base_root = "alpha_test/emnist_digits/0.2/64_7_20/no_noise"


    # alpha_test/emnist_digits/0.2/64_7_20/noise_random
    # alpha_test/emnist_digits/0.2/64_7_20/noise_top100_select_sample
    # print("[Info] Plotting mean match rates per epoch...")
    # try:
    #     plot_mean_std_match_rates_per_epoch(
    #         base_dir=base_root,
    #         plot_save_path=f"/workspace/vizualize/ACML/EMNIST/irekawari_no/match_{mode}.png",
    #         csv_save_path=f"/workspace/vizualize/ACML/EMNIST/irekawari_no/match{mode}.csv",
    #         mode=mode
    #     )
    # except Exception as e:
    #     print(f"[Error] Failed to plot match rates: {e}")
    # # 解析したい (start, end) のペアを好きなだけ並べる
    # plot_configs = [
    #     {"epoch_range": (0, 1000), "abc": [("A", "B")]}, # Example: new abc for this range
    #     # {"epoch_range": (27, 42), "abc": [("B", "C")]},
    #     # {"epoch_range": (42, 55), "abc": [("C", "D")]},
    #     # {"epoch_range": (27, 55), "abc": [("C", "E")]},
    #     # {"epoch_range": (55, 120), "abc": [("E", "G")]},
    #     # {"epoch_range": (120, 1000), "abc": [("G", "H")]},
    #     # Add more configurations as needed
    # ]
    # plot_configs = [
    #     {"epoch_range": (1, 27), "abc": [("A", "B")]}, 
    #     {"epoch_range": (27, 55), "abc": [("B", "C")]},
    #     {"epoch_range": (55, 120), "abc": [("C", "D")]},
    #     # {"epoch_range": (27, 55), "abc": [("C", "E")]},
    #     # {"epoch_range": (55, 120), "abc": [("E", "G")]},
    #     # {"epoch_range": (120, 1000), "abc": [("G", "H")]},
    #     # Add more configurations as needed
    # ]
    # plot_configs = [
    #     {"epoch_range": (1, 1000), "abc": None},
    #     {"epoch_range": (27, 55), "abc": None},
    #     {"epoch_range": (55, 120), "abc": None},
    # ]
    plot_configs = [
        {"epoch_range": (1, 30), "abc": None},
        {"epoch_range": (30, 53), "abc": None},
        {"epoch_range": (53, 140), "abc": None},
    ]
    sample_dirs = get_sample_dirs(base_root)

    for config in plot_configs:
        epoch_s, epoch_end = config["epoch_range"]
        current_abc = config["abc"] # Get the abc list for the current configuration

        print(f"[Info] Epoch range {epoch_s}-{epoch_end} with abc: {current_abc}")

        stats_df = aggregate_instability_across_samples(
            sample_dirs = sample_dirs,
            target      = "combined",
            mode        = "epoch",      # temporal instability
            y_scale     = "raw",
            epoch_range = (epoch_s, epoch_end),
        )

        # Output file names embed the epoch range
        fname_base = f"temporal_instability_w{width}_{mode}_{epoch_s}_{epoch_end}"
        svg_path   = os.path.join(save_dir, f"{fname_base}_irekawari_top100.svg")
        pdf_path   = os.path.join(save_dir, f"is_correct_0{fname_base}_irekawari.pdf")

    #     # SVG plot
        plot_aggregate_temporal_instability(
            stats_df    = stats_df,
            mode        = mode,
            xlabel      = "alpha",
            ylabel      = "Temporal Instability",
            save_path   = svg_path,
            log_scale_x = False,
            epoch_range = (epoch_s, epoch_end),
            y_lim       = (-0.01, 0.83),  # Specify y-axis range
            abc         = current_abc,    # Pass the current abc list here!
        )
        # PDF plot
        plot_aggregate_temporal_instability(
            stats_df    = stats_df,
            mode        = mode,
            xlabel      = "alpha",
            ylabel      = "Temporal Instability",
            save_path   = pdf_path,
            log_scale_x = False,
            epoch_range = (epoch_s, epoch_end),
            y_lim       = (-0.01, 0.83),  # Specify y-axis range
            abc         = current_abc,    # Pass the current abc list here!
        )
    
    # epoch_s=0
    # epoch_end=1000
    
    
    # ----------------------------------------spatial------------------------------------
def spatial():
    mode      = "no_noise"          # "noise" or "no_noise"
    width     = 64       # ディレクトリ階層で使う幅
    base_root = f"alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/no_noise"
    # base_root = f"alpha_test/emnist_digits/0.2/{width}/{mode}"
    save_dir  = "vizualize/miru_oral"
    # base_root = "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/noise"
    # base_root = "alpha_test/emnist_digits/0.2/64_7_20/noise_random"
    sample_dirs = get_sample_dirs(base_root)

    stats_df = aggregate_instability_across_samples(
            sample_dirs = sample_dirs,
            target      = "combined",
            mode        = "alpha",      # temporal instability
            y_scale     = "raw",
            epoch_range = None,
    )

    svg_path = os.path.join(save_dir, f"spatial_instability_width_{width}{mode}_raw.svg")
    pdf_path = os.path.join(save_dir, f"spatial_instability_width_{width}{mode}_raw.pdf")

    # SVG 出力
    plot_aggregate_instability(
        stats_df    = stats_df,
        xlabel      = "epoch",
        ylabel      = "Spatial Instability",
        save_path   = svg_path,
        log_scale_x = True,
        y_lim       = (-0.0001, 0.012),
        highlight=[1,27,55,120]
        # highlight=[30,80]
        # highlight=[100]
    )

    # PDF 出力
    plot_aggregate_instability(
        stats_df    = stats_df,
        xlabel      = "epoch",
        ylabel      = "Spatial Instability",
        save_path   = pdf_path,
        log_scale_x = True,
        # y_lim       = (-0.0001, 0.012),
        highlight=[1,27,55,120]
        # highlight=[30,80]
        # highlight=[100]

    )
        
if __name__ == '__main__':
    # spatial()
    #temporal()
    temporal_gattai()
    
    
    # print("[Info] Analyzing temporal instability for all alpha values...")
    # try:
    #     analyze_all_temporal_instability(
    #         base_root='/workspace/alpha_test/cifar10/0.2',
    #         widths=[1, 2, 4, 8,10,12,16, 32, 64],
    #         output_dir='/workspace/alpha_test/cifar10/0.2/temporal_instability_analysis',
    #         target_row=None  # すべてのalphaを解析
    #     )
    # except Exception as e:
    #     print(f"[Error] Failed to analyze temporal instability: {e}")
# if __name__ == '__main__':
    # print("[Info] Plotting mean match rates per epoch...")
    # try:
    #     plot_mean_match_rates_per_epoch(
    #         base_dir="/workspace/alpha_test/cifar10/0.2/64/noise",
    #         plot_save_path="/workspace/alpha_test/cifar10/0.2/64/noise/fig/match.png",
    #         csv_save_path="/workspace/alpha_test/cifar10/0.2/64/noise/fig/match.csv"
    #     )
    # except Exception as e:
    #     print(f"[Error] Failed to plot match rates: {e}")


# if __name__ == '__main__':
#     # 使用例
#     save_label_change_to_csv_with_sample_dirs(
#         base_root='alpha_test/cifar10/0.2',
#         widths=[1, 2, 4, 8, 10, 12, 16, 32, 64],
#         target_epoch=4000,  # 注目するエポックを指定
#         save_path='alpha_test/cifar10/0.2/label_change_results.csv'
#     )
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(
#         description='Instability Analysis: compute spatial/temporal instability per sample and aggregate.'
#     )
#     parser.add_argument('base_dir',help='Base directory containing sample subdirectories (two levels down with csv/ and fig_and_log/)'  )
#     parser.add_argument('--mode', choices=['alpha','epoch'], default='alpha',help="""'alpha': spatial instability per epoch (predict label changes along alpha axis)'epoch': temporal instability per alpha (changes across epochs)""")
#     parser.add_argument('--y_scale', choices=['ratio','percent','raw'], default='ratio',help='Scale for instability scores')
#     parser.add_argument('--epoch_start', type=int, default=None, help='Start epoch filter')
#     parser.add_argument('--epoch_end',   type=int, default=None, help='End epoch filter')
#     parser.add_argument('--aggregate', action='store_true', help='After per-sample eval, aggregate across samples')
#     parser.add_argument('--target', default='combined', help='Target name for aggregation CSV suffix')
#     parser.add_argument('--plot_save', default='aggregate_instability.svg', help='Save path for aggregated plot')

#     args = parser.parse_args()

#     # サンプルディレクトリ一覧取得
#     samples = get_sample_dirs(args.base_dir)
#     # if not samples:
#     #     print(f"[Error] No sample directories found under {args.base_dir}")
#     #     sys.exit(1)


#     # for sample in samples:
#     #     csv_dir = os.path.join(sample, 'csv')
#     #     out_dir = os.path.join(sample, 'fig_and_log')
#     #     print(f"[Info] Evaluating sample: {sample}")
#     #     try:
#     #         evaluate_label_changes(
#     #             pair_csv_dir=csv_dir,
#     #             output_dir=out_dir,
#     #             mode=args.mode,
#     #             y_scale=args.y_scale,
#     #             epoch_start=args.epoch_start,
#     #             epoch_end=args.epoch_end,
#     #             plot=False
#     #         )
#     #     except Exception as e:
#     #         print(f"[Error] Failed to evaluate {sample}: {e}")

#     if args.aggregate:
#         stats_df = aggregate_instability_across_samples(
#             sample_dirs=samples,
#             target=args.target,
#             mode=args.mode,
#             y_scale=args.y_scale,
#             epoch_range=(args.epoch_start, args.epoch_end) if args.epoch_start is not None or args.epoch_end is not None else None
#         )
#         plot_aggregate_instability(
#             stats_df=stats_df,
#             xlabel='Epoch' if args.mode == 'alpha' else 'Alpha',
#             ylabel=f"{'Spatial' if args.mode == 'alpha' else 'Temporal'} Instability ({args.y_scale})",
#             save_path=args.plot_save,
#             log_scale_x=(args.mode == 'alpha'),
#             y_lim=None  # 必要なら(0,1)など指定
#         )    # # 各サンプルごとの評価
#     # for s in samples:
#     #     csv_dir = os.path.join(s, 'csv')
#     #     out_dir = os.path.join(s, 'fig_and_log')
#     #     print(f"[Info] Evaluating sample: {s}")
#     #     try:
#     #         evaluate_label_changes(
#     #             pair_csv_dir=csv_dir,
#     #             output_dir=out_dir,
#     #             mode=args.mode,
#     #             y_scale=args.y_scale,
#     #             epoch_start=args.epoch_start,
#     #             epoch_end=args.epoch_end,
#     #             plot=True
#     #         )
#     #     except Exception as e:
#     #         print(f"[Warning] Failed sample {s}: {e}")

#     # # 集約処理
#     # if args.aggregate:
#     #     print("[Info] Aggregating across samples...")
#     #     erange = (args.epoch_start, args.epoch_end) if args.epoch_start is not None or args.epoch_end is not None else None
#     #     stats_df = aggregate_instability_across_samples(
#     #         sample_dirs=samples,
#     #         target=args.target,
#     #         mode=args.mode,
#     #         y_scale=args.y_scale,
#     #         epoch_range=erange
#     #     )
#     #     if stats_df.empty:
#     #         print("[Error] No data to aggregate.")
#     #         sys.exit(1)
#     #     xlabel = 'Epoch' if args.mode == 'alpha' else 'Alpha'
#     #     ylabel = ('Spatial Instability' if args.mode=='alpha' else 'Temporal Instability') + f" ({args.y_scale})"
#     #     plot_aggregate_instability(
#     #         stats_df,
#     #         xlabel=xlabel,
#     #         ylabel=ylabel,
#     #         save_path=args.plot_save,
#     #         highlight=None,
#     #         log_scale_x=(args.mode=='alpha')
#     #     )

# # python viz_colored_cifar_alpha_sample_gattai.py \
# #   alpha_test/cifar10/0.0/64/no_noise/ \
# #   --mode alpha \
# #   --y_scale raw \
# #   --epoch_start 0 \
# #   --epoch_end  4000\
# #   --aggregate \
# #   --target combined \
# #   --plot_save alpha_test/cifar10/0.0/64/no_noise/spatial_instability.png
