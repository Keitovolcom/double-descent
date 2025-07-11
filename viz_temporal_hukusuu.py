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
import glob

# ===================================================================
# Matplotlibのグローバル設定
# ===================================================================
plt.rcParams["font.size"] = 23
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 400
plt.rcParams['font.family'] = 'DejaVu Sans'


# ===================================================================
# ヘルパー関数群
# ===================================================================

def get_sample_dirs(base_dir: str) -> List[str]:
    """
    base_dir 以下の 2 階層下で "fig_and_log" を含むディレクトリを返す。
    """
    sample_dirs = []
    if not os.path.isdir(base_dir):
        return []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if not os.path.isdir(d_path):
            continue
        for sub in os.listdir(d_path):
            sub_path = os.path.join(d_path, sub)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "fig_and_log")):
                sample_dirs.append(sub_path)
    return sample_dirs

def list_epoch_files(
    csv_dir: str,
    start: Optional[int] = None,
    end: Optional[int] = None
) -> List[Tuple[int, str]]:
    """
    csv_dir 内の epoch_{n}.csv を (epoch, filepath) のリストで返す。
    """
    files = []
    if not os.path.isdir(csv_dir):
        return []
    for fname in os.listdir(csv_dir):
        match = re.match(r'epoch_(\d+)\.csv$', fname)
        if not match:
            continue
        epoch = int(match.group(1))
        if start is not None and epoch < start:
            continue
        if end is not None and epoch > end:
            continue
        files.append((epoch, os.path.join(csv_dir, fname)))
    files.sort(key=lambda x: x[0])
    return files

def compute_temporal_instability(
    dfs: List[pd.DataFrame],
    y_scale: Literal['ratio', 'percent', 'raw'] = 'raw'
) -> Dict[float, float]:
    """
    複数 epoch の df リストを受け取り、各 alpha ごとに変化回数を計算。
    """
    if len(dfs) < 2:
        return {}
    alphas = dfs[0]['alpha'].to_numpy()
    M = len(alphas)
    counts = np.zeros(M, dtype=int)
    for i in range(len(dfs) - 1):
        cur = dfs[i]['predicted_label'].to_numpy()
        nxt = dfs[i+1]['predicted_label'].to_numpy()
        counts += (cur != nxt).astype(int)
    if y_scale == 'ratio':
        scores = counts / (len(dfs) - 1)
    elif y_scale == 'percent':
        scores = (counts / (len(dfs) - 1)) * 100
    else:
        scores = counts.astype(float)
    return {alpha: float(score) for alpha, score in zip(alphas, scores)}

def evaluate_label_changes(
    pair_csv_dir: str,
    output_dir: str,
    mode: Literal['alpha','epoch'] = 'alpha',
    y_scale: Literal['ratio','percent','raw'] = 'raw',
    epoch_start: Optional[int] = None,
    epoch_end: Optional[int] = None,
    plot: bool = True
) -> Dict[float,float]:
    """
    単一サンプルに対して不安定性を計算し、CSV/プロットとして保存。
    """
    files = list_epoch_files(pair_csv_dir, epoch_start, epoch_end)
    if not files:
        raise ValueError("No epoch CSV files found in directory.")
    suffix = ''
    if epoch_start is not None or epoch_end is not None:
        s = str(epoch_start) if epoch_start is not None else 'start'
        e = str(epoch_end)   if epoch_end   is not None else 'end'
        suffix = f"_epoch_{s}_to_{e}"

    if mode == 'epoch':
        dfs = [pd.read_csv(fp) for _, fp in files]
        scores = compute_temporal_instability(dfs, y_scale)
        df_out = pd.DataFrame({'alpha': list(scores.keys()), 'unsmoothed_scores': list(scores.values())})
        csv_path = os.path.join(output_dir, f'epoch_unsmoothed_scores{suffix}.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_out.to_csv(csv_path, index=False)
        return scores
    return {}

def aggregate_instability_across_samples(
    sample_dirs: List[str],
    target: str,
    mode: Literal['alpha', 'epoch'],
    y_scale: Literal['ratio', 'percent', 'raw'],
    epoch_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    各サンプルのスコアCSVを読み込み、x_valueごとのmean/stdを計算。
    """
    suffix = ''
    if epoch_range:
        s, e = epoch_range
        suffix = f"_epoch_{s}_to_{e}"

    rows = []
    for d in sample_dirs:
        base = os.path.join(d, 'fig_and_log')
        fname = f'epoch_unsmoothed_scores{suffix}.csv'
        x_col, y_col = 'alpha', 'unsmoothed_scores'
        fpath = os.path.join(base, fname)

        if not os.path.exists(fpath):
            try:
                pair_csv_dir = os.path.join(d, 'csv')
                evaluate_label_changes(
                    pair_csv_dir=pair_csv_dir,
                    output_dir=base,
                    mode='epoch',
                    y_scale=y_scale,
                    epoch_start=epoch_range[0] if epoch_range else None,
                    epoch_end=epoch_range[1] if epoch_range else None,
                    plot=False
                )
            except Exception as e:
                print(f"[Warn] Failed to generate scores for {d}: {e}")
                continue

        if not os.path.exists(fpath):
            print(f"[Warn] missing {fpath}")
            continue

        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            rows.append((row[x_col], row[y_col]))

    if not rows:
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

def draw_on_ax_temporal_instability(
    ax: plt.Axes,
    stats_df: pd.DataFrame,
    xlabel: str,
    *,
    mode: Literal["no_noise", "noise"] = "no_noise",
    epoch_range: Optional[Tuple[int, int]] = None,
    log_scale_x: bool = False,
    y_lim: Optional[Tuple[float, float]] = None,
    marker_size: int = 15,
    abc: Optional[List[Tuple[str, str]]] = None # ★abcの型を明示
) -> None:
    """与えられたaxオブジェクト上に、集計されたinstabilityの結果をプロットする。"""
    x = stats_df["x_value"].to_numpy()
    y = stats_df["mean_score"].to_numpy()
    std = stats_df["std_score"].to_numpy()

    if xlabel.lower() == "alpha" and epoch_range is not None:
        denom = max(epoch_range[1] - epoch_range[0], 1)
        y = y / denom
        std = std / denom

    ax.plot(x, y, linewidth=2, zorder=3,
            label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color="blue")
    ax.fill_between(x, y - std, y + std,
                    alpha=0.2, zorder=2, color="blue")

    if xlabel.lower() == "alpha":
        # ★★★ abcパラメータを使って凡例を動的に生成するロジックを修正 ★★★
        if abc and isinstance(abc, list) and all(isinstance(t, tuple) and len(t) == 2 for t in abc):
            for p1, p2 in abc:
                if epoch_range is not None:
                    ax.plot([], [], label=rf'$\mathcal{{T}}_{{{p1}{p2}}}$=[{epoch_range[0]},{epoch_range[1]}]', linewidth=0)
                else:
                    ax.plot([], [], label=rf'$\mathcal{{T}}_{{{p1}{p2}}}$', linewidth=0)
        
        ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)
        if mode == "no_noise":
            ax.plot([0.0, 1.0], [0.0, 0.0], "o",
                    color="blue", markersize=marker_size, zorder=5)
        else: # mode == "noise"
            ax.plot(0.0, 0.0, "o", color="blue", markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, "o", color="red", markersize=marker_size, zorder=5)

    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r'$x_{0}$', r'$x_{1}$'], fontsize=30)
    if y_lim:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale("log")
    ax.legend()
    ax.grid(True)


# ===================================================================
# メインの実行ブロック
# ===================================================================

def main():
    """
    1行x3列のレイアウトで、指定されたepoch_rangeとabcラベルのグラフを生成する。
    """
    # --- 基本設定 (ご自身の環境に合わせて修正してください) ---
    mode = "noise"  # "noise" or "no_noise"
    width = 8
    dataset_root = "alpha_test/emnist_digits/0.2/8/"
    save_dir = "vizualize/abc_comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    base_root = f"{dataset_root}/{mode}"

    # --- 描画設定 (AB, BC, CD) ---
    plot_configs = [
        {"epoch_range": (0, 27),   "abc": [("A", "B")]},
        {"epoch_range": (27, 55),  "abc": [("B", "C")]},
        {"epoch_range": (55, 120), "abc": [("C", "D")]},
    ]
    num_plots = len(plot_configs)

    # --- グラフの準備 (1行3列、Y軸を共有) ---
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5.5), sharey=True, dpi=300)
    
    # 単一のプロットしかない場合にaxesがリストにならない問題に対応
    if num_plots == 1:
        axes = [axes]

    print(f"--- Processing plots for mode={mode} ---")
    try:
        sample_dirs = get_sample_dirs(base_root)
        if not sample_dirs:
            print(f"[Error] No sample directories found in: {base_root}")
            return
    except Exception as e:
        print(f"[Error] Failed to get sample dirs for {base_root}: {e}")
        return

    # --- ループ処理で各サブプロットに描画 ---
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        epoch_range = config["epoch_range"]
        current_abc = config["abc"]

        print(f"  Processing Plot {i+1}/{num_plots}: Epochs {epoch_range} with label {current_abc[0]}")

        stats_df = aggregate_instability_across_samples(
            sample_dirs=sample_dirs,
            target="combined",
            mode="epoch",
            y_scale="raw",
            epoch_range=epoch_range,
        )

        if stats_df.empty:
            print(f"  [Warning] No data found for epoch_range={epoch_range}")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue

        draw_on_ax_temporal_instability(
            ax=ax,
            stats_df=stats_df,
            xlabel="alpha",
            mode=mode,
            epoch_range=epoch_range,
            y_lim=(-0.01, 0.63), # 必要に応じてY軸の範囲を調整
            abc=current_abc,
        )

    # --- 見た目の調整 ---
    # Y軸ラベルは一番左のプロットにのみ表示
    axes[0].set_ylabel("Temporal Instability")
    fig.tight_layout()
    
    # --- 保存 ---
    fname_base = f"temporal_instability_w{width}_{mode}_ABC_comparison"
    save_path = os.path.join(save_dir, f"{fname_base}.svg")
    print(f"\n[✓] Saving combined plot to: {save_path}")
    fig.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    main()