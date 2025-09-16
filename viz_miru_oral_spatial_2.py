# --- 既存インポート & WANDB 部分はそのまま ---
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib


import matplotlib.font_manager as fm

import matplotlib, pathlib, sys
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
from pathlib import Path
plt.rcParams["font.size"] = 50
plt.rcParams["figure.figsize"] = [13, 8]
plt.rcParams["figure.dpi"] = 400
plt.rcParams['font.family'] = 'Times New Roman'
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
            ax.axvline(v, color="black", linestyle="-",linewidth=2)

    ax.plot(x, y, linewidth=2, zorder=3,
            label="$INST_s(\\chi,t)$", color="blue")
    ax.fill_between(x, y - std, y + std,
                    alpha=0.2, zorder=2, color="blue")
    if highlight_epochs:
        for ep in highlight_epochs:
            ax.axvline(x=ep, color='black', linestyle='-', linewidth=2, zorder=1)
    # ---------- α = 0,1 の補助線 & マーカー ----------
    if xlabel.lower() == "alpha":
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=2)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=2)

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
def plot_aggregate_instability_multi(
    stats_dfs,                # List[pd.DataFrame]
    labels,                   # List[str]
    colors,                   # List[str]
    xlabel, ylabel, save_path,
    *, log_scale_x=False,
    y_lim=None, highlight=None,
    marker_size=10,
    highlight_epochs=None,
):
    """
    stats_dfs[i] は `aggregate_instability_across_samples()` が返す
    x_value / mean_score / std_score 列を持つ DataFrame
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    for stats_df, lab, col in zip(stats_dfs, labels, colors):
        x   = stats_df["x_value"].to_numpy()
        y   = stats_df["mean_score"].to_numpy() / 200      # ←必要なら正規化値に合わせて変更
        std = stats_df["std_score"].to_numpy()  / 200

        ax.plot(x, y, label=lab, color=col, linewidth=2, zorder=3)
        ax.fill_between(x, y-std, y+std, color=col, alpha=0.2, zorder=2)

    # 汎用の目盛り・ハイライト
    if highlight:
        for v in highlight:
            ax.axvline(v, color="black", linestyle="-")
    if highlight_epochs:
        for ep in highlight_epochs:
            ax.axvline(x=ep, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale("log")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Multi‑curve plot saved to: {save_path}")
print(pathlib.Path(matplotlib.get_data_path()) / "fonts/ttf")


# rebuild_if_missing=True で見つからなければ自動でキャッシュ再構築


# ===== 設定 =====
entity = "dsml-kernel24"
project = "kobayashi_emnist"
target_run_name = "cnn_5layers_width8_emnist_digits_lr0.01_batch_size128_epoch1000_LabelNoiseRate0.2_Optimsgd_momentum0.0"
output_base = "/workspace/vizualize/ACML/emnist_learning_curve2"

# ===== W&B Run一覧から目的のrun.idを検索 =====
api = wandb.Api()
run_id = None
for run in api.runs(f"{entity}/{project}"):
    if run.name == target_run_name:
        run_id = run.id
        break

if run_id is None:
    raise ValueError(f"[Error] Run name '{target_run_name}' が見つかりませんでした。")

# ===== run.idでログ取得 =====
run = api.run(f"{entity}/{project}/{run_id}")
history = run.history(keys=[
    "epoch",
    "test_error",
    "train_error_total",
    "train_error_clean",
    "train_error_noisy", # カンマが抜けていたのを修正
    "test_loss",
    "train_loss"
])
df = pd.DataFrame(history)
# "epoch"列にNaNが含まれる行を削除してからソート
df = df.dropna(subset=['epoch']).sort_values("epoch")


# ===== データ準備 =====
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

epochs = df["epoch"]
# NOTE: 全てのエラー率を100で割り、0.0-1.0の割合に変換します。
test_error = df["test_error"] / 100
train_error = df["train_error_total"] / 100
train_error_noisy = df["train_error_noisy"] / 100
train_error_clean = df["train_error_clean"] / 100
# lossデータを抽出
train_loss = df["train_loss"]
test_loss = df["test_loss"]


vertical_epochs = [1,27,55,120]  # 縦線を引きたいエポック（必要に応じて）
# ===== Instability 用の前処理 =====
# ── ① base_root を2本用意 ──
base_roots = [
    "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/no_noise",
    "alpha_test/emnist_digits/0.2/128_kyu_ver2_8_random/noise",
]
inst_labels = ["clean-clean", "clean-noisy"]   # 凡例用
inst_colors = ["blue", "red"]

# ===== Instability の統計を取得 =====
stats_list = []
for br in base_roots:
    sample_dirs = get_sample_dirs(br)                    # 既存 util 関数
    stats_df = aggregate_instability_across_samples(     # 既存 util 関数
        sample_dirs=sample_dirs,
        target="combined",
        mode="alpha",
        y_scale="raw",
        epoch_range=None,
    )
    stats_list.append(stats_df)

# ===== 2 段グラフ作成 =====
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]   = 20

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), constrained_layout=True)

# ──────────────────────────────────
# 1 段目：Test / Train error
# ──────────────────────────────────
ax1.plot(epochs, test_error,  label="test",  color="red",  linewidth=3)
ax1.plot(epochs, train_error, label="train", color="blue", linewidth=3)
ax1.plot(epochs, train_error_noisy, label="train noisy", color="blue", linewidth=3,linestyle='--')
ax1.set_ylabel("train noisy error", fontsize=55,)
for ep in vertical_epochs:
    ax1.axvline(x=ep, color="black", linestyle="-", linewidth=2, zorder=0)

ax1.set_xscale("log")
ax1.set_ylabel("error", fontsize=55)
ax1.set_ylim(-0.02, 1.02)
ax1.set_xlim(0.9, 1000)
ax1.set_yticks([0.0, 0.2, 0.4, 0.6,0.8,1.0])
ax1.legend(loc="upper right", fontsize=45)
ax1.tick_params(axis="both", labelsize=45, labelbottom=False)

ax3 = ax1.twinx()
ax3.plot(epochs, train_error_clean, label="train clean", color="blue", linewidth=3)
ax3.set_ylabel("train clean error", fontsize=55)
# NOTE: CleanのY軸範囲を0-4%に対応する割合 (-0.2% から 4.2%) に修正
ax3.legend(loc="upper left", fontsize=45)
ax3.set_ylim(-0.00235, 0.11) 
ax3.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])

ax3.tick_params(axis='y', labelsize=45)


# ──────────────────────────────────
# 2 段目：Spatial Instability (multi‑curve)
# ──────────────────────────────────
for stats_df, lab, col in zip(stats_list, inst_labels, inst_colors):
    x   = stats_df["x_value"].to_numpy()
    y   = stats_df["mean_score"].to_numpy() / 200   # ←元コードと同じ正規化
    std = stats_df["std_score"].to_numpy()  / 200

    ax2.plot(x, y, label=lab, color=col, linewidth=3, zorder=3)
    ax2.fill_between(x, y-std, y+std, color=col, alpha=0.2, zorder=2)

for ep in vertical_epochs:
    ax2.axvline(x=ep, color="black", linestyle="-", linewidth=2, zorder=0)

ax2.set_xscale("log")
ax2.set_xlabel("epoch", fontsize=55)
ax2.set_ylabel("Spatial Instability", fontsize=55)
ax2.set_xlim(0.9, 1000)
ax2.set_ylim(-0.0001, 0.012)
ax2.set_yticks([0.0, 0.01])
ax2.tick_params(axis="both", labelsize=45)
ax2.legend(loc="upper right", fontsize=45)


# ===== 保存 =====
plt.savefig(f"{output_base}_spatial_gattai.svg", format="svg")
plt.savefig(f"{output_base}_spatial_gattai.pdf", format="pdf")
print(f"[✓] Saved: {output_base}_spatial_gattai.pdf and .svg")

