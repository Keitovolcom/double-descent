# eval_mis_on_epochs.py
import os
import csv
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ==== あなたのプロジェクトのモジュール ====
from config import parse_args_model_save
from utils import set_seed, set_device, seed_worker
from datasets import load_or_create_noisy_dataset
from models import load_models
# ===========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def reconstruct_experiment_name(args, wandb_style_dir: bool) -> str:
    """
    あなたの main() と同じ命名を再現。wandb 有効時と無効時で文言が微妙に違うので切替可能。
    """
    if wandb_style_dir:
        # main() 内の wandb 有効名
        name = (
            f'7_14_use_mixup_{args.use_mixup}_alpha{args.mixup_alpha}_test_seed_{args.fix_seed}'
            f'width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}'
            f'_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}'
            f'_Optim{args.optimizer}_Momentum{args.momentum}'
        )
    else:
        # main() 内の wandb 無効名
        name = (
            f'7_14_use_mixup_{args.use_mixup}_alpha{args.mixup_alpha}_seed_{args.fix_seed}'
            f'width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}'
            f'_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}'
            f'_Optim{args.optimizer}_Momentum{args.momentum}'
        )
    return name

def build_base_save_dir(args, wandb_style_dir: bool) -> str:
    exp = reconstruct_experiment_name(args, wandb_style_dir)
    return f"save_model/{args.dataset}/noise_{args.label_noise_rate}/{exp}"

def load_ckpt_into_model(model: nn.Module, ckpt_path: str) -> nn.Module:
    """あなたの保存形式に合わせて state_dict をロード（half→float32へ）。"""
    state = torch.load(ckpt_path, map_location="cpu")

    # 2 形式に対応: 1) state_dictそのもの  2) {"model_state_dict": ...}
    if isinstance(state, dict) and all(torch.is_tensor(v) for v in state.values()):
        sd = state
    elif isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")

    # half → float32 変換（保存時に half だったため）
    sd = {k: v.float() if torch.is_tensor(v) else v for k, v in sd.items()}

    # DDPで保存されていない想定（必要なら 'module.' プレフィクス除去を追加）
    # 一応両対応
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def get_misclassified_indices(model: nn.Module, loader: DataLoader) -> List[int]:
    mis = []
    base = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[:2]
        else:
            x, y = batch["image"], batch["label"]
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(1)
        wrong = (pred != y).cpu()

        idxs = torch.arange(base, base + x.size(0))
        mis.extend(idxs[wrong].tolist())
        base += x.size(0)
    return mis

@torch.no_grad()
def eval_mean_loss(model: nn.Module, subset_loader: DataLoader, criterion) -> Tuple[float, int]:
    total_loss = 0.0
    total_n = 0
    for batch in subset_loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[:2]
        else:
            x, y = batch["image"], batch["label"]
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)  # (B,)
        total_loss += loss.sum().item()
        total_n += loss.numel()
    return (total_loss / max(total_n, 1)), total_n

def save_indices_csv(indices: List[int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_index"])
        for i in indices:
            w.writerow([i])

def run(anchor_epoch: int,
        eval_epochs: List[int],
        base_save_dir: str,
        batch_size: int,
        num_workers: int,
        out_dir: str,
        ckpt_fmt: str):
    os.makedirs(out_dir, exist_ok=True)

    # 学習時と同じパラメータでデータとモデルを構築
    args = parse_args_model_save()
    set_seed(args.fix_seed)
    _ = set_device(args.gpu)

    # データ読み込み（testのみ必要）
    train_ds, test_ds, meta = load_or_create_noisy_dataset(
        args.dataset, args.target, args.gray_scale, args, return_type="torch"
    )
    num_classes = meta["num_classes"]
    in_channels = meta["in_channels"]
    imagesize = meta["imagesize"]

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker
    )

    # モデル骨格
    model = load_models(in_channels, args, imagesize, num_classes)

    # 1) アンカー epoch で誤分類インデックス抽出
    ckpt_anchor = os.path.join(base_save_dir, ckpt_fmt.format(epoch=anchor_epoch))
    if not os.path.exists(ckpt_anchor):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_anchor}")
    model = load_ckpt_into_model(model, ckpt_anchor)
    print(f"[Anchor] epoch={anchor_epoch} → 推論中（誤分類抽出）")
    mis_idx = get_misclassified_indices(model, test_loader)
    print(f"  誤分類: {len(mis_idx)} / {len(test_ds)}")

    idx_csv = os.path.join(out_dir, f"misclassified_indices_epoch{anchor_epoch}.csv")
    save_indices_csv(mis_idx, idx_csv)
    if len(mis_idx) == 0:
        print("誤分類が0件のため終了します。")
        return

    # Subset & Loader
    subset = Subset(test_ds, mis_idx)
    subset_loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 2) 各 epoch で loss を計測
    rows: List[Dict] = []
    print(f"[Eval] epochs = {eval_epochs[0]}..{eval_epochs[-1]} (|S|={len(mis_idx)})")
    for ep in eval_epochs:
        ckpt_path = os.path.join(base_save_dir, ckpt_fmt.format(epoch=ep))
        if not os.path.exists(ckpt_path):
            print(f"  [Skip] no checkpoint: {ckpt_path}")
            continue

        model = load_models(in_channels, args, imagesize, num_classes)  # 新規に作る
        model = load_ckpt_into_model(model, ckpt_path)

        mean_loss, n = eval_mean_loss(model, subset_loader, criterion)
        print(f"  epoch={ep:4d} | mean_loss_on_mis={mean_loss:.6f} | N={n}")
        rows.append({"epoch": ep, "mean_loss_on_mis": mean_loss, "N": n})

    # 3) CSV 保存
    loss_csv = os.path.join(out_dir, f"loss_over_epochs_anchor{anchor_epoch}.csv")
    with open(loss_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "mean_loss_on_mis", "N"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Save] {loss_csv}")

    # 4) 図保存
    if rows:
        xs = [r["epoch"] for r in rows]
        ys = [r["mean_loss_on_mis"] for r in rows]
        plt.figure(figsize=(8,5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss (misclassified@anchor)")
        plt.title(f"Loss on misclassified samples at anchor epoch {anchor_epoch}")
        plt.grid(True, alpha=0.3)
        fig_path = os.path.join(out_dir, f"loss_curve_anchor{anchor_epoch}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=220)
        print(f"[Save] {fig_path}")

def parse_cli():
    p = argparse.ArgumentParser(description="Evaluate loss over epochs on misclassified@anchor")
    p.add_argument("--anchor_epoch", type=int, required=True)
    p.add_argument("--eval_start", type=int, required=True)
    p.add_argument("--eval_end", type=int, required=True)
    p.add_argument("--eval_step", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./mis_eval")
    p.add_argument("--ckpt_fmt", type=str, default="model_epoch_{epoch}.pth",
                   help="チェックポイントのファイル名フォーマット")
    # base_save_dir は明示指定も再現生成も可
    p.add_argument("--base_save_dir", type=str, default="",
                   help="未指定なら訓練時の命名規則から再構築")
    p.add_argument("--wandb_style_dir", action="store_true",
                   help="訓練時に wandb をONにしていた命名（test_seed〜）を使って再構築する場合に指定")
    args_cli = p.parse_args()

    # parse_args_model_save() は run() 内で呼ぶ（実験名再現のため）
    return args_cli

if __name__ == "__main__":
    cli = parse_cli()

    # eval epochs
    eval_epochs = list(range(cli.eval_start, cli.eval_end + 1, cli.eval_step))

    # base_save_dir を決める
    if cli.base_save_dir:
        base = cli.base_save_dir
    else:
        # config から再構築（実際の run() 内で同じ parse を使うため、
        # ここでは一度だけ parse して名前だけ決めたい場合は run 側に任せてもOK）
        args_tmp = parse_args_model_save()
        base = build_base_save_dir(args_tmp, cli.wandb_style_dir)

    run(
        anchor_epoch=cli.anchor_epoch,
        eval_epochs=eval_epochs,
        base_save_dir=base,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        out_dir=cli.out_dir,
        ckpt_fmt=cli.ckpt_fmt,
    )
