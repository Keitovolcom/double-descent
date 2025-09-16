#!/usr/bin/env python
"""
main.py — Misclassification counting over an epoch window.
Generates a CSV with columns:
index, misclassified_count, is_noisy, s_epoch_correct

* misclassified_count : number of times the sample was mis‑classified between
  epochs [epoch_t, epoch_k].
* s_epoch_correct     : 1 if the sample is correct at `s_epoch`, else 0.

Assumes pre‑trained checkpoints saved as:
 save_model/{dataset}/noise_{label_noise_rate}/{experiment_name}/epoch_{n}.pth
"""

import os
import csv
import warnings
import argparse
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- project utilities -------------------------------------------------------
# These must exist in your code‑base. If module paths differ, adjust imports.
from utils import set_seed, set_device
from datasets import load_or_create_noisy_dataset
from models import load_models
from torch.distributed import is_initialized, get_rank

# -----------------------------------------------------------------------------
# 1. Argument parser
# -----------------------------------------------------------------------------

def parse_args_model_save_separate() -> argparse.Namespace:
    """Argument set shared with the training pipeline, plus counting options."""
    p = argparse.ArgumentParser(
        description="Count misclassifications of train samples over epoch window")

    # Dataset / model params --------------------------------------------------
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--target', type=str, default='digits')
    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--model_width', type=int, default=64)
    p.add_argument('--epoch', type=int, default=200,
                   help='Max training epoch (used only for experiment name)')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--optimizer', type=str, default='sgd')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--fix_seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=2)

    # Noise / dataset variations ---------------------------------------------
    p.add_argument('--label_noise_rate', type=float, default=0.0)
    p.add_argument('--variance', type=float, default=0.0)
    p.add_argument('--gray_scale', action='store_true')

    # Counting window ---------------------------------------------------------
    p.add_argument('--s_epoch', type=int, default=10,
                   help='Epoch at which s_epoch_correct is evaluated')
    p.add_argument('--epoch_t', type=int, default=11,
                   help='Start epoch (inclusive) of counting window')
    p.add_argument('--epoch_k', type=int, default=50,
                   help='End epoch (inclusive) of counting window')

    return p.parse_args()

# -----------------------------------------------------------------------------
# 2. Dataset unwrapping helpers (copied from original project)
# -----------------------------------------------------------------------------

def unwrap_to_tensor_dataset(ds):
    """Recursively unwrap wrappers to obtain the core TensorDataset."""
    visited = set()
    while True:
        if isinstance(ds, TensorDataset):
            return ds
        if hasattr(ds, 'dataset'):
            ds = ds.dataset
        elif hasattr(ds, 'base_dataset'):
            ds = ds.base_dataset
        else:
            raise TypeError(f'Cannot unwrap dataset of type {type(ds)}')
        if id(ds) in visited:
            raise RuntimeError('Circular reference detected while unwrapping dataset')
        visited.add(id(ds))

def get_tensor_dataset_components(ds) -> Tuple[torch.Tensor, torch.Tensor]:
    core = unwrap_to_tensor_dataset(ds)
    return core.tensors  # (inputs, targets)

# -----------------------------------------------------------------------------
# 3. Inference helpers
# -----------------------------------------------------------------------------

def run_inference(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> np.ndarray:
    """Return boolean array (N,) indicating correctness for each sample."""
    model.eval()
    correct_flags = []
    with torch.no_grad():
        for x, y, _ in dl:  # (_, _, index)
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct_flags.extend(preds.eq(y).cpu().numpy())
    return np.asarray(correct_flags, dtype=np.bool_)

# -----------------------------------------------------------------------------
# 4. Counting core logic
# -----------------------------------------------------------------------------

def count_misclassifications(args, device):
    # 4‑1. Load dataset -------------------------------------------------------
    train_ds, _, meta = load_or_create_noisy_dataset(
        args.dataset, args.target, args.gray_scale, args, return_type='torch')

    noise_info = getattr(train_ds, 'noise_info', None)  # Tensor or None

    x_train, y_train = get_tensor_dataset_components(train_ds)
    indices = torch.arange(len(x_train))
    idx_ds = TensorDataset(x_train, y_train, indices)
    loader = DataLoader(idx_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # 4‑2. Load model skeleton ----------------------------------------------
    model = load_models(meta['in_channels'], args,
                        meta['imagesize'], meta['num_classes']).to(device)

    # exp_name = (
    #     f"test_seed_{args.fix_seed}width{args.model_width}_{args.model}_"
    #     f"{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_"
    #     f"batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_"
    #     f"Optim{args.optimizer}_Momentum{args.momentum}")
    exp_name="7_14_use_mixup_False_alpha0.0_test_seed_42width64_cnn_5layers_cus_emnist_digits_variance0_combined_lr0.01_batch128_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0"
    base_dir = f"save_model/{args.dataset}/noise_{args.label_noise_rate}/{exp_name}"

    n_samples = len(train_ds)
    miscount = np.zeros(n_samples, dtype=np.int32)

    # 4‑3. Compute s_epoch_correct ------------------------------------------
    path_s = os.path.join(base_dir, f"model_epoch_{args.s_epoch}.pth")
    if not os.path.exists(path_s):
        raise FileNotFoundError(f'Reference epoch checkpoint not found: {path_s}')

    model.load_state_dict({k: v.float() for k, v in torch.load(path_s, map_location=device).items()})
    s_epoch_correct = run_inference(model, loader, device).astype(np.int8)  # 1/0

    # 4‑4. Loop over counting window ----------------------------------------
    for ep in range(args.epoch_t, args.epoch_k + 1):
        ckpt_path = os.path.join(base_dir, f"model_epoch_{ep}.pth")
        if not os.path.exists(ckpt_path):
            print(f'[WARN] checkpoint missing for epoch {ep}, skipping.')
            continue
        model.load_state_dict({k: v.float() for k, v in torch.load(ckpt_path, map_location=device).items()})
        correct = run_inference(model, loader, device)
        miscount += (~correct).astype(np.int32)
        torch.cuda.empty_cache()

    # 4‑5. Save CSV ----------------------------------------------------------
    out_dir = os.path.join(base_dir, 'miscount_full_csv')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                           f"miscount_s{args.s_epoch}_t{args.epoch_t}k{args.epoch_k}.csv")

    header = ['index', 'misclassified_count', 'is_noisy', 's_epoch_correct']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for idx in range(n_samples):
            row = [idx, int(miscount[idx])]
            if noise_info is not None:
                v = noise_info[idx]
                if torch.is_tensor(v):
                    v = v.item()
                row.append(int(v))
            else:
                row.append('')
            row.append(int(s_epoch_correct[idx]))
            wr.writerow(row)

    print(f'[DONE] saved to → {out_path}')

# -----------------------------------------------------------------------------
# 5. Utilities
# -----------------------------------------------------------------------------

def is_main_process() -> bool:
    return (not is_initialized()) or get_rank() == 0

# -----------------------------------------------------------------------------
# 6. Entrypoint
# -----------------------------------------------------------------------------

def main():
    warnings.filterwarnings('ignore')
    args = parse_args_model_save_separate()
    set_seed(args.fix_seed)
    device = set_device(args.gpu)

    if is_main_process():
        print('👉 Counting misclassifications...')
    count_misclassifications(args, device)
    if is_main_process():
        print('✅ Finished.')


if __name__ == '__main__':
    main()
