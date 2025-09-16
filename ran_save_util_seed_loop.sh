#!/usr/bin/env bash
# ==========================================================
#  multi_seed_run.sh
#  Run dd_scratch_model_save.py for multiple random seeds
#  Usage:  bash multi_seed_run.sh
# ==========================================================
set -euo pipefail            # 安全策: エラーで即終了 & 未定義変数はエラー

# ---------- 共通ハイパーパラメータ -------------------------
MODEL="cnn_5layers_cus"
MODEL_WIDTH=64
DATASET="emnist_digits"
LR=0.01
BATCH_SIZE=256
EPOCH=2000
LABEL_NOISE_RATE=0.2
OPTIMIZER="sgd"
MOMENTUM=0.0
GPU=2
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
NUM_WORKERS=4
LOSS="cross_entropy"
USE_SAVED_DATA=false

# ---------- カラー設定（使わない場合は無視） ---------------
VARIANCE=0
CORR=0.5

# ---------- Weights & Biases ------------------------------
WANDB=false 
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"

# ---------- ループさせたいシード値 -------------------------
# SEEDS=(43 44 45)
# SEEDS=(46 47 48)
SEEDS=(42 51) # Uncomment to add more seeds
echo "Running experiments for seeds: ${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
  echo "===== Seed: ${SEED} ====="

  python dd_scratch_model_save.py \
    --fix_seed "$SEED" \
    --model "$MODEL" \
    --model_width "$MODEL_WIDTH" \
    --epoch "$EPOCH" \
    --dataset "$DATASET" \
    --target "combined" \
    --label_noise_rate "$LABEL_NOISE_RATE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --optimizer "$OPTIMIZER" \
    --momentum "$MOMENTUM" \
    --loss "$LOSS" \
    --gpu "$GPU" \
    --num_workers "$NUM_WORKERS" \
    $( $WANDB && echo "--wandb" ) \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

  echo "Finished seed ${SEED}"
  echo
done

echo "All runs completed."
