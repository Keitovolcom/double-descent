#!/bin/bash

# Default arguments
SEED=42
# seed値を必ずいじってください
EPOCH=1000
DATASET="emnist_digits"
GRAY_SCALE=false
BATCH_SIZE=128
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_distance2"
WANDB_ENTITY="dsml-kernel24"

# 単一のモデル、パラメータ設定で実行
MODEL="cnn_5layers_cus"
MODEL_WIDTH=64
LABEL_NOISE_RATE=0.2
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=0.0
GPU=0
MODE="noise"
DISTANCE_METRIC="euclidean"

echo "[Running] MODEL_WIDTH=$MODEL_WIDTH"

python alpha_interpolation.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target combined \
  --label_noise_rate $LABEL_NOISE_RATE \
  --variance $VAR \
  --correlation 0.5 \
  $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --num_workers $NUM_WORKERS \
  --gpu $GPU \
  $(if [ "$WANDB" = true ]; then echo '--wandb'; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY \
  --weight_noisy $WEIGHT_NOISY \
  --weight_clean $WEIGHT_CLEAN \
  --mode $MODE \
  --distance_metric $DISTANCE_METRIC

echo "[✓ Done] MODEL_WIDTH=$MODEL_WIDTH"
