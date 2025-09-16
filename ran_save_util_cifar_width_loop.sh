#!/bin/bash

# 実験設定（必要に応じて変更）
SEED=42
MODEL="resnet18k"
DATASET="cifar10"
LR=0.0001
BATCH_SIZE=128
EPOCH=4000
LABEL_NOISE_RATE=0.2
OPTIMIZER="adam"
MOMENTUM=0.0
GPU=2
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
NUM_WORKERS=4
LOSS="cross_entropy"
USE_SAVED_DATA=false
VARIANCE=0
CORR=0.5
WANDB=true
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"

# 幅のリスト
width_list=(6)

for MODEL_WIDTH in "${width_list[@]}"
do
  echo "Running with MODEL_WIDTH=${MODEL_WIDTH}"

  python dd_scratch_model_save.py \
    --fix_seed $SEED \
    --model $MODEL \
    --model_width $MODEL_WIDTH \
    --epoch $EPOCH \
    --dataset $DATASET \
    --label_noise_rate $LABEL_NOISE_RATE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --momentum $MOMENTUM \
    --loss $LOSS \
    --gpu $GPU \
    --num_workers $NUM_WORKERS \
    $(if $WANDB; then echo "--wandb"; fi) \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY
done
