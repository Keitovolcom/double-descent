SEED=42
MODEL="resnet18k"
MODEL_WIDTH=64
EPOCH=4000
DATASET="cifar10"
GRAY_SCALE=false
BATCH=128
LR=0.0001
OPTIMIZER="adam"
MOMENTUM=0.0
LOSS="cross_entropy"
GPU=1
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"
USE_SAVE_DATA=false
LABEL_NOISE_RATE=0.0

python dd_scratch_model_save.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target "combined" \
  --label_noise_rate $LABEL_NOISE_RATE \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --gpu $GPU \
  --num_workers $NUM_WORKERS \
  $(if $WANDB; then echo "--wandb"; fi) \
  $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY