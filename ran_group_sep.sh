#!/usr/bin/env bash
# =====================================================================
#  misclassification カウント用実行スクリプト
#  - 基準 epoch (s_epoch) での正誤フラグを付与しつつ
#    [epoch_t, epoch_k] 区間の不正解回数を CSV に保存する
# =====================================================================

# ---------- モデル設定 (学習時と同じ) --------------------------------
SEED=42
MODEL="cnn_5layers_cus"
MODEL_WIDTH=64
DATASET="emnist_digits"
TARGET="combined"
LR=0.01
BATCH_SIZE=128
EPOCH=1000
LABEL_NOISE_RATE=0.2
OPTIMIZER="sgd"
MOMENTUM=0.0
GPU=3
NUM_WORKERS=4
VARIANCE=0.0        # distribution_colored_emnist 等の場合に使用

# ---------- カウント区間設定 ------------------------------------------
# 基準となる epoch (s_epoch) ─ 正解/不正解フラグ用
S_EPOCH=30
# 区間の開始 epoch (inclusive)
EPOCH_T=30
# 区間の終了 epoch (inclusive)
EPOCH_K=140

# ---------- 実行 ------------------------------------------------------
echo "==============================================================="
echo " 🚀 misclassification カウント開始"
echo "  - モデル         : $MODEL (width=$MODEL_WIDTH)"
echo "  - データセット   : $DATASET"
echo "  - シード         : $SEED"
echo "  - 基準 epoch     : s_epoch=$S_EPOCH"
echo "  - カウント区間   : [$EPOCH_T, $EPOCH_K]"
echo "==============================================================="

# 例: main.py を呼ぶ場合
python dd_scratch_model_separate_group.py \
  --fix_seed        $SEED \
  --model           $MODEL \
  --model_width     $MODEL_WIDTH \
  --epoch           $EPOCH \
  --dataset         $DATASET \
  --target          $TARGET \
  --label_noise_rate $LABEL_NOISE_RATE \
  --batch_size      $BATCH_SIZE \
  --lr              $LR \
  --optimizer       $OPTIMIZER \
  --momentum        $MOMENTUM \
  --gpu             $GPU \
  --num_workers     $NUM_WORKERS \
  --variance        $VARIANCE \
  --s_epoch         $S_EPOCH \
  --epoch_t         $EPOCH_T \
  --epoch_k         $EPOCH_K

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ 完了しました (exit 0)"
else
  echo "❌ エラーで終了しました (exit $EXIT_CODE)"
fi
