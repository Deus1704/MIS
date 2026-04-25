#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/teamspace/studios/this_studio/MIS_Project/Path_A_LPD"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Rough target for beating FBP on this synthetic circular-phantom demo:
# - 10k+ training samples for a serious baseline
# - 20k if you want the gain to be more stable
# This script uses the safer target while still staying CPU-friendly.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
NUM_ANGLES="${NUM_ANGLES:-45}"
NUM_DETECTORS="${NUM_DETECTORS:-64}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEED="${SEED:-0}"

cd "$ROOT_DIR"
mkdir -p checkpoints

exec "$PYTHON_BIN" minimal_lpd_cpu.py \
  --image-size "$IMAGE_SIZE" \
  --num-angles "$NUM_ANGLES" \
  --num-detectors "$NUM_DETECTORS" \
  --num-iterations "$NUM_ITERATIONS" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --train-samples "$TRAIN_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --seed "$SEED"
