#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${THIS_DIR}/.." && pwd)"
PATH_B_DL_DIR="${PROJECT_DIR}/Path_B_FreqHybridNet/ct_recon_dl"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR_REL="results/${RUN_NAME}"
RUN_DIR_ABS="${THIS_DIR}/${RUN_DIR_REL}"

DATA_PATH="${DATA_PATH:-../real_data/organamnist/raw/organamnist.npz}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
N_ANGLES="${N_ANGLES:-90}"
DOSE_FRACTION="${DOSE_FRACTION:-0.25}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-8}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-2000}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_TEST="${MAX_TEST:-}"
SEED="${SEED:-42}"
BASE_CH="${BASE_CH:-32}"
METHODS="${METHODS:-red_cnn unet attention_unet}"

mkdir -p "${RUN_DIR_ABS}"
read -r -a METHODS_ARR <<< "${METHODS}"

echo "[Path C] Running FBP + DL enhancement pipeline"
echo "[Path C] Run directory: ${RUN_DIR_ABS}"
echo "[Path C] Methods: ${METHODS_ARR[*]}"
echo "[Path C] Dose fraction: ${DOSE_FRACTION}"

cd "${PATH_B_DL_DIR}"

CMD=(
  "${PYTHON_BIN}" "scripts/run_dl_pipeline.py"
  --data "${DATA_PATH}"
  --image-size "${IMAGE_SIZE}"
  --n-angles "${N_ANGLES}"
  --dose-fraction "${DOSE_FRACTION}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --patience "${PATIENCE}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --n-permutations "${N_PERMUTATIONS}"
  --out-dir "../../Path_C_FBP_DL/${RUN_DIR_REL}"
  --seed "${SEED}"
  --base-ch "${BASE_CH}"
  --methods "${METHODS_ARR[@]}"
)

if [[ -n "${MAX_TRAIN}" ]]; then
  CMD+=(--max-train "${MAX_TRAIN}")
fi

if [[ -n "${MAX_TEST}" ]]; then
  CMD+=(--max-test "${MAX_TEST}")
fi

"${CMD[@]}"

cd "${THIS_DIR}"
"${PYTHON_BIN}" "scripts/collect_path_c_results.py" --run-dir "${RUN_DIR_ABS}"

echo "[Path C] Complete."
echo "[Path C] Metrics and visuals organized at: ${RUN_DIR_ABS}"
