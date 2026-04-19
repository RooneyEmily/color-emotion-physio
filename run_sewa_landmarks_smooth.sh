#!/usr/bin/env bash
# One-shot SEWA eval: landmark crops + temporal smoothing. Edit the variables below if paths differ.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMONET_ROOT="${EMONET_TRAIN_REPO:-$REPO}"
PP="$EMONET_ROOT"
if [[ -f "$REPO/color-emotion-physio/src/pipeline/process_videos_av.py" ]]; then
  PP="$REPO/color-emotion-physio/src/pipeline:$EMONET_ROOT"
fi
export PYTHONPATH="$PP${PYTHONPATH:+:$PYTHONPATH}"

SEWA_ROOT="${SEWA_ROOT:-/media/mabl-main/Data-Karthik/HRI-AVC/SEWA}"
CHECKPOINT="${CHECKPOINT:-/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_final/lab/run_1/best_model.pth}"
OUTPUT="${OUTPUT:-/home/mabl-main/results/sewa_landmarks_smooth.json}"

cd "$REPO"

python3 evaluate_sewa_ood.py \
  --sewa-root "$SEWA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --color_space lab \
  --crop-source landmarks \
  --landmark-padding 0.3 \
  --smooth-k 5 \
  --label-scale auto \
  -o "$OUTPUT"
