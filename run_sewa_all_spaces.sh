#!/usr/bin/env bash
# Run SEWA OOD eval for RGB/YCbCr/XYZ/LAB with the same protocol.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# process_videos_av + emonet: flat repo (research tree) vs nested color-emotion-physio clone
EMONET_ROOT="${EMONET_TRAIN_REPO:-$REPO}"
PP="$EMONET_ROOT"
if [[ -f "$REPO/color-emotion-physio/src/pipeline/process_videos_av.py" ]]; then
  PP="$REPO/color-emotion-physio/src/pipeline:$EMONET_ROOT"
fi
export PYTHONPATH="$PP${PYTHONPATH:+:$PYTHONPATH}"

SEWA_ROOT="${SEWA_ROOT:-/media/mabl-main/Data-Karthik/HRI-AVC/SEWA}"
RUN_ID="${RUN_ID:-1}"
CKPT_BASE="${CKPT_BASE:-/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_final}"
OUT_DIR="${OUT_DIR:-/home/mabl-main/results}"

mkdir -p "$OUT_DIR"

for SPACE in rgb ycbcr xyz lab; do
  CKPT="$CKPT_BASE/$SPACE/run_${RUN_ID}/best_model.pth"
  OUT_JSON="$OUT_DIR/sewa_ood_${SPACE}_landmarks_smooth_run${RUN_ID}.json"
  if [[ ! -f "$CKPT" ]]; then
    echo "Missing checkpoint: $CKPT" >&2
    continue
  fi
  echo "=== Running $SPACE ==="
  python3 evaluate_sewa_ood.py \
    --sewa-root "$SEWA_ROOT" \
    --checkpoint "$CKPT" \
    --color_space "$SPACE" \
    --crop-source landmarks \
    --landmark-padding 0.3 \
    --smooth-k 5 \
    --label-scale auto \
    -o "$OUT_JSON"
done

echo "Done. Outputs in $OUT_DIR"
