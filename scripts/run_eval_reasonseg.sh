#!/bin/bash
# Evaluate PSRS on ReasonSeg (val/test).
#
# Usage:
#   bash scripts/run_eval_reasonseg.sh [CKPT_PATH] [SPLIT]
#
# Example:
#   bash scripts/run_eval_reasonseg.sh ./checkpoints/PSRS.pth reasonseg_val
set -e

CKPT=${1:-"./checkpoints/PSRS.pth"}
SPLIT=${2:-"reasonseg_val"}   # reasonseg_val | reasonseg_test

python inference_reasonseg_evaluate.py \
    --resume "$CKPT" \
    --dataset "$SPLIT" \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --reasonseg_root "./dataset/ReasonSeg" \
    --output_dir "./evaluate_results/ReasonSeg_results" \
    --use_SEG_token True \
    --limit 1000
