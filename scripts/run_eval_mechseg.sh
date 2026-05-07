#!/bin/bash
# Evaluate PSRS on MechSeg-Bench.
#
# Usage:
#   bash scripts/run_eval_mechseg.sh [CKPT_PATH]
#
# Expects:
#   ./dataset/MechSeg-Bench/new_final_test.json
#   ./dataset/COCO/train2017/  (image dir)
set -e

CKPT=${1:-"./checkpoints/PSRS.pth"}

python inference_MechSeg_Bench_evaluate.py \
    --resume "$CKPT" \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --json_path "./dataset/MechSeg-Bench/new_final_test.json" \
    --image_dir "./dataset/COCO/train2017" \
    --result_save_path "./evaluate_results/MechSeg_results" \
    --use_SEG_token True
