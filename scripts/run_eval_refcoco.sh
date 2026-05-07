#!/bin/bash
# Evaluate PSRS on RefCOCO/+/g (8 splits).
#
# Usage:
#   bash scripts/run_eval_refcoco.sh [CKPT_PATH] [BENCHMARKS]
#
# Example:
#   bash scripts/run_eval_refcoco.sh ./checkpoints/PSRS.pth refcoco_val,refcoco_testA
set -e

CKPT=${1:-"./checkpoints/PSRS.pth"}
BENCHMARKS=${2:-"refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"}

python inference_refcoco_evaluate.py \
    --resume "$CKPT" \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --jsonl_dir "./dataset/refcoco_eval" \
    --image_dir "./dataset/train2014" \
    --result_dir "./evaluate_results/RefCOCO_results" \
    --benchmarks "$BENCHMARKS" \
    --use_SEG_token True
