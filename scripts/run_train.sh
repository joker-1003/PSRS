#!/bin/bash
# PSRS multi-GPU training entry.
#
# Usage:
#   bash scripts/run_train.sh [NUM_GPUS] [RESUME_CKPT]
#
# Examples:
#   bash scripts/run_train.sh                     # single GPU, no resume
#   bash scripts/run_train.sh 8                   # 8 GPUs
#   bash scripts/run_train.sh 8 ./runs/last.pth   # 8 GPUs, resume
#
# Before running: prepare ./dataset (see README) and download SAM weights to ./weights/.
set -e

NUM_GPUS=${1:-1}
RESUME_CKPT=${2:-""}

VLM="Qwen/Qwen3-VL-4B-Instruct"
SAM_WEIGHTS="./weights/sam_vit_h_4b8939.pth"
DATASET_DIR="./dataset"
LOG_DIR="./runs/psrs_main_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TRAIN_ARGS=(
    --version "$VLM"
    --vision_pretrained "$SAM_WEIGHTS"
    --dataset_dir "$DATASET_DIR"
    --dataset "sem_seg||refer_seg||ReasonSeg||overlap_reasonseg"
    --sample_rates "9,9,1,3"
    --sem_seg_data "ade20k||cocostuff"
    --refer_seg_data "refcoco||refcoco+||refcocog"
    --epochs 30
    --steps_per_epoch 5000
    --batch_size 2
    --grad_accumulation_steps 5
    --lr 4e-5
    --lora_r 16
    --image_size 1024
    --use_SEG_token True
    --num_points 1
    --log_base_dir "$LOG_DIR"
)

# overlap_reasonseg requires a JSON describing the MechSeg-Bench training split.
# Set OVERLAP_JSON before invoking this script if you train on overlap_reasonseg.
if [ -n "${OVERLAP_JSON:-}" ]; then
    TRAIN_ARGS+=(--overlap_json_path "$OVERLAP_JSON")
fi

if [ -n "$RESUME_CKPT" ]; then
    TRAIN_ARGS+=(--resume "$RESUME_CKPT")
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    export MASTER_ADDR=localhost MASTER_PORT=29500 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
    python train_ddp.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
else
    torchrun --nproc_per_node="$NUM_GPUS" \
        --master_addr=localhost --master_port=29500 \
        train_ddp.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
fi
