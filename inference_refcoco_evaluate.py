#!/usr/bin/env python
"""
RefCOCO Series Evaluation (JSONL-based)
Supports: refcoco (val/testA/testB), refcoco+ (val/testA/testB), refcocog (val/test)

Uses single conversation[0] question per sample (standard evaluation).
GT masks are RLE-encoded, decoded via pycocotools.

Usage:
    python inference_refcoco_evaluate.py \
        --resume /path/to/PSRS.pth \
        --version /path/to/Qwen3-VL-4B-Instruct \
        --jsonl_dir /path/to/refcoco_jsonls \
        --image_dir /path/to/COCO/train2014 \
        --benchmarks refcoco_val,refcoco_testA,refcoco_testB
"""

import argparse
import os
import sys
import json
import random
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from pycocotools import mask as mask_util

from transformers import AutoConfig, AutoProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.vlmsam import VlmSamSegForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# Helper Functions
# ============================================================================

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def decode_rle_masks(rle_list, height, width):
    """Decode a list of RLE strings into a single binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for rle_str in rle_list:
        if isinstance(rle_str, str):
            rle = {"size": [height, width], "counts": rle_str.encode("utf-8")}
        elif isinstance(rle_str, dict):
            rle = rle_str
            if not isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].encode("utf-8")
        else:
            continue
        m = mask_util.decode(rle)
        mask = np.maximum(mask, m)
    return mask


def calculate_iou(pred_mask, gt_mask):
    """Calculate intersection, union, IoU between two binary masks."""
    pred_bool = pred_mask.bool()
    gt_bool = gt_mask.bool()
    intersection = torch.logical_and(pred_bool, gt_bool).sum()
    union = torch.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        iou = torch.tensor(1.0 if gt_bool.sum() == 0 else 0.0, device=pred_mask.device)
    else:
        iou = intersection.float() / union.float()
    return intersection, union, iou


def load_processed_ids(result_path):
    """Load already-processed sample IDs for resume support."""
    processed = set()
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data["id"])
                except:
                    continue
    return processed


def calculate_final_metrics(result_path, benchmark_name):
    """Calculate and print gIoU and cIoU from result JSONL."""
    if not os.path.exists(result_path):
        return

    ious, intersections, unions = [], [], []
    with open(result_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                ious.append(data["iou"])
                intersections.append(data["intersection"])
                unions.append(data["union"])
            except:
                continue

    if not ious:
        print(f"[{benchmark_name}] No results found.")
        return

    giou = sum(ious) / len(ious)
    ciou = sum(intersections) / (sum(unions) + 1e-10)

    print("=" * 60)
    print(f"  {benchmark_name}: {len(ious)} samples")
    print(f"  gIoU (avg per-sample IoU): {giou:.4f}")
    print(f"  cIoU (total inter/union):  {ciou:.4f}")
    print("=" * 60)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_benchmark(benchmark_name, args, model, processor, tokenizer, device, transform_sam, torch_dtype):
    """Evaluate a single RefCOCO benchmark split."""

    jsonl_path = os.path.join(args.jsonl_dir, f"{benchmark_name}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"JSONL not found: {jsonl_path}, skipping {benchmark_name}")
        return

    # Load data
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                continue
    print(f"\n[{benchmark_name}] Loaded {len(samples)} samples from {jsonl_path}")

    # Result file
    result_dir = os.path.join(args.result_dir, "refcoco_results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{benchmark_name}_results.jsonl")

    processed_ids = load_processed_ids(result_path)
    print(f"[{benchmark_name}] Already processed: {len(processed_ids)}, remaining: {len(samples) - len(processed_ids)}")

    for sample in tqdm(samples, desc=f"Eval {benchmark_name}"):
        sample_id = sample["id"]
        if sample_id in processed_ids:
            continue

        # Get image
        image_name = sample["images"][0]
        image_path = os.path.join(args.image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        try:
            image_pil = Image.open(image_path).convert("RGB")
        except:
            continue
        image_np = np.array(image_pil)
        orig_size = image_np.shape[:2]  # (H, W)

        # SAM image preprocessing
        image_sam = transform_sam.apply_image(image_np)
        resize = image_sam.shape[:2]
        image_sam_tensor = (
            torch.from_numpy(image_sam)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(device)
            .to(torch_dtype)
        )
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device).to(torch_dtype)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device).to(torch_dtype)
        image_sam_tensor = (image_sam_tensor - pixel_mean) / pixel_std
        h, w = image_sam_tensor.shape[-2:]
        padh = args.image_size - h
        padw = args.image_size - w
        image_sam_tensor = F.pad(image_sam_tensor, (0, padw, 0, padh))

        # GT mask (RLE)
        height = sample["height_list"][0]
        width = sample["width_list"][0]
        gt_mask_np = decode_rle_masks(sample["masks"][0], height, width)
        gt_mask_tensor = torch.from_numpy(gt_mask_np).to(device).float()

        # Question: use conversations[0] (standard single-question evaluation)
        question_text = sample["conversations"][0]["value"]
        # Remove image tags (both LISA-style and Qwen3-VL-style)
        question_text = question_text.replace("<image>\n", "").replace("<image>", "")
        question_text = question_text.replace("<|vision_start|>", "").replace("<|image_pad|>", "").replace("<|vision_end|>", "")
        question_text = question_text.strip()

        # --- Generate stage ---
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            inputs = processor(text=[prompt], images=[image_pil], return_tensors="pt").to(device)
        except (IndexError, Exception) as e:
            print(f"[WARN] Skipping {sample_id}: processor error: {e}")
            continue

        with torch.no_grad():
            out_ids = model.vlm.generate(**inputs, max_new_tokens=256, do_sample=False)
            response = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()

        # --- Forward stage (decode mask) ---
        pred_mask_bool = torch.zeros(orig_size, dtype=torch.bool, device=device)

        msgs_fwd = msgs + [{"role": "assistant", "content": response}]
        text_fwd = processor.apply_chat_template(msgs_fwd, tokenize=False, add_generation_prompt=False)
        try:
            inputs_fwd = processor(text=[text_fwd], images=[image_pil], return_tensors="pt")
        except (IndexError, Exception) as e:
            print(f"[WARN] Skipping {sample_id} forward: processor error: {e}")
            inter, union, iou = calculate_iou(pred_mask_bool, gt_mask_tensor)
            record = {"id":sample_id,"image":image_name,"question":question_text,"response":response,
                      "intersection":inter.item(),"union":union.item(),"iou":iou.item()}
            with open(result_path, "a") as f: f.write(json.dumps(record)+"\n")
            continue

        if "SEG" in response or "seg" in response:
            with torch.no_grad():
                labels = inputs_fwd["input_ids"].clone()
                labels[:] = -100
                vlm_inputs = {"pixel_values": inputs_fwd["pixel_values"].to(device)}
                if "image_grid_thw" in inputs_fwd:
                    vlm_inputs["image_grid_thw"] = inputs_fwd["image_grid_thw"].to(device)

                out_dict = model(
                    images=image_sam_tensor,
                    input_ids=inputs_fwd["input_ids"].to(device),
                    labels=labels.to(device),
                    attention_masks=inputs_fwd["attention_mask"].to(device),
                    vlm_inputs=vlm_inputs,
                    offset=torch.LongTensor([0, 1]).to(device),
                    masks_list=[gt_mask_tensor],
                    label_list=[gt_mask_tensor],
                    resize_list=[resize],
                    inference=True,
                    conversation_list=[text_fwd],
                )

                if out_dict.get("pred_masks") is not None and len(out_dict["pred_masks"]) > 0:
                    raw = out_dict["pred_masks"][0][-1]
                    mask_np = (raw > 0).cpu().numpy().astype(np.uint8)
                    if mask_np.shape != orig_size:
                        mask_np = cv2.resize(mask_np, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
                    pred_mask_bool = torch.from_numpy(mask_np).to(device).bool()

        # Calculate IoU
        inter, union, iou = calculate_iou(pred_mask_bool, gt_mask_tensor)

        record = {
            "id": sample_id,
            "image": image_name,
            "question": question_text,
            "response": response,
            "intersection": inter.item(),
            "union": union.item(),
            "iou": iou.item(),
        }
        with open(result_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Print metrics
    calculate_final_metrics(result_path, benchmark_name)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RefCOCO Series Evaluation (JSONL-based)")
    parser.add_argument("--resume", type=str, required=True, help="Path to PSRS checkpoint")
    parser.add_argument("--version", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Pretrained Qwen3-VL HuggingFace id or local path")
    parser.add_argument("--jsonl_dir", type=str, default="./dataset/refcoco_eval",
                        help="Directory containing RefCOCO JSONL files (refcoco_val.jsonl etc.)")
    parser.add_argument("--image_dir", type=str, default="./dataset/train2014",
                        help="Directory containing COCO train2014 images")
    parser.add_argument("--result_dir", type=str, default="./evaluate_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--benchmarks", type=str,
                        default="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test",
                        help="Comma-separated benchmark splits to evaluate")
    parser.add_argument("--image_size", type=int, default=1024, help="SAM image size")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--use_SEG_token", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    print("=" * 60)
    print(f"  RefCOCO Evaluation")
    print(f"  Checkpoint: {args.resume}")
    print(f"  Benchmarks: {args.benchmarks}")
    print(f"  JSONL dir:  {args.jsonl_dir}")
    print(f"  Image dir:  {args.image_dir}")
    print("=" * 60)

    device = torch.device("cuda")
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    # Load processor & tokenizer
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    tokenizer.add_tokens(["<SEG>", "<neg_SEG>"])
    seg_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    neg_seg_idx = tokenizer("<neg_SEG>", add_special_tokens=False).input_ids[0]

    config = AutoConfig.from_pretrained(args.version)

    # Build model
    model = VlmSamSegForCausalLM(
        config,
        seg_token_idx=seg_idx,
        neg_seg_token_idx=neg_seg_idx,
        use_SEG_token=args.use_SEG_token,
        torch_dtype=torch_dtype,
        model=args.version,
        attention="flash_attention_2",
        train_mask_decoder=True,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
    ).to(device)

    model.vlm.resize_token_embeddings(len(tokenizer))

    # Load checkpoint
    print(f"Loading checkpoint: {args.resume}")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned)
    model.eval()
    print("Model loaded successfully.")

    transform = ResizeLongestSide(args.image_size)

    # Run benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    for bench_name in benchmarks:
        evaluate_benchmark(bench_name, args, model, processor, tokenizer, device, transform, torch_dtype)
        torch.cuda.empty_cache()

    print("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
