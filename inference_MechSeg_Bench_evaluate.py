#!/usr/bin/env python
import argparse
import os
import sys
from typing import List
import json
from collections import OrderedDict
import random
import textwrap 
import re
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig
from PIL import Image
from tqdm import tqdm   

from model.segment_anything.utils.transforms import ResizeLongestSide
from model.vlmsam import VlmSamSegForCausalLM

# --- 辅助函数保持不变 ---
def update_stats(stats_dict, intersection, union, iou):
    stats_dict['intersection'] += intersection.item()
    stats_dict['union'] += union.item()
    stats_dict['iou_sum'] += iou.item()
    stats_dict['count'] += 1

def create_mask_from_polygons(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons or not any(polygons):
        return mask.astype(np.float32)
    pts_list = []
    for poly in polygons:
        if not poly: continue
        try:
            pts = np.array(poly, dtype=np.int32)
            if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2: pass
            elif pts.ndim == 2 and pts.shape[1] == 2: pts = pts.reshape((-1, 1, 2))
            elif pts.ndim == 1: pts = pts.reshape((-1, 1, 2))
            else: continue
            pts_list.append(pts)
        except: continue
    if not pts_list: return mask.astype(np.float32)
    cv2.fillPoly(mask, pts=pts_list, color=1)
    return mask.astype(np.float32)

def calculate_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    # 逻辑与 = Intersection
    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    # 逻辑或 = Union
    union = torch.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        iou = torch.tensor(1.0, device=pred_mask.device) 
    else:
        iou = intersection / union
    return intersection, union, iou

def preprocess_sam_image(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    return F.pad(x, (0, padw, 0, padh))


def parse_pos_neg_points(text):
    pos_points = []
    neg_points = []
    pt_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]'
    
    mask_start = text.find("The mask is")
    inter_start = text.find("The interference is")
    
    if mask_start != -1 and inter_start != -1:
        if mask_start < inter_start:
            mask_text = text[mask_start:inter_start]
            inter_text = text[inter_start:]
        else:
            inter_text = text[inter_start:mask_start]
            mask_text = text[mask_start:]
        
        pos_matches = re.findall(pt_pattern, mask_text)
        pos_points = [(float(x), float(y)) for x, y in pos_matches]
        neg_matches = re.findall(pt_pattern, inter_text)
        neg_points = [(float(x), float(y)) for x, y in neg_matches]
        return pos_points, neg_points
    
    elif mask_start != -1:
        mask_text = text[mask_start:]
        pos_matches = re.findall(pt_pattern, mask_text)
        pos_points = [(float(x), float(y)) for x, y in pos_matches]
        return pos_points, []
    
    elif inter_start != -1:
        inter_text = text[inter_start:]
        neg_matches = re.findall(pt_pattern, inter_text)
        neg_points = [(float(x), float(y)) for x, y in neg_matches]
        return [], neg_points
    
    else:
        all_matches = re.findall(pt_pattern, text)
        pos_points = [(float(x), float(y)) for x, y in all_matches]
        return pos_points, []

def parse_args(args):
    parser = argparse.ArgumentParser(description="VlmSamSeg Batch Inference (Strict)")
    parser.add_argument("--json_path", default="./dataset/MechSeg-Bench/new_final_test.json", type=str,
                        help="Path to MechSeg-Bench evaluation JSON.")
    parser.add_argument("--image_dir", default="./dataset/COCO/train2017", type=str,
                        help="COCO train2017 image directory.")
    parser.add_argument("--version", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Pretrained Qwen3-VL HuggingFace id or local path")

    parser.add_argument("--vis_save_path", default=None, type=str,
                        help="Vis output dir. Auto-named under result_save_path if None.")
    parser.add_argument("--result_save_path", default="./evaluate_results/MechSeg_results", type=str,
                        help="JSONL result dir.")

    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--resume", default="./checkpoints/PSRS.pth", type=str,
                        help="Path to merged checkpoint (LoRA already merged into base weights).")

    parser.add_argument("--use_SEG_token", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--save_vis", default=True, help="If set, save visualization images.")

    return parser.parse_args(args)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    setup_seed(42)  # <--- 在这里调用，固定种子为 42 或其他任意整数
    args = parse_args(args)

    # --- 1. 路径自动处理 ---
    if args.resume:
        checkpoint_name = os.path.basename(args.resume)
        if checkpoint_name.endswith('.pth'):
            checkpoint_name = checkpoint_name[:-4]
    else:
        checkpoint_name = "default_model"

    if args.vis_save_path is None:
        args.vis_save_path = f"./vis_output_{checkpoint_name}_MechSeg"
    
    os.makedirs(args.result_save_path, exist_ok=True)
    result_filename = f"{checkpoint_name}_results_MechSeg.jsonl"
    result_jsonl_path = os.path.join(args.result_save_path, result_filename)

    print(f"🔹 Checkpoint: {checkpoint_name}")
    print(f"🔹 Result Path: {result_jsonl_path}")
    print(f"🔹 Vis Path:    {args.vis_save_path} (Enabled: {args.save_vis})")

    vis_subdirs = {}
    if args.save_vis:
        os.makedirs(args.vis_save_path, exist_ok=True)
        vis_subdirs = {
            "original": os.path.join(args.vis_save_path, "original"),
            "gt_mask": os.path.join(args.vis_save_path, "gt_mask"),
            "pred_mask": os.path.join(args.vis_save_path, "pred_mask"),
            "overlay": os.path.join(args.vis_save_path, "overlay"),   
            "overlay_with_points": os.path.join(args.vis_save_path, "overlay_with_points") 
        }
        for p in vis_subdirs.values():
            os.makedirs(p, exist_ok=True)

    # --- 2. 统计变量初始化 ---
    stats_valid = {'intersection': 0.0, 'union': 0.0, 'iou_sum': 0.0, 'count': 0}
    stats_total = {'intersection': 0.0, 'union': 0.0, 'iou_sum': 0.0, 'count': 0}

    processed_ids = set()

    # --- 3. 断点续跑恢复 ---
    if os.path.exists(result_jsonl_path):
        print(f"Checking existing results in {result_jsonl_path}...")
        with open(result_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_ids.add(record['id'])
                    
                    iou_val = record['iou']
                    int_val = record['intersection']
                    uni_val = record['union']
                    # 读取是否是有效格式
                    valid_fmt = record.get('valid_format', False) 

                    # 恢复 Total 统计
                    stats_total['intersection'] += int_val
                    stats_total['union'] += uni_val
                    stats_total['iou_sum'] += iou_val
                    stats_total['count'] += 1

                    # 恢复 Valid 统计
                    if valid_fmt:
                        stats_valid['intersection'] += int_val
                        stats_valid['union'] += uni_val
                        stats_valid['iou_sum'] += iou_val
                        stats_valid['count'] += 1
                        
                except json.JSONDecodeError:
                    continue
        print(f"✅ Resumed! Skipped {len(processed_ids)} items.")
    else:
        print("Starting new inference.")

    # --- 4. 模型加载 ---
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = transformers.AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    if "<SEG>" not in tokenizer.get_vocab(): tokenizer.add_tokens("<SEG>")
    if "<neg_SEG>" not in tokenizer.get_vocab(): tokenizer.add_tokens("<neg_SEG>")
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    args.neg_seg_token_idx = tokenizer("<neg_SEG>", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32
    config = AutoConfig.from_pretrained(args.version)
    kwargs = {
        "torch_dtype": torch_dtype, 
        "model": args.version, 
        "attention": "flash_attention_2", 
        "train_mask_decoder": True, 
        "out_dim": 256, 
        "ce_loss_weight": 1.0, 
        "dice_loss_weight": 0.5, 
        "bce_loss_weight": 2.0
    }
    
    model = VlmSamSegForCausalLM(
        config, 
        seg_token_idx=args.seg_token_idx, 
        neg_seg_token_idx=args.neg_seg_token_idx, 
        use_SEG_token=args.use_SEG_token,
        **kwargs
    ).to(device)
    
    model.vlm.resize_token_embeddings(len(tokenizer))
    
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint file not found at {args.resume}")
        
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        cleaned_state_dict[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    print("Model loaded.")

    transform_sam = ResizeLongestSide(args.image_size)

    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    f_result = open(result_jsonl_path, 'a', encoding='utf-8')

    # --- 5. 定义生成函数 ---
    def execute_generate(curr_question, curr_image):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": curr_question}]}]
        gen_prompt_text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        vlm_gen_inputs = processor(text=[gen_prompt_text], images=[curr_image], return_tensors="pt").to(device)
        with torch.no_grad():
            prompt_len = vlm_gen_inputs['input_ids'].shape[1]
            output_ids = model.vlm.generate(**vlm_gen_inputs, max_new_tokens=512, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=False).strip()
        return ans

    try:
        pbar = tqdm(data, desc="Inference", total=len(data))
        
        for item in pbar:
            if item['id'] in processed_ids:
                continue

            raw_prompt = item['conversations'][0]['value']
            question = raw_prompt.split('\n', 1)[1].strip()
            image_filename = item['image']
            image_path = os.path.join(args.image_dir, os.path.basename(image_filename))

            if not os.path.exists(image_path):
                tqdm.write(f"⚠️ Image not found: {image_path}")
                continue
            
            try:
                image_pil = Image.open(image_path).convert("RGB")
            except Exception as e:
                tqdm.write(f"⚠️ Error opening image {image_path}: {e}")
                continue
                
            image_np = np.array(image_pil)
            original_size = image_np.shape[:2]
            
            # --- Stage 1: Generate ---
            generated_answer = execute_generate(question, image_pil)
            
            # 尝试补充提示词
            if "<SEG>" not in generated_answer:
                target_suffixes = ["Please respond with segmentation mask.", "Please output segmentation mask."]
                if not any(s in question for s in target_suffixes):
                    # 【修改回旧逻辑】直接覆盖 question 变量
                    suffix = random.choice(target_suffixes)
                    question = f"{question} {suffix}" 
                    generated_answer = execute_generate(question, image_pil)

            pos_points, neg_points = parse_pos_neg_points(generated_answer)

            # --- 准备 GT Mask (所有情况都需要) ---
            polygons = item.get('segmentation', {}).get('polygons')
            if polygons:
                gt_mask_np = create_mask_from_polygons(polygons, original_size[0], original_size[1])
                gt_mask_tensor = torch.from_numpy(gt_mask_np).to(device, dtype=torch.float32)
            else:
                gt_mask_np = np.zeros(original_size, dtype=np.uint8) 
                gt_mask_tensor = torch.zeros(original_size, dtype=torch.float32, device=device)

            # --- Stage 2: Mask Generation Logic ---
            valid_format = False
            
            # 检查是否成功输出 <SEG>
            if "<SEG>" in generated_answer:
                valid_format = True
                
                # --- A. 成功分支：运行模型 Forward ---
                with torch.no_grad():
                    messages_for_forward = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}, {"role": "assistant", "content": generated_answer}]
                    text_forward = processor.apply_chat_template(messages_for_forward, tokenize=False, add_generation_prompt=False)
                    tokenized_output = processor(text=[text_forward], images=[image_pil], return_tensors="pt")
                    
                    image_sam = transform_sam.apply_image(image_np)
                    resize = image_sam.shape[:2]
                    
                    image_sam_tensor = torch.from_numpy(image_sam).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
                    if args.precision == "bf16": image_sam_tensor = image_sam_tensor.to(torch.bfloat16)
                    elif args.precision == "fp16": image_sam_tensor = image_sam_tensor.to(torch.float16)
                    
                    image_sam_tensor = preprocess_sam_image(
                        image_sam_tensor, 
                        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device).to(image_sam_tensor.dtype),
                        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device).to(image_sam_tensor.dtype),
                        img_size=args.image_size
                    )
                    
                    labels = tokenized_output['input_ids'].clone(); labels[:] = -100
                    vlm_inputs = {"pixel_values": tokenized_output['pixel_values'].to(device)}
                    if 'image_grid_thw' in tokenized_output: vlm_inputs["image_grid_thw"] = tokenized_output['image_grid_thw'].to(device)

                    input_dict = {
                        "images": image_sam_tensor, "input_ids": tokenized_output['input_ids'].to(device),
                        "labels": labels.to(device), "attention_masks": tokenized_output['attention_mask'].to(device),
                        "vlm_inputs": vlm_inputs, "offset": torch.LongTensor([0, 1]).to(device),
                        "masks_list": [gt_mask_tensor], "label_list": [gt_mask_tensor],
                        "resize_list": [resize], "change_list": [], "inference": True, "conversation_list": [text_forward],
                    }
                    output_dict = model(**input_dict)
                    pred_masks = output_dict.get("pred_masks")

                if not pred_masks or pred_masks[0].shape[0] == 0:
                    # 异常：有Token但没Mask，生成全黑Mask
                    pred_mask_binary = torch.zeros_like(gt_mask_tensor, dtype=torch.bool)
                    pred_mask_np = np.zeros(original_size, dtype=bool)
                else:
                    pred_mask_tensor = pred_masks[0][-1]
                    pred_mask_binary = (pred_mask_tensor > 0)
                    pred_mask_np = pred_mask_binary.detach().cpu().numpy()

                # 计算 Metrics (正常)
                int_std, uni_std, iou_std = calculate_iou(pred_mask_binary, gt_mask_tensor)

            else:
                # --- B. 失败分支：跳过 Forward，直接全零 ---
                valid_format = False
                
                # 1. 创建全零的 Pred Mask
                pred_mask_binary = torch.zeros_like(gt_mask_tensor, dtype=torch.bool)
                pred_mask_np = np.zeros(original_size, dtype=bool)
                
                # 2. [关键] 调用完全一样的 calculate_iou 计算
                # 这样 Intersection=0, Union=GT面积 (因为 pred是空, union就是gt)
                # 保证了 cIoU 的分母和原逻辑完全一致
                int_std, uni_std, iou_std = calculate_iou(pred_mask_binary, gt_mask_tensor)

            # --- 更新统计 ---
            if valid_format:
                update_stats(stats_valid, int_std, uni_std, iou_std)
            
            # Total 始终更新 (包含了失败样本的计算结果)
            update_stats(stats_total, int_std, uni_std, iou_std)

            pbar.set_postfix({"iou": f"{iou_std.item():.4f}", "fmt": "OK" if valid_format else "FAIL"})

            result_record = {
                "id": item['id'], "image": item['image'],
                "question": question, "generated_answer": generated_answer,
                "valid_format": valid_format,
                "iou": iou_std.item(), "intersection": int_std.item(), "union": uni_std.item()
            }
            f_result.write(json.dumps(result_record) + "\n")
            f_result.flush()

            # --- Visualization ---
            if args.save_vis:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_prefix = f"{item['id']}_{base_name}"
                
                # 1. Prediction Mask
                pred_mask_gray = (pred_mask_np.astype(np.uint8) * 255)
                img_pred_mask_bgr = cv2.cvtColor(pred_mask_gray, cv2.COLOR_GRAY2BGR)

                # 2. Original
                img_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                h, w, _ = img_original_bgr.shape 

                # 3. GT Mask Overlay
                gt_overlay_img = image_np.copy()
                gt_mask_bool = gt_mask_np.astype(bool)
                overlay_color = np.array([255, 0, 0]) # RGB Red
                alpha = 0.5
                gt_overlay_img[gt_mask_bool] = (gt_overlay_img[gt_mask_bool] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)
                img_gt_overlay_bgr = cv2.cvtColor(gt_overlay_img, cv2.COLOR_RGB2BGR)

                # 4. Pred Overlay (Mask Only)
                overlay_img_mask_only = image_np.copy()
                pred_mask_bool = pred_mask_np.astype(bool)
                overlay_color = np.array([255, 0, 0]) 
                
                if np.any(pred_mask_bool):
                    overlay_img_mask_only[pred_mask_bool] = (overlay_img_mask_only[pred_mask_bool] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)
                
                # 5. Pred Overlay (Mask + Points)
                overlay_img_with_points = overlay_img_mask_only.copy()
                
                for px, py in pos_points:
                    real_x = int((px / 1000.0) * w)
                    real_y = int((py / 1000.0) * h)
                    if 0 <= real_x < w and 0 <= real_y < h:
                        cv2.circle(overlay_img_with_points, (real_x, real_y), max(5, w // 200), (0, 255, 255), -1) 
                for nx, ny in neg_points:
                    real_x = int((nx / 1000.0) * w)
                    real_y = int((ny / 1000.0) * h)
                    if 0 <= real_x < w and 0 <= real_y < h:
                        cv2.circle(overlay_img_with_points, (real_x, real_y), max(5, w // 200), (255, 0, 255), -1)

                img_overlay_mask_bgr = cv2.cvtColor(overlay_img_mask_only, cv2.COLOR_RGB2BGR)
                img_overlay_points_bgr = cv2.cvtColor(overlay_img_with_points, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(vis_subdirs["original"], f"{save_prefix}_orig.png"), img_original_bgr)
                cv2.imwrite(os.path.join(vis_subdirs["gt_mask"], f"{save_prefix}_gt.png"), img_gt_overlay_bgr)
                cv2.imwrite(os.path.join(vis_subdirs["pred_mask"], f"{save_prefix}_pred.png"), img_pred_mask_bgr)
                cv2.imwrite(os.path.join(vis_subdirs["overlay"], f"{save_prefix}_overlay.png"), img_overlay_mask_bgr)
                cv2.imwrite(os.path.join(vis_subdirs["overlay_with_points"], f"{save_prefix}_overlay_pts.png"), img_overlay_points_bgr)

    except Exception as e:
        tqdm.write(f"\n❌ Error encountered: {e}")
        traceback.print_exc()
    finally:
        f_result.close()
        pbar.close()

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY (STRICT MODE)")
    print("="*60)
    def print_stat(name, stats):
        if stats['count'] > 0:
            print(f"\n{name} (Items: {stats['count']})")
            print(f"  Average IoU (GIoU): {stats['iou_sum'] / stats['count']:.4f}")
            print(f"  Overall IoU (cIoU): {stats['intersection'] / (stats['union'] + 1e-10):.4f}")
        else:
            print(f"\n{name}: No items processed.")

    print_stat("[Metric 1] Valid Format Only (Only Success)", stats_valid)
    print_stat("[Metric 2] End-to-End Total (Failures Included)", stats_total)
    print("="*60 + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])