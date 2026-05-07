#!/usr/bin/env python
import argparse
import os
import sys
from typing import List, Dict, Any
import json
from collections import OrderedDict
import random
from tqdm import tqdm
import textwrap
import cv2
import re
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig
from PIL import Image
import pycocotools.mask as mask_util
from torch.utils.data import Dataset, DataLoader

# --- [环境优化] 必须放在 import 后，其他代码前 ---
# 1. 禁用 Tokenizers 并行，防止 DataLoader fork 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. 开启 TF32 加速 (A100/3090/4090 提速关键)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 请确保这些引用在你本地是可用的
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.vlmsam import VlmSamSegForCausalLM 

random.seed(42)

# ==============================================================================
# Helper Functions (保持原样)
# ==============================================================================

def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    height, width = img.shape[:2]

    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]
        label_value = 255 if "ignore" in label_id.lower() else 1
        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def update_stats(stats_dict, intersection, union, iou):
    stats_dict['intersection'] += intersection
    stats_dict['union'] += union
    stats_dict['iou_sum'] += iou
    stats_dict['count'] += 1

def preprocess_sam_image(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    return F.pad(x, (0, padw, 0, padh))

def parse_pos_neg_points(text):
    pos_points = []
    neg_points = []
    pt_pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
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
    DESIRED_DATASET = "reasonseg_val"
    DESIRED_RESUME_PATH = "./checkpoints/PSRS.pth"
    use_SEG_token = True
    DESIRED_LIMIT = 1000

    hardcoded_cmd_args = [
        '--dataset', DESIRED_DATASET,
        '--resume', DESIRED_RESUME_PATH
    ]
    if DESIRED_LIMIT is not None:
        hardcoded_cmd_args.extend(['--limit', str(DESIRED_LIMIT)])

    parser = argparse.ArgumentParser(description="VlmSamSeg Chat")

    parser.add_argument("--use_SEG_token", default=use_SEG_token,
                        type=lambda x: str(x).lower() == "true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--json_path", default=None, type=str)
    parser.add_argument("--filtered_json_path", default=None, type=str)
    parser.add_argument("--image_dir", default=None, type=str)
    parser.add_argument("--output_dir", default="./evaluate_results/ReasonSeg_results", type=str)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_vis_freq", default=1, type=int)
    parser.add_argument("--vis_mode", default="combined", type=str)
    parser.add_argument("--version", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Pretrained Qwen3-VL HuggingFace id or local path")
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--limit", default=758, type=int)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for VLM generation")
    parser.add_argument("--reasonseg_root", default="./dataset/ReasonSeg", type=str,
                        help="Root dir containing reasonseg_{val,test}_fixed.jsonl and val/test image folders")
    parser.add_argument("--refcoco_json_base", default="./dataset/refcoco_eval", type=str,
                        help="Dir containing refcoco split jsonls (e.g. refcoco_val.jsonl)")
    parser.add_argument("--refcoco_img_base", default="./dataset/train2014", type=str,
                        help="COCO train2014 image dir")

    parsed_args = parser.parse_args(hardcoded_cmd_args + args)

    refcoco_json_base = parsed_args.refcoco_json_base
    refcoco_img_base = parsed_args.refcoco_img_base
    reasonseg_base = parsed_args.reasonseg_root

    dataset_name = parsed_args.dataset
    if dataset_name is None: raise ValueError("--dataset must not be empty.")

    if parsed_args.json_path is None and parsed_args.image_dir is None:
        if "reasonseg" in dataset_name:
            if dataset_name == "reasonseg_val":
                parsed_args.json_path = os.path.join(reasonseg_base, "reasonseg_val_fixed.jsonl")
                parsed_args.filtered_json_path = os.path.join(reasonseg_base, "reasonseg_val_fixed_filtered.jsonl")
                parsed_args.image_dir = os.path.join(reasonseg_base, "val")
            elif dataset_name == "reasonseg_test":
                parsed_args.json_path = os.path.join(reasonseg_base, "reasonseg_test_fixed.jsonl")
                parsed_args.filtered_json_path = os.path.join(reasonseg_base, "reasonseg_test_fixed_filtered.jsonl")
                parsed_args.image_dir = os.path.join(reasonseg_base, "test")
            else:
                raise ValueError(f"未知的 ReasonSeg 数据集: {dataset_name}")
        elif "refcoco" in dataset_name:
            parsed_args.json_path = os.path.join(refcoco_json_base, f"{dataset_name}.jsonl")
            parsed_args.filtered_json_path = None
            parsed_args.image_dir = refcoco_img_base
        else:
            raise ValueError(f"未知的数据集类型: {dataset_name}。")

    resume_filename = os.path.basename(parsed_args.resume)
    resume_stem = os.path.splitext(resume_filename)[0]
    model_name_part = resume_stem.replace("merged_model_statedict_", "")
    parsed_args.result_jsonl_path = os.path.join(parsed_args.output_dir, f"output_{model_name_part}_{dataset_name}_results.jsonl")

    return parsed_args

# ==============================================================================
# Dataset Definition
# ==============================================================================

class EvalDataset(Dataset):
    def __init__(self, data_list, image_dir, image_size=1024):
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform_sam = ResizeLongestSide(image_size)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Parse Question
        question = ""
        for conv in item['conversations']:
            if conv['from'] == 'human':
                raw_prompt = conv['value']
                question = raw_prompt.split('\n', 1)[-1].strip()
                break
        if not question: return None

        # Image Load
        image_filename = item['images'][0]
        image_path = os.path.join(self.image_dir, image_filename)
        if not os.path.exists(image_path): return None
            
        try:
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            original_size = image_np.shape[:2] # (H, W)
        except Exception as e:
            return None

        # Mask Load
        if "ReasonSeg/test" in self.image_dir:
            reasonseg_json_path = os.path.join(self.image_dir, os.path.splitext(image_filename)[0] + ".json")
        else:
            reasonseg_json_path = os.path.join(self.image_dir, os.path.splitext(image_filename)[0] + ".json")

        if os.path.exists(reasonseg_json_path):
            gt_mask_combined_np = get_mask_from_json(reasonseg_json_path, image_np)
            if gt_mask_combined_np.shape != original_size:
                gt_mask_combined_np = cv2.resize(gt_mask_combined_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask_combined_np = np.zeros(original_size, dtype=np.uint8)

        # SAM Resize Precomputation
        image_sam = self.transform_sam.apply_image(image_np)
        
        return {
            "id": item['id'],
            "image_filename": image_filename,
            "question": question,
            "image_np": image_np,           # Raw image for vis
            "image_sam": image_sam,         # Resized image for SAM
            "gt_mask_np": gt_mask_combined_np,
            "original_size": original_size, 
            "image_path": image_path
        }

def collate_fn(batch):
    # 返回 List[Dict] 而不是 Stacked Tensor，以便灵活处理不同尺寸的图像
    batch = [b for b in batch if b is not None]
    return batch

# ==============================================================================
# Main Loop
# ==============================================================================

def main(args):
    args = parse_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
        
    result_jsonl_path = args.result_jsonl_path
    print(f"Output path: {result_jsonl_path}")

    # --- Load Filtered IDs ---
    filtered_ids = set()
    if args.filtered_json_path and os.path.exists(args.filtered_json_path):
        with open(args.filtered_json_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    filtered_ids.add(int(item['id']))
                except: pass
        print(f"✅ Loaded {len(filtered_ids)} filtered IDs.")

    # --- Resume Logic ---
    stats_orig_valid = {'intersection': 0.0, 'union': 0.0, 'iou_sum': 0.0, 'count': 0} 
    stats_orig_total = {'intersection': 0.0, 'union': 0.0, 'iou_sum': 0.0, 'count': 0} 
    stats_filt_valid = {'intersection': 0.0, 'union': 0.0, 'iou_sum': 0.0, 'count': 0} 
    stats_filt_total = {'intersection': 0.0, 'union': 0.0, 'iou_sum': 0.0, 'count': 0} 

    processed_ids = set()
    if os.path.exists(result_jsonl_path):
        print(f"🔄 Found existing results at {result_jsonl_path}. Loading for resume...")
        with open(result_jsonl_path, 'r') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    res_id = int(res['id'])
                    processed_ids.add(res_id)
                    inter, union, iou = float(res['intersection']), float(res['union']), float(res['iou'])
                    is_valid = res.get('is_valid', True)
                    update_stats(stats_orig_total, inter, union, iou)
                    if is_valid: update_stats(stats_orig_valid, inter, union, iou)
                    if res_id in filtered_ids:
                        update_stats(stats_filt_total, inter, union, iou)
                        if is_valid: update_stats(stats_filt_valid, inter, union, iou)
                except ValueError: continue
        print(f"✅ Resumed stats from {len(processed_ids)} items.")
    
    # --- Model Init ---
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = transformers.AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    tokenizer.add_tokens("<SEG>")
    tokenizer.add_tokens("<neg_SEG>")
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    args.neg_seg_token_idx = tokenizer("<neg_SEG>", add_special_tokens=False).input_ids[0]

    # 【重要】Batch Inference 必须设置 Padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32

    config = AutoConfig.from_pretrained(args.version)
    kwargs = {
        "torch_dtype": torch_dtype, "model": args.version, "attention": "flash_attention_2",
        "train_mask_decoder": True, "out_dim": 256,
        "ce_loss_weight": 1.0, "dice_loss_weight": 0.5, "bce_loss_weight": 2.0,
    }
    model = VlmSamSegForCausalLM(config, seg_token_idx=args.seg_token_idx, neg_seg_token_idx=args.neg_seg_token_idx, use_SEG_token=args.use_SEG_token, **kwargs).to(device)
    model.vlm.resize_token_embeddings(len(tokenizer))
    
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        cleaned_state_dict[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # --- Data Loading ---
    print(f"Loading ORIGINAL data from {args.json_path}...")
    data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: pass
    
    data_to_process = [d for d in data if int(d['id']) not in processed_ids]
    print(f"--- After resume filtering: {len(data_to_process)} items remaining (Total: {len(data)}). ---")

    if args.limit is not None and args.limit > 0:
        remaining_limit = args.limit - len(processed_ids)
        if remaining_limit > 0 and remaining_limit < len(data_to_process):
             print(f"--- Sampling {remaining_limit} items from remaining data based on limit. ---")
             data_to_process = random.sample(data_to_process, remaining_limit)
        elif remaining_limit <= 0:
             data_to_process = []

    # --- DataLoader Setup ---
    # batch_size=8 + num_workers=8 for optimal speed
    if len(data_to_process) > 0:
        dataset = EvalDataset(data_to_process, args.image_dir, image_size=args.image_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, # 设置 Batch Size
            shuffle=False, 
            num_workers=8, 
            collate_fn=collate_fn, 
            pin_memory=True,
            prefetch_factor=2
        )
    else:
        dataloader = []

    result_file = open(result_jsonl_path, 'a', buffering=1)
    
    # --- Helper: Batch Generate Function ---
    def execute_batch_generate(batch_qs, batch_imgs_pil):
        """
        输入: List of questions, List of PIL images
        输出: List of generated answer strings
        """
        # 1. Template
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}] for q in batch_qs]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        
        # 2. Tokenize & Pad (padding=True -> Pad to longest in batch)
        # 注意：images 是一个 PIL List，Processor 会自动处理成 Tensor List 或 Stacked Tensor
        inputs = processor(text=texts, images=batch_imgs_pil, return_tensors="pt", padding=True).to(device)
        
        # 3. Generate
        with torch.no_grad():
            output_ids = model.vlm.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 4. Decode (Strip Input) - per-sample to handle different prompt lengths
        answers = []
        for i in range(output_ids.shape[0]):
            # Find actual (non-pad) input length for this sample
            input_mask = inputs.attention_mask[i]
            actual_input_len = input_mask.sum().item()
            # Decode only the generated tokens
            gen_ids = output_ids[i, int(actual_input_len):]
            ans = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            answers.append(ans)
        return answers

    # --- Main Inference Loop ---
    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    processed_count = 0

    for batch in pbar:
        if not batch: continue
        
        # Unpack Batch (Because collate_fn returned a list of dicts)
        b_ids = [b['id'] for b in batch]
        b_qs = [b['question'] for b in batch] # Strings
        b_imgs_pil = [Image.fromarray(b['image_np']) for b in batch] # PIL Images
        
        # --- PASS 1: Batch Generation ---
        b_answers = execute_batch_generate(b_qs, b_imgs_pil)

        # --- PASS 2: Check & Retry (Sub-Batch) ---
        retry_indices = []
        for i, ans in enumerate(b_answers):
            if "<SEG>" not in ans:
                retry_indices.append(i)
        
        if len(retry_indices) > 0:
            retry_qs = []
            retry_imgs_pil = []
            target_suffixes = ["Please respond with segmentation mask.", "Please output segmentation mask."]
            
            for idx in retry_indices:
                orig_q = b_qs[idx]
                if not any(s in orig_q for s in target_suffixes):
                    new_q = f"{orig_q} {random.choice(target_suffixes)}"
                else:
                    new_q = orig_q
                
                # 更新原始列表里的问题，这样 SAM forward 也能看到新的 prompt
                b_qs[idx] = new_q
                
                retry_qs.append(new_q)
                retry_imgs_pil.append(b_imgs_pil[idx])
            
            # Run Retry Inference
            if retry_qs:
                retry_results = execute_batch_generate(retry_qs, retry_imgs_pil)
                # Merge back
                for list_idx, res in zip(retry_indices, retry_results):
                    b_answers[list_idx] = res

        # --- SAM Decoding & Stats (Sequential) ---
        # 遍历 Batch 里的每一项进行 SAM 处理和统计
        for i in range(len(batch)):
            item_data = batch[i]
            
            # Prepare Inputs
            current_id = int(item_data['id'])
            question = b_qs[i]     # 可能被 retry 修改过
            gen_ans = b_answers[i] # 可能是 retry 后的结果
            image_pil = b_imgs_pil[i]
            
            image_sam_np = item_data['image_sam']
            gt_mask_np = item_data['gt_mask_np']
            original_size = item_data['original_size']
            image_filename = item_data['image_filename']
            image_np = item_data['image_np']
            image_path = item_data['image_path']

            gt_mask_tensor = torch.from_numpy(gt_mask_np).long().to(device)

            # Analyze Answer
            is_none_detected = False
            is_gen_failed = False
            
            gen_ans_lower = gen_ans.lower()
            if "step 5: result summary:" in gen_ans_lower:
                summary = gen_ans_lower[gen_ans_lower.find("step 5: result summary:"):]
                if "none" in summary or "not visible" in summary:
                    is_none_detected = True
            
            has_seg_token = "<SEG>" in gen_ans
            
            pred_mask_binary = None
            pred_mask_vis = None

            if is_none_detected or not has_seg_token:
                if not has_seg_token and not is_none_detected:
                    is_gen_failed = True
                pred_mask_binary = torch.zeros(original_size, dtype=torch.long, device=device)
                pred_mask_vis = np.zeros(original_size, dtype=np.uint8)
            else:
                # Run SAM (Keep sequential to avoid complex padding of prompts/masks)
                with torch.no_grad():
                    # Clean gen_ans of any vision tokens that could confuse the processor
                    clean_gen_ans = gen_ans.replace("<|vision_start|>", "").replace("<|image_pad|>", "").replace("<|vision_end|>", "")
                    msgs_fwd = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}, {"role": "assistant", "content": clean_gen_ans}]
                    text_fwd = processor.apply_chat_template(msgs_fwd, tokenize=False, add_generation_prompt=False)
                    try:
                        tok_out = processor(text=[text_fwd], images=[image_pil], return_tensors="pt")
                    except (IndexError, Exception) as e:
                        tqdm.write(f"[WARN] Forward processor error for id={current_id}: {e}")
                        pred_mask_binary = torch.zeros(original_size, dtype=torch.long, device=device)
                        pred_mask_vis = np.zeros(original_size, dtype=np.uint8)
                        # Skip to metrics calc
                        is_gen_failed = True
                        # Jump past the SAM block
                        tok_out = None

                    if tok_out is not None:
                        # Image Tensor
                        img_sam_tensor = preprocess_sam_image(torch.from_numpy(image_sam_np).permute(2,0,1).contiguous()).unsqueeze(0).to(device).to(torch_dtype)

                        labels = tok_out['input_ids'].clone(); labels[:] = -100
                        vlm_inputs = {"pixel_values": tok_out['pixel_values'].to(device)}
                        if 'image_grid_thw' in tok_out: vlm_inputs["image_grid_thw"] = tok_out['image_grid_thw'].to(device)

                        input_dict = {
                            "images": img_sam_tensor, "input_ids": tok_out['input_ids'].to(device),
                            "labels": labels.to(device), "attention_masks": tok_out['attention_mask'].to(device),
                            "vlm_inputs": vlm_inputs, "offset": torch.LongTensor([0, 1]).to(device),
                            "masks_list": [gt_mask_tensor], "label_list": [gt_mask_tensor],
                            "resize_list": [image_sam_np.shape[:2]], "change_list": [], "inference": True, "conversation_list": [text_fwd]
                        }
                        out_dict = model(**input_dict)
                        pred_masks = out_dict.get("pred_masks")

                if tok_out is None:
                    pred_masks = None

                if not pred_masks or pred_masks[0].shape[0] == 0:
                    pred_mask_binary = torch.zeros(original_size, dtype=torch.long, device=device)
                    pred_mask_vis = np.zeros(original_size, dtype=np.uint8)
                else:
                    pred_mask_tensor = pred_masks[0][-1]
                    pred_mask_np = (pred_mask_tensor > 0).detach().cpu().numpy().astype(np.uint8)
                    
                    if pred_mask_np.shape != tuple(original_size):
                        pred_mask_vis = cv2.resize(pred_mask_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                        pred_mask_binary = torch.from_numpy(pred_mask_vis).to(device, dtype=torch.long)
                    else:
                        pred_mask_vis = pred_mask_np
                        pred_mask_binary = (pred_mask_tensor > 0).long()

            # Calc IoU
            if pred_mask_binary.shape != gt_mask_tensor.shape:
                 pred_mask_binary = torch.nn.functional.interpolate(
                    pred_mask_binary.unsqueeze(0).unsqueeze(0).float(),
                    size=gt_mask_tensor.shape, mode="nearest"
                ).squeeze(0).squeeze(0).long()

            area_inter, area_union, _ = intersectionAndUnionGPU(pred_mask_binary, gt_mask_tensor, K=2, ignore_index=255)
            inter_val = area_inter[1].item()
            union_val = area_union[1].item()
            iou_val = inter_val / (union_val + 1e-10)

            # Update Stats
            if not is_gen_failed: update_stats(stats_orig_valid, inter_val, union_val, iou_val)
            update_stats(stats_orig_total, inter_val, union_val, iou_val)
            if current_id in filtered_ids:
                if not is_gen_failed: update_stats(stats_filt_valid, inter_val, union_val, iou_val)
                update_stats(stats_filt_total, inter_val, union_val, iou_val)

            # Write JSONL
            rec = {
                "id": current_id, "image_filename": image_filename,
                "intersection": inter_val, "union": union_val, "iou": iou_val,
                "is_valid": not is_gen_failed, "generated_answer": gen_ans
            }
            result_file.write(json.dumps(rec) + "\n")

            # Visualization
            processed_count += 1
            if args.save_vis and (processed_count % args.save_vis_freq == 0):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_prefix = f"{base_name}_{current_id}"
                
                pos_points, neg_points = parse_pos_neg_points(gen_ans)
                img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                
                gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
                gt_vis[gt_mask_np == 255] = (128, 128, 128)
                gt_vis[gt_mask_np == 1] = (255, 255, 255)
                img_gt = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
                
                img_pred = cv2.cvtColor((pred_mask_vis * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                
                overlay = image_np.copy()
                valid = pred_mask_vis.astype(bool) & (gt_mask_np != 255)
                overlay[valid] = (overlay[valid] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                
                pt_r = max(5, w // 200); pt_t = max(2, w // 300)
                for px, py in pos_points:
                    rx, ry = int((px/1000)*w), int((py/1000)*h)
                    if 0<=rx<w and 0<=ry<h:
                        cv2.circle(overlay, (rx, ry), pt_r, (0, 255, 255), -1)
                        cv2.circle(overlay, (rx, ry), pt_r+2, (255, 255, 255), pt_t)
                img_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

                if args.vis_mode == "combined":
                    comb = np.vstack((np.hstack((img_bgr, img_gt)), np.hstack((img_pred, img_overlay))))
                    # Top bar
                    wrapped = textwrap.wrap(f"Q: {question}", width=120)
                    top_h = max(100, len(wrapped)*34+20)
                    top = np.ones((top_h, comb.shape[1], 3), dtype=np.uint8)*40
                    y=30
                    for l in wrapped:
                        cv2.putText(top, l, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                        y+=34
                    cv2.imwrite(os.path.join(vis_dir, f"{save_prefix}_combined.png"), np.vstack((top, comb)))
                else:
                    cv2.imwrite(os.path.join(vis_dir, f"{save_prefix}_overlay.png"), img_overlay)

    result_file.close()

    # --- Summary ---
    print("\n" + "="*60 + "\nFINAL RESULTS SUMMARY\n" + "="*60)
    def print_stat(name, stats):
        if stats['count'] > 0:
            print(f"\n{name} (Items: {stats['count']})")
            print(f"  Avg IoU: {stats['iou_sum']/stats['count']:.4f}")
            print(f"  Overall IoU: {stats['intersection']/(stats['union']+1e-10):.4f}")
        else: print(f"\n{name}: No items.")

    print("\n--- ORIGINAL ---")
    print_stat("[A1] Valid", stats_orig_valid)
    print_stat("[A2] Total", stats_orig_total) 
    print("\n--- FILTERED ---")
    print_stat("[B1] Valid", stats_filt_valid)
    print_stat("[B2] Total", stats_filt_total)
    print("="*60 + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])