
import argparse
import os
import sys
from typing import List
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

# 请确保这些引用在你本地是可用的
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.vlmsam import VlmSamSegForCausalLM 

random.seed(42)

# ==============================================================================
# [LISA Alignment] Helper Functions from LISA utils
# ==============================================================================

def get_mask_from_json(json_path, img):
    """
    来自 LISA utils/data_processing.py
    直接处理 ReasonSeg 的原始 JSON，生成包含 0(BG), 1(FG), 255(Ignore) 的 Mask。
    """
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    
    height, width = img.shape[:2]

    ### sort polies by area
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

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """
    来自 LISA utils/utils.py
    标准的分割评测函数，支持 Ignore Index。
    """
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

# ==============================================================================
# Helper Functions
# ==============================================================================

def update_stats(stats_dict, intersection, union, iou):
    """更新统计字典"""
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

    parser.add_argument("--output_dir", default="./evaluate_results/ReasonSeg_results", type=str,
                        help="Root directory for output (JSONL and images)")
    parser.add_argument("--save_vis", action="store_true", help="Whether to save visualization images")
    parser.add_argument("--save_vis_freq", default=1, type=int, help="Save frequency (every N images)")
    parser.add_argument("--vis_mode", default="combined", type=str, choices=["combined", "separate"],
                        help="Visualization mode")

    parser.add_argument("--version", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Pretrained Qwen3-VL HuggingFace id or local path")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--limit", default=758, type=int)

    parser.add_argument("--reasonseg_root", default="./dataset/ReasonSeg", type=str,
                        help="Dir containing reasonseg_{val,test}_fixed.jsonl plus val/test image folders")
    parser.add_argument("--refcoco_json_base", default="./dataset/refcoco_eval", type=str,
                        help="Dir containing refcoco split jsonls")
    parser.add_argument("--refcoco_img_base", default="./dataset/train2014", type=str,
                        help="COCO train2014 image dir")

    parsed_args = parser.parse_args(hardcoded_cmd_args + args)

    refcoco_json_base = parsed_args.refcoco_json_base
    refcoco_img_base = parsed_args.refcoco_img_base
    reasonseg_base = parsed_args.reasonseg_root

    dataset_name = parsed_args.dataset
    if dataset_name is None:
        raise ValueError("--dataset must not be empty.")

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
# Main Loop
# ==============================================================================

def main(args):
    args = parse_args(args)

    # 准备目录
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
        
    result_jsonl_path = args.result_jsonl_path

    print(f"Output path: {result_jsonl_path}")

    # --- 1. 预加载净化数据集的 ID 列表 ---
    filtered_ids = set()
    if args.filtered_json_path and os.path.exists(args.filtered_json_path):
        print(f"Loading filtered IDs from: {args.filtered_json_path}")
        with open(args.filtered_json_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    filtered_ids.add(int(item['id']))
                except:
                    pass
        print(f"✅ Loaded {len(filtered_ids)} filtered IDs.")

    # --- 2. 恢复逻辑：加载已有的 JSONL 结果 ---
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
                    
                    inter = float(res['intersection'])
                    union = float(res['union'])
                    iou = float(res['iou'])
                    is_valid = res.get('is_valid', True) 
                    
                    update_stats(stats_orig_total, inter, union, iou)
                    if is_valid:
                        update_stats(stats_orig_valid, inter, union, iou)
                    
                    if res_id in filtered_ids:
                        update_stats(stats_filt_total, inter, union, iou)
                        if is_valid:
                            update_stats(stats_filt_valid, inter, union, iou)
                            
                except ValueError:
                    continue
        print(f"✅ Resumed stats from {len(processed_ids)} items.")
    
    # --- 模型初始化 ---
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = transformers.AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    tokenizer.add_tokens("<SEG>")
    tokenizer.add_tokens("<neg_SEG>")
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    args.neg_seg_token_idx = tokenizer("<neg_SEG>", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32

    config = AutoConfig.from_pretrained(args.version)
    kwargs = {
        "torch_dtype": torch_dtype, "model": args.version, "attention": "flash_attention_2",
        "train_mask_decoder": True, "out_dim": 256,
        "ce_loss_weight": 1.0, "dice_loss_weight": 0.5, "bce_loss_weight": 2.0,
    }
    model = VlmSamSegForCausalLM(config, seg_token_idx=args.seg_token_idx, neg_seg_token_idx=args.neg_seg_token_idx, use_SEG_token=args.use_SEG_token, **kwargs).to(device)
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
    transform_sam = ResizeLongestSide(args.image_size)

    # --- 数据加载 ---
    print(f"Loading ORIGINAL data from {args.json_path}...")
    data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    data_to_process = [d for d in data if int(d['id']) not in processed_ids]
    print(f"--- After resume filtering: {len(data_to_process)} items remaining (Total: {len(data)}). ---")

    if args.limit is not None and args.limit > 0:
        remaining_limit = args.limit - len(processed_ids)
        if remaining_limit > 0:
             if remaining_limit < len(data_to_process):
                 print(f"--- Sampling {remaining_limit} items from remaining data based on limit. ---")
                 data_to_process = random.sample(data_to_process, remaining_limit)
        else:
             print("--- Limit reached by processed items. Nothing to do. ---")
             data_to_process = []

    result_file = open(result_jsonl_path, 'a', buffering=1)
    pbar = tqdm(data_to_process, desc="Evaluating", unit="item")
    processed_count = 0

    for item in pbar:
        question = ""
        for conv in item['conversations']:
            if conv['from'] == 'human':
                raw_prompt = conv['value']
                question = raw_prompt.split('\n', 1)[-1].strip() 
                break
        
        if not question: continue
            
        image_filename = item['images'][0]
        image_path = os.path.join(args.image_dir, image_filename)
        if not os.path.exists(image_path): continue
        
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        original_size = image_np.shape[:2]

        # [LISA Alignment] Mask Loading
        gt_mask_combined_np = None
        gt_mask_tensor = None

        if "ReasonSeg/test" in args.image_dir:
            reasonseg_json_path = os.path.join(args.image_dir, os.path.splitext(image_filename)[0] + ".json")
        else:
            reasonseg_json_path = os.path.join(args.image_dir, os.path.splitext(image_filename)[0] + ".json")

        if os.path.exists(reasonseg_json_path):
            gt_mask_combined_np = get_mask_from_json(reasonseg_json_path, image_np)
            if gt_mask_combined_np.shape != original_size:
                gt_mask_combined_np = cv2.resize(gt_mask_combined_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask_tensor = torch.from_numpy(gt_mask_combined_np).long().to(device)
        else:
            print(f"⚠️ JSON not found: {reasonseg_json_path}")
            gt_mask_combined_np = np.zeros(original_size, dtype=np.uint8)
            gt_mask_tensor = torch.zeros(original_size, dtype=torch.long, device=device)

        def execute_generate(curr_question, curr_image):
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": curr_question}]}]
            gen_prompt_text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            vlm_gen_inputs = processor(text=[gen_prompt_text], images=[curr_image], return_tensors="pt").to(device)
            with torch.no_grad():
                prompt_len = vlm_gen_inputs['input_ids'].shape[1]
                output_ids = model.vlm.generate(**vlm_gen_inputs, max_new_tokens=2048, do_sample=False, eos_token_id=tokenizer.eos_token_id)
                generated_ids = output_ids[0, prompt_len:]
                ans = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            return ans
        
        target_suffixes = ["Please respond with segmentation mask.", "Please output segmentation mask."]
        if not any(s in question for s in target_suffixes):
            suffix = random.choice(target_suffixes)
            question = f"{question} {suffix}"
        
        generated_answer = execute_generate(question, image_pil)

        pred_mask_binary_for_iou = None
        pred_mask_np_for_vis = None
        is_none_detected = False 
        is_generation_failed = False 
        
        gen_ans_lower = generated_answer.lower()
        if "step 5: result summary:" in gen_ans_lower:
            summary_text = gen_ans_lower[gen_ans_lower.find("step 5: result summary:"):]
            if "none" in summary_text or "not visible" in summary_text or "no utensils" in summary_text:
                is_none_detected = True

        # ======================================================================
        # [NEW LOGIC] 点提取及有效性判定 
        # ======================================================================
        pos_points, neg_points = parse_pos_neg_points(generated_answer)
        has_points = (len(pos_points) > 0) or (len(neg_points) > 0)
        has_seg_token = "<SEG>" in generated_answer

        # 判定生成是否有效，并正确分流
        if is_none_detected:
            # 合理的空预测
            is_generation_failed = False
            pred_mask_binary_for_iou = torch.zeros(original_size, dtype=torch.long, device=device)
            pred_mask_np_for_vis = np.zeros(original_size, dtype=np.uint8)
        elif args.use_SEG_token and not has_seg_token:
            # 开启了 SEG 参数，但没生成
            is_generation_failed = True 
            pred_mask_binary_for_iou = torch.zeros(original_size, dtype=torch.long, device=device)
            pred_mask_np_for_vis = np.zeros(original_size, dtype=np.uint8)
        elif not args.use_SEG_token and not has_points:
            # 没开启 SEG 参数，强依赖于点提示，但也没有点
            is_generation_failed = True 
            pred_mask_binary_for_iou = torch.zeros(original_size, dtype=torch.long, device=device)
            pred_mask_np_for_vis = np.zeros(original_size, dtype=np.uint8)
        else:
            # 包含有效的提示信息，往下送到模型处理
            with torch.no_grad():
                messages_for_forward = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}, {"role": "assistant", "content": generated_answer}]
                text_forward = processor.apply_chat_template(messages_for_forward, tokenize=False, add_generation_prompt=False)
                tokenized_output = processor(text=[text_forward], images=[image_pil], return_tensors="pt")

                image_sam = transform_sam.apply_image(image_np)
                resize = image_sam.shape[:2]
                image_sam_tensor = preprocess_sam_image(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device).to(torch_dtype)
                
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
                
                # 如果不使用 SEG token，将点转换到目标分辨率后传入模型
                if not args.use_SEG_token and has_points:
                    point_coords = []
                    point_labels = []
                    for px, py in pos_points:
                        orig_x = (px / 1000.0) * original_size[1]
                        orig_y = (py / 1000.0) * original_size[0]
                        point_coords.append([orig_x, orig_y])
                        point_labels.append(1) # Positive points
                    for nx, ny in neg_points:
                        orig_x = (nx / 1000.0) * original_size[1]
                        orig_y = (ny / 1000.0) * original_size[0]
                        point_coords.append([orig_x, orig_y])
                        point_labels.append(0) # Negative points
                        
                    if len(point_coords) > 0:
                        point_coords_np = np.array(point_coords)
                        # 应用 SAM 的几何变换
                        point_coords_sam = transform_sam.apply_coords(point_coords_np, original_size)
                        point_coords_tensor = torch.from_numpy(point_coords_sam).unsqueeze(0).to(device).to(torch_dtype)
                        point_labels_tensor = torch.tensor(point_labels).unsqueeze(0).to(device).to(torch.long)
                        
                        # 兼容传入，防止由于底层变量名不一致导致遗漏
                        input_dict["point_coords"] = [point_coords_tensor]
                        input_dict["point_labels"] = [point_labels_tensor]
                        input_dict["point_coords_list"] = [point_coords_tensor]
                        input_dict["point_labels_list"] = [point_labels_tensor]
                
                output_dict = model(**input_dict)
                pred_masks = output_dict.get("pred_masks")

            if not pred_masks or pred_masks[0].shape[0] == 0:
                pred_mask_binary_for_iou = torch.zeros(original_size, dtype=torch.long, device=device)
                pred_mask_np_for_vis = np.zeros(original_size, dtype=np.uint8)
            else:
                pred_mask_tensor = pred_masks[0][-1]
                pred_mask_np_resized = (pred_mask_tensor > 0).detach().cpu().numpy().astype(np.uint8)

                if pred_mask_np_resized.shape != original_size:
                    pred_mask_np_for_vis = cv2.resize(pred_mask_np_resized, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                    pred_mask_tensor_orig_size = torch.from_numpy(pred_mask_np_for_vis).to(device, dtype=torch.long)
                    pred_mask_binary_for_iou = pred_mask_tensor_orig_size
                else:
                    pred_mask_np_for_vis = pred_mask_np_resized
                    pred_mask_binary_for_iou = (pred_mask_tensor > 0).long()

        # Resize Check
        if pred_mask_binary_for_iou.shape != gt_mask_tensor.shape:
             pred_mask_binary_for_iou = torch.nn.functional.interpolate(
                pred_mask_binary_for_iou.unsqueeze(0).unsqueeze(0).float(),
                size=gt_mask_tensor.shape,
                mode="nearest"
            ).squeeze(0).squeeze(0).long()

        # IoU Calc
        area_inter, area_union, _ = intersectionAndUnionGPU(
            pred_mask_binary_for_iou, gt_mask_tensor, K=2, ignore_index=255
        )
        intersection_val = area_inter[1].item()
        union_val = area_union[1].item()
        iou_val = intersection_val / (union_val + 1e-10)

        # Update Stats
        if not is_generation_failed:
            update_stats(stats_orig_valid, intersection_val, union_val, iou_val)
        update_stats(stats_orig_total, intersection_val, union_val, iou_val)

        current_id = int(item['id'])
        if current_id in filtered_ids:
            if not is_generation_failed:
                update_stats(stats_filt_valid, intersection_val, union_val, iou_val)
            update_stats(stats_filt_total, intersection_val, union_val, iou_val)

        # --- [NEW] 结果实时写入 JSONL ---
        result_record = {
            "id": current_id,
            "image_filename": image_filename,
            "intersection": intersection_val,
            "union": union_val,
            "iou": iou_val,
            "is_valid": not is_generation_failed,
            "generated_answer": generated_answer
        }
        result_file.write(json.dumps(result_record) + "\n")
        
        # --- [NEW] 可视化控制逻辑 ---
        processed_count += 1
        
        if args.save_vis and (processed_count % args.save_vis_freq == 0):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_prefix = f"{base_name}_{item['id']}"
            
            img_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            h, w, _ = img_original_bgr.shape
            
            # GT 可视化
            gt_vis = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)
            ignore_region = (gt_mask_combined_np == 255)
            target_region = (gt_mask_combined_np == 1)
            gt_vis[ignore_region] = (128, 128, 128)
            gt_vis[target_region] = (255, 255, 255)
            img_gt_mask_bgr = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
            
            # Pred 可视化
            pred_mask_gray = (pred_mask_np_for_vis.astype(np.uint8) * 255)
            img_pred_mask_bgr = cv2.cvtColor(pred_mask_gray, cv2.COLOR_GRAY2BGR)

            # Overlay 可视化
            overlay_img = image_np.copy()
            valid_pred = pred_mask_np_for_vis.astype(bool) & (~ignore_region)
            overlay_img[valid_pred] = (overlay_img[valid_pred] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            overlay_img[ignore_region] = (overlay_img[ignore_region] * 0.6 + np.array([160, 160, 160]) * 0.4).astype(np.uint8)

            point_radius = max(5, w // 200); point_thickness = max(2, w // 300)
            for px, py in pos_points:
                real_x = int((px / 1000.0) * w); real_y = int((py / 1000.0) * h)
                if 0 <= real_x < w and 0 <= real_y < h:
                    cv2.circle(overlay_img, (real_x, real_y), point_radius, (0, 255, 255), -1) 
                    cv2.circle(overlay_img, (real_x, real_y), point_radius + 2, (255, 255, 255), point_thickness)
            for nx, ny in neg_points:
                real_x = int((nx / 1000.0) * w); real_y = int((ny / 1000.0) * h)
                if 0 <= real_x < w and 0 <= real_y < h:
                    cv2.circle(overlay_img, (real_x, real_y), point_radius, (255, 0, 255), -1)
                    cv2.circle(overlay_img, (real_x, real_y), point_radius + 2, (255, 255, 255), point_thickness)
            img_overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

            if args.vis_mode == "combined":
                combined_image = np.vstack((np.hstack((img_original_bgr, img_gt_mask_bgr)), np.hstack((img_pred_mask_bgr, img_overlay_bgr))))
                font = cv2.FONT_HERSHEY_SIMPLEX
                top_bar_height = 100; w_comb = combined_image.shape[1]
                wrapped_lines = textwrap.wrap(f"Q: {question}", width=120)
                needed_height = len(wrapped_lines) * 34 + 20
                top_bar = np.ones((max(top_bar_height, needed_height), w_comb, 3), dtype=np.uint8) * 40
                y_off = 30
                for line in wrapped_lines:
                    cv2.putText(top_bar, line, (20, y_off), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    y_off += 34
                
                save_path = os.path.join(vis_dir, f"{save_prefix}_combined.png")
                cv2.imwrite(save_path, np.vstack((top_bar, combined_image)))
                
            elif args.vis_mode == "separate":
                cv2.imwrite(os.path.join(vis_dir, f"{save_prefix}_origin.png"), img_original_bgr)
                cv2.imwrite(os.path.join(vis_dir, f"{save_prefix}_gt.png"), img_gt_mask_bgr)
                cv2.imwrite(os.path.join(vis_dir, f"{save_prefix}_pred.png"), img_pred_mask_bgr)
                cv2.imwrite(os.path.join(vis_dir, f"{save_prefix}_overlay.png"), img_overlay_bgr)

    # 关闭结果文件
    result_file.close()

    # --- Final Report ---
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    def print_stat(name, stats):
        if stats['count'] > 0:
            giou = stats['iou_sum'] / stats['count']
            ciou = stats['intersection'] / (stats['union'] + 1e-10)
            print(f"\n{name} (Items: {stats['count']})")
            print(f"  Average IoU (GIoU): {giou:.4f}")
            print(f"  Overall IoU (cIoU): {ciou:.4f}")
        else:
            print(f"\n{name}: No items processed.")

    # 1. Original
    print("\n--- ORIGINAL DATASET (All Files) ---")
    print_stat("[A1] Valid Only (Skip Invalid Generation)", stats_orig_valid)
    print_stat("[A2] Total (Include Invalid as Empty Mask)", stats_orig_total) 

    # 2. Filtered
    print("\n--- FILTERED DATASET (Subset via ID) ---")
    print_stat("[B1] Valid Only (Skip Invalid Generation)", stats_filt_valid)
    print_stat("[B2] Total (Include Invalid as Empty Mask)", stats_filt_total)
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])