import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .conversation import get_default_conv_template  
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


# --- 新增的辅助函数 ---
def create_mask_from_polygons(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 过滤掉空的多边形列表
    if not polygons or not any(polygons):
        return mask.astype(np.float32)

    # cv2.fillPoly 需要一个由numpy数组组成的列表
    pts_list = [np.array(poly, dtype=np.int32) for poly in polygons if poly]
    
    if not pts_list:
        return mask.astype(np.float32)

    cv2.fillPoly(mask, pts=pts_list, color=1)
    return mask.astype(np.float32)

    
class OverlapReasonsegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        json_file_path,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        overlap_reasonseg_data="Overlap_ReasonSeg|train",
        explanatory=0.1,
        model_name="qwen_vl", 
    ):
        """
        Initialize the ReasonSegDataset with dataset-specific configurations.

        Args:
            base_image_dir (str): Base directory for dataset files.
            tokenizer: Tokenizer for text processing.
            samples_per_epoch (int): Number of samples per epoch.
            precision (str): Data precision ("fp32" or "fp16").
            image_size (int): Target image size for resizing.
            num_classes_per_sample (int): Number of classes to sample per image.
            exclude_val (bool): Whether to exclude validation data.
            overlap_reasonseg_data (str): Dataset and splits to use, separated by "|".
            explanatory (float): Probability of including explanatory questions.
            model_name (str): Model name ("llava" or "qwen_vl").
        """
        self.exclude_val = exclude_val
        self.overlap_reasonseg_data = overlap_reasonseg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = os.path.join(base_image_dir, "COCO", "train2017")
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.model_name = model_name.lower()

        # 从单个JSON文件加载所有数据
        print(f"Loading data from {json_file_path}...")
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples.")


    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input for SAM."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # 使用取模运算来确保索引在数据范围内
        idx = idx % len(self.data)
        item = self.data[idx]

        # 构造图像路径
        image_name = item['image']
        image_path = os.path.join(self.base_image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # 从JSON中的"polygons"字段创建掩码
        polygons = item['polygons']
        mask = create_mask_from_polygons(polygons, ori_size[0], ori_size[1])
        
        # 预处理VLM的图像
        image_vlm = None
        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)  # Qwen2.5-VL 需要 PIL Image

        # 预处理SAM的图像
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        # 从JSON中直接获取对话
        human_turn = item['conversations'][0]['value']
        gpt_turn = item['conversations'][1]['value']

        # 移除问题中的<image> token，因为它将在后面被模板添加
        question = human_turn.replace('<image>\n', '').strip()
        answer = gpt_turn
        # print(f"GT回答：{answer}")

        # 格式化对话
        conv = get_default_conv_template(self.model_name).copy()
        if "qwen" in self.model_name:
            # Qwen-VL 的模板需要特殊的 tokens
            user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{question}"
        else:
            # LLaVA 或其他模型的通用模板
            user_message = "<image>\n" + question
        
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        # 预处理SAM的图像张量
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # 准备掩码和标签张量
        masks = torch.from_numpy(mask).unsqueeze(0)  # 增加一个维度 -> (1, H, W)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # 返回与原始Dataset格式一致的元组
        return (
            image_path,
            image,
            image_vlm,
            conversations,
            masks,
            label,
            resize,
            [question],  # 为了保持格式一致，将问题放入列表中
            [question],  # 同上
        )
