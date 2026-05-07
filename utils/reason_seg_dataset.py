# import glob
# import json
# import os
# import random

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image

# from .conversation import get_default_conv_template  
# from model.segment_anything.utils.transforms import ResizeLongestSide

# from .data_processing import get_mask_from_json
# from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
#                     EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
#                     SHORT_QUESTION_LIST)

# class ReasonSegDataset(torch.utils.data.Dataset):
#     pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
#     img_size = 1024
#     ignore_label = 255

#     def __init__(
#         self,
#         base_image_dir,
#         tokenizer,
#         samples_per_epoch=500 * 8 * 2 * 10,
#         precision: str = "fp32",
#         image_size: int = 224,
#         num_classes_per_sample: int = 3,
#         exclude_val=False,
#         reason_seg_data="ReasonSeg|train",
#         explanatory=0.1,
#         model_name="qwen_vl", 
#     ):
#         """
#         Initialize the ReasonSegDataset with dataset-specific configurations.

#         Args:
#             base_image_dir (str): Base directory for dataset files.
#             tokenizer: Tokenizer for text processing.
#             samples_per_epoch (int): Number of samples per epoch.
#             precision (str): Data precision ("fp32" or "fp16").
#             image_size (int): Target image size for resizing.
#             num_classes_per_sample (int): Number of classes to sample per image.
#             exclude_val (bool): Whether to exclude validation data.
#             reason_seg_data (str): Dataset and splits to use, separated by "|".
#             explanatory (float): Probability of including explanatory questions.
#             model_name (str): Model name ("llava" or "qwen_vl").
#         """
#         self.exclude_val = exclude_val
#         self.reason_seg_data = reason_seg_data
#         self.samples_per_epoch = samples_per_epoch
#         self.explanatory = explanatory
#         self.num_classes_per_sample = num_classes_per_sample

#         self.base_image_dir = base_image_dir
#         self.image_size = image_size
#         self.tokenizer = tokenizer
#         self.precision = precision
#         self.transform = ResizeLongestSide(image_size)
#         self.model_name = model_name.lower()

#         self.short_question_list = SHORT_QUESTION_LIST
#         self.long_question_list = LONG_QUESTION_LIST
#         self.answer_list = ANSWER_LIST

#         reason_seg_data, splits = reason_seg_data.split("|")
#         splits = splits.split("_")
#         images = []
#         for split in splits:
#             images_split = glob.glob(
#                 os.path.join(
#                     base_image_dir, reason_seg_data, split, "*.jpg"
#                 )
#             )
#             images.extend(images_split)
#         jsons = [path.replace(".jpg", ".json") for path in images]
#         self.reason_seg_data = (images, jsons)

#         print("Number of reason_seg samples: ", len(images))

#         if explanatory != -1:
#             self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
#             self.img_to_explanation = {}
#             with open(
#                 os.path.join(
#                     base_image_dir,
#                     reason_seg_data,
#                     "explanatory",
#                     "train.json",
#                 )
#             ) as f:
#                 items = json.load(f)
#             for item in items:
#                 img_name = item["image"]
#                 self.img_to_explanation[img_name] = {
#                     "query": item["query"],
#                     "outputs": item["outputs"],
#                 }

#             print("Number of explanatory samples: ", len(self.img_to_explanation))

#     def __len__(self):
#         return self.samples_per_epoch

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input for SAM."""
#         x = (x - self.pixel_mean) / self.pixel_std
#         h, w = x.shape[-2:]
#         padh = self.img_size - h
#         padw = self.img_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x

#     def __getitem__(self, idx):
#         """根据索引获取一个样本，并支持循环采样。"""
#         actual_idx = idx % len(self.data_samples)
#         sample_data = self.data_samples[actual_idx]
        
#         image_filename = sample_data['image']
#         height = sample_data['height']
#         width = sample_data['width']
#         # This is a list of compressed RLE strings
#         segmentation_rle_strings = sample_data['segmentation'] 
#         conversation_data = sample_data['conversation']

#         local_image_path = os.path.join(self.image_base_dir, image_filename)

#         # --- 从本地文件系统读取图像 (This part is correct) ---
#         try:
#             image_pil = Image.open(local_image_path).convert('RGB')
#             image = np.array(image_pil)
#         except FileNotFoundError:
#             print(f"Error: Image file not found at {local_image_path}. Trying another sample.")
#             return self.__getitem__(random.randint(0, len(self) - 1))
#         except Exception as e:
#             print(f"Error reading or processing image {local_image_path}: {e}. Trying another sample.")
#             return self.__getitem__(random.randint(0, len(self) - 1))

#         # --- 后续的数据处理部分 ---
#         image_vlm = image_pil
#         image = self.transform.apply_image(image)
#         resize = image.shape[:2]

#         # ... (Your conversation processing logic remains here) ...
#         conv = get_default_conv_template(self.model_name).copy()
#         # (Please keep your original conversation processing logic)
#         conversations = [] # Placeholder

#         image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
#         # <<< MODIFICATION START: Correctly format RLE data for pycocotools >>>
#         try:
#             # 1. Create a list of dictionaries in the format pycocotools expects for RLE.
#             rle_dicts = []
#             for rle_string in segmentation_rle_strings:
#                 rle_dicts.append({
#                     'counts': rle_string, 
#                     'size': [height, width]
#                 })

#             # 2. Now, pass the correctly formatted list of dictionaries to the function.
#             #    This tells the function to use its RLE decoder.
#             rles = mask.frPyObjects(rle_dicts, height, width)
#             decoded_mask = mask.decode(rles)
            
#             # Merge masks if there are multiple RLEs for the same object
#             if len(decoded_mask.shape) == 3:
#                 merged_mask = np.sum(decoded_mask, axis=2, dtype=np.uint8)
#             else:
#                 merged_mask = decoded_mask.astype(np.uint8)
            
#             merged_mask[merged_mask > 0] = 1 # Ensure the mask is binary (0 or 1)
#             masks = torch.from_numpy(merged_mask).unsqueeze(0) # Add a channel dimension

#         except Exception as e:
#             print(f"Error decoding mask for image {image_filename}: {e}. Trying another sample.")
#             return self.__getitem__(random.randint(0, len(self) - 1))
#         # <<< MODIFICATION END >>>

#         label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        
#         # Adjust your return values as needed by your training loop
#         return (
#             local_image_path,
#             image,
#             image_vlm,
#             conversations,
#             masks,
#             label,
#             resize,
#             [], # Placeholder for questions
#             [], # Placeholder for sampled_classes
#         )

import glob
import json
import os
import random
import re

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


# --------------------------------------------------------------------------------
# Helper Functions (Mask Processing)
# --------------------------------------------------------------------------------

def get_mask_center(mask_np):
    """
    Calculate the centroid (mean center) of the foreground mask.
    Modified to select the largest connected component first (ignoring noise),
    then calculate the center.
    """
    # 转为 uint8 二值图
    mask_np = (mask_np > 0).astype(np.uint8)
    
    # 如果 mask 全黑，直接返回
    if mask_np.sum() == 0:
        return [0, 0]

    # 1. 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    # 2. 决定计算哪个区域的矩 (抗噪：只取最大连通域)
    if num_labels > 1:
        # stats[:, 4] 是面积列, stats[1:] 排除了背景 label 0
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_to_compute = (labels == largest_label).astype(np.uint8)
        M = cv2.moments(mask_to_compute)
    else:
        M = cv2.moments(mask_np)
        
    # 3. 计算重心坐标
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        ys, xs = np.where(mask_np)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

    # 4. 最后一道防线：确保点在 mask 内部
    if mask_np[cy, cx] == 0:
        ys, xs = np.where(mask_np)
        idx = len(xs) // 2
        cx, cy = int(xs[idx]), int(ys[idx])

    return [cx, cy]


def get_mask_points(mask_np, num_points=1):
    """
    获取多个点：第1个是重心，其余是随机前景点。
    Args:
        mask_np: 二值掩码
        num_points: 需要采样的点数
    """
    # 1. 获取重心作为第一个点
    center = get_mask_center(mask_np)
    points = [center]

    if num_points <= 1:
        return points

    # 2. 获取Mask所有前景点的坐标
    ys, xs = np.where(mask_np > 0)
    
    # 如果mask极小或为空，直接重复中心点
    if len(xs) == 0:
        return points * num_points

    all_coords = list(zip(xs, ys))
    
    # 3. 随机采样剩余的点
    # 过滤掉中心点（避免重复），如果前景点太少就不强求过滤
    candidates = [p for p in all_coords if not (p[0] == center[0] and p[1] == center[1])]
    
    num_to_sample = num_points - 1
    
    if len(candidates) >= num_to_sample:
        extras = random.sample(candidates, num_to_sample)
        points.extend(extras)
    else:
        # 如果前景点不够（mask极小），允许重复采样
        points.extend(candidates)
        # 剩下的缺口从 all_coords 里随机补（包含重复）
        while len(points) < num_points:
            points.append(random.choice(all_coords))

    return points



class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 2,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        model_name="qwen_vl",
        num_points=1,  # <--- [新增参数] 默认为1，你可以改为3
        use_SEG_token=True,
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
            reason_seg_data (str): Dataset and splits to use, separated by "|".
            explanatory (float): Probability of including explanatory questions.
            model_name (str): Model name ("llava" or "qwen_vl").
        """
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.num_points = num_points # <--- [保存参数]
        self.use_SEG_token = use_SEG_token

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.model_name = model_name.lower()

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        # self.reason_seg_data = (images, jsons)

        print(f"Total images found: {len(images)}")

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    reason_seg_data,
                    "explanatory",
                    "correct_train_data.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("Number of explanatory samples: ", len(self.img_to_explanation))

            # **关键修改：过滤只保留有 explanation 的图片，即清洗过后的训练数据**
            filtered_images = []
            filtered_jsons = []
            
            for img_path, json_path in zip(images, jsons):
                img_name = img_path.split("/")[-1]
                if img_name in self.img_to_explanation:
                    filtered_images.append(img_path)
                    filtered_jsons.append(json_path)
            
            print(f"Filtered images (with explanation): {len(filtered_images)}")
            print(f"Filtered out (without explanation): {len(images) - len(filtered_images)}")
            
            # 更新数据集为过滤后的版本
            images = filtered_images
            jsons = filtered_jsons
        self.reason_seg_data = (images, jsons)
        print(f"Final dataset size: {len(images)}")

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

    ###################################################################原始版本代码#########################################################################
    # def __getitem__(self, idx):
    #     """Fetch a sample with dataset-specific logic."""
    #     images, jsons = self.reason_seg_data
    #     idx = random.randint(0, len(images) - 1)
    #     image_path = images[idx]
    #     json_path = jsons[idx]

    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     ori_size = image.shape[:2]

    #     # Preprocess image for VLM
    #     if "qwen" in self.model_name:
    #         image_vlm = Image.fromarray(image)  # PIL Image for Qwen2.5-VL

    #     mask, sents, is_sentence = get_mask_from_json(json_path, image)
    #     if len(sents) >= self.num_classes_per_sample:
    #         sampled_inds = random.sample(range(len(sents)), self.num_classes_per_sample)
    #     else:
    #         sampled_inds = list(range(len(sents)))
    #     sampled_sents = [sents[i] for i in sampled_inds]
    #     sampled_masks = [(mask == 1).astype(np.float32) for _ in range(len(sampled_inds))]

    #     image = self.transform.apply_image(image) 
    #     resize = image.shape[:2]

    #     image_name = image_path.split("/")[-1]
    #     if self.explanatory != -1 and image_name in self.img_to_explanation:
    #         if random.random() < self.explanatory:
    #             choice = 2
    #         else:
    #             choice = random.randint(0, 1)

    #     questions = []
    #     answers = []
    #     for text in sampled_sents:
    #         if is_sentence:
    #             question_template = random.choice(self.long_question_list)
    #             questions.append(question_template.format(sent=text))
    #         else:
    #             question_template = random.choice(self.short_question_list)
    #             questions.append(question_template.format(class_name=text.lower()))

    #         img_name = image_path.split("/")[-1]
    #         if self.explanatory != -1 and img_name in self.img_to_explanation:
    #             if choice == 0:  # [SEG] token
    #                 answers.append(random.choice(self.answer_list))
    #             elif choice == 1:  # [SEG] token + text answer
    #                 answer = self.img_to_explanation[img_name]["outputs"]
    #                 answer = random.choice(self.answer_list) + " {}".format(answer)
    #                 questions[-1] = (
    #                     DEFAULT_IMAGE_TOKEN
    #                     + "\n"
    #                     + text
    #                     + " {}".format(random.choice(self.explanatory_question_list))
    #                 )
    #                 answers.append(answer)
    #             elif choice == 2:  # vanilla text answer
    #                 answer = self.img_to_explanation[img_name]["outputs"]
    #                 questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
    #                 answers.append(answer)
    #             else:
    #                 raise ValueError("Not implemented yet.")
    #         else:
    #             answers.append(random.choice(self.answer_list))

    #     conversations = []
    #     for q, a in zip(questions, answers):
    #         conv = get_default_conv_template(self.model_name).copy()
    #         if "qwen" in self.model_name:
    #             user_message = f"<|vision_start|><|image_pad|><|vision_end>\n{q}"
    #         else:
    #             user_message = "<image>" + "\n" + q
    #         conv.append_message(conv.roles[0], user_message)
    #         conv.append_message(conv.roles[1], a)
    #         conversations.append(conv.get_prompt())

    #     image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

    #     image_name = image_path.split("/")[-1]
    #     if (
    #         self.explanatory != -1
    #         and image_name in self.img_to_explanation
    #         and choice == 2
    #     ):
    #         masks = torch.rand(0, *ori_size)
    #         label = torch.ones(ori_size) * self.ignore_label
    #     else:
    #         masks = np.stack(sampled_masks, axis=0)
    #         masks = torch.from_numpy(masks)
    #         label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

    #     return (
    #         image_path,
    #         image,
    #         image_vlm,
    #         conversations,
    #         masks,
    #         label,
    #         resize,
    #         questions,
    #         sampled_sents,
    #     )


    # ################################################sentence采取CoT，非sentence不采取CoT##################################################

    def normalize_coordinates_in_text(self, text, width, height):
        """
        将文本中的坐标归一化到 [0, 1000] 范围
        """
        def normalize_coord_match(match):
            # 提取 x, y 坐标
            x_str, y_str = match.group(1), match.group(2)
            abs_x, abs_y = float(x_str), float(y_str)
            
            # 归一化到 [0, 1000]
            norm_x = int(round(abs_x / width * 1000))
            norm_y = int(round(abs_y / height * 1000))
            
            # 确保不越界
            norm_x = max(0, min(1000, norm_x))
            norm_y = max(0, min(1000, norm_y))
            
            # 返回归一化后的坐标字符串
            return f"[{norm_x}, {norm_y}]"
        
        # 匹配 [x, y] 格式的坐标,支持整数和小数
        pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        normalized_text = re.sub(pattern, normalize_coord_match, text)
        
        return normalized_text


    ##################################################带有point###################################################

    def __getitem__(self, idx):
        """Fetch a sample with dataset-specific logic."""

        # --- 数据加载和图像预处理 ---
        images, jsons = self.reason_seg_data
        
        while True:
            try:
                idx = random.randint(0, len(images) - 1)
                image_path = images[idx]
                json_path = jsons[idx]

                image = cv2.imread(image_path)
                if image is None: 
                    print(f"Warning: Failed to load image {image_path}. Retrying...")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                # ori_size = image.shape[:2]

                # Preprocess image for VLM 
                if "qwen" in self.model_name:
                    image_vlm = Image.fromarray(image) 
                else:
                    image_vlm = None 

                # --- 获取 mask 和 sentences ---
                mask, sents, is_sentence = get_mask_from_json(json_path, image) 
                if not sents: 
                    print(f"Warning: No sentences found in {json_path}. Retrying...")
                    continue

                # --- 强制单目标采样 ---
                sampled_ind = random.choice(range(len(sents))) 
                sampled_sents = [sents[sampled_ind]] 
                
                # --- 获取掩码 ---
                final_mask_np = (mask == 1).astype(np.uint8)
                sampled_masks = [final_mask_np.astype(np.float32)] 

                # ==============================================================================
                # [修改重点]: 使用 get_mask_points 获取多个点，并循环归一化
                # ==============================================================================
                
                # 1. 获取点 (重心 + 随机点)
                # 读取 self.num_points, 如果未定义则默认为 1
                target_num_points = getattr(self, 'num_points', 1)
                target_points = get_mask_points(final_mask_np, num_points=target_num_points)
                
                # 2. 准备循环归一化
                height, width = image.shape[:2]
                norm_points_strs = []

                for pt in target_points:
                    abs_x, abs_y = pt
                    
                    # 归一化到 [0, 1000] (Qwen3 风格)
                    norm_x = int(round(abs_x / width * 1000))
                    norm_y = int(round(abs_y / height * 1000))
                    
                    # 越界保护
                    norm_x = max(0, min(1000, norm_x))
                    norm_y = max(0, min(1000, norm_y))
                    
                    norm_points_strs.append(f"[{norm_x}, {norm_y}]")
                
                # 拼接所有点字符串, 结果如: "[250, 300], [255, 310]"
                points_string = ", ".join(norm_points_strs)
                
                # ==============================================================================

                # --- Image transformation for SAM ---
                image_sam_processed = self.transform.apply_image(image)
                resize = image_sam_processed.shape[:2]

                # --- 生成 Question 和 Answer ---
                questions = []
                answers = []
                
                # 模式选项
                answer_mode = 'simple_point' # explanation或simple_point

                for text in sampled_sents:
                    img_name = image_path.split("/")[-1]

                    # -------------------------------------------------
                    # 第一步：根据 is_sentence 决定 Question 的格式
                    # -------------------------------------------------
                    if is_sentence:
                        question_template = random.choice(self.long_question_list)
                        questions.append(question_template.format(sent=text))
                    else:
                        question_template = random.choice(self.short_question_list)
                        questions.append(question_template.format(class_name=text.lower()))

                    # -------------------------------------------------
                    # 第二步：根据 answer_mode 决定 Answer 的格式
                    # -------------------------------------------------
                    if answer_mode == 'explanation':
                        # --- 方案 1: 使用预定义的答案 (通常用于 Chain-of-Thought) ---
                        # 注意：Explanation 里的文本是固定的，所以这里我们通常
                        # 只对它里面的坐标进行归一化，而不强行插入随机生成的 points_string。
                        if img_name in self.img_to_explanation:
                            raw_output = self.img_to_explanation[img_name]['outputs']
                            normalized_output = self.normalize_coordinates_in_text(raw_output, width, height)
                            answers.append(normalized_output)
                        else:
                            # 如果找不到 explanation，回退到 simple_point 模式
                            answer_string = f"<think></think><answer>The mask is {points_string}.<SEG></answer>"
                            answers.append(answer_string)

                    elif answer_mode == 'simple_point':
                        # --- 方案 2: Think 置空 + 多个坐标点 ---
                        # 使用上面生成的包含多个点的 points_string
                        if self.use_SEG_token:
                            answer_string = f"<think></think><answer>The mask is {points_string}.<SEG></answer>"
                        else:
                            answer_string = f"<think></think><answer>The mask is {points_string}.</answer>"
                            # answer_string = "<SEG>"
                        answers.append(answer_string)

                    else:
                        raise ValueError(f"Unknown answer_mode: {answer_mode}")

                # --- 生成 Conversation ---
                conversations = []
                for q, a in zip(questions, answers):
                    conv = get_default_conv_template(self.model_name).copy() 
                    if "qwen" in self.model_name:
                        user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}" 
                    else:
                        user_message = DEFAULT_IMAGE_TOKEN + "\n" + q 
                    conv.append_message(conv.roles[0], user_message) 
                    conv.append_message(conv.roles[1], a) 
                    conversations.append(conv.get_prompt()) 

                # --- Final SAM image preprocessing ---
                image_sam = self.preprocess(torch.from_numpy(image_sam_processed).permute(2, 0, 1).contiguous()) 

                # --- 准备 Mask 和 Label ---
                masks = np.stack(sampled_masks, axis=0) 
                masks = torch.from_numpy(masks) 
                label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label 

                # --- 返回样本 ---
                return (
                    image_path,
                    image_sam,
                    image_vlm,
                    conversations, 
                    masks,         
                    label,
                    resize,
                    questions,    
                    sampled_sents, 
                )
            except Exception as e:
                print(f"Error processing sample idx {idx}, image {image_path}: {e}. Retrying...")
                # 继续循环以重试


    ##################################################不带point############################################################
    # def __getitem__(self, idx):
    #     """Fetch a sample with dataset-specific logic."""

    #     # --- 数据加载和图像预处理 (保持不变) ---
    #     images, jsons = self.reason_seg_data
    #     # Loop to ensure a valid sample is returned (Optional but recommended) img_name
    #     while True:
    #         try:
    #             idx = random.randint(0, len(images) - 1)
    #             image_path = images[idx]
    #             json_path = jsons[idx]

    #             image = cv2.imread(image_path)
    #             if image is None: # Check if image loaded correctly
    #                 print(f"Warning: Failed to load image {image_path}. Retrying...")
    #                 continue
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #
    #             ori_size = image.shape[:2]

    #             # Preprocess image for VLM (Qwen expects PIL)
    #             if "qwen" in self.model_name:
    #                 image_vlm = Image.fromarray(image) #
    #             else:
    #                 image_vlm = None # Handle other models if necessary

    #             # --- 获取 mask 和 sentences ---
    #             mask, sents, is_sentence = get_mask_from_json(json_path, image) #
    #             if not sents: # If no sentences found in JSON, retry
    #                 print(f"Warning: No sentences found in {json_path}. Retrying...")
    #                 continue

    #             # --- 强制单目标采样 ---
    #             sampled_ind = random.choice(range(len(sents))) # Choose one index randomly
    #             sampled_sents = [sents[sampled_ind]] # List with single sentence
    #             # sampled_masks 需要对应 sampled_sents，所以只取一个
    #             sampled_masks = [(mask == 1).astype(np.float32)] # List with single mask

    #             # --- Image transformation for SAM ---
    #             image_sam_processed = self.transform.apply_image(image) #
    #             resize = image_sam_processed.shape[:2] #

    #             # --- 生成 Question 和 Answer ---
    #             questions = []
    #             answers = []

    #             # 因为 sampled_sents 只有一个元素，循环只会执行一次
    #             for text in sampled_sents:
    #                 img_name = image_path.split("/")[-1]
    #                 # 1. 创建 Question (逻辑不变)
    #                 if is_sentence:
    #                     question_template = random.choice(self.long_question_list) #
    #                     questions.append(question_template.format(sent=text))
    #                     answers.append(self.img_to_explanation[img_name]["outputs"])
    #                     # print(f"Question: {question_template.format(sent=text)}")
    #                 else:
    #                     question_template = random.choice(self.short_question_list) #
    #                     questions.append(question_template.format(class_name=text.lower()))
    #                     sampled_sent_lower = text.lower() # 使用当前循环的 text
    #                     answer_string = f"<think>The target is the {sampled_sent_lower}.</think><answer>Target Object: <pos><SEG></pos></answer>"
    #                     # print(f"Question: {question_template.format(class_name=text.lower())}")

    #                 # img_name = image_path.split("/")[-1] #

    #                 # # --- 修改答案生成逻辑 ---
    #                 # # 检查是否存在对应的解释性数据
    #                 # if self.explanatory != -1 and img_name in self.img_to_explanation:
    #                 #     # 如果存在，直接使用 explanatory json 中的 "outputs" 字段
    #                 #     # 这个 "outputs" 字段已经包含了 <think>...</think><answer>...<pos>[SEG]</pos>...</answer> 格式
    #                 #     answers.append(self.img_to_explanation[img_name]["outputs"]) #
    #                 #     # print(f"Answer: {self.img_to_explanation[img_name]['outputs']}")
    #                 # else:
    #                 #     # 如果不存在解释性数据，使用你自定义的单目标格式
    #                 #     sampled_sent_lower = text.lower() # 使用当前循环的 text
    #                 #     answer_string = f"<think>The target is the {sampled_sent_lower}.</think><answer>Target Object: <pos><SEG></pos></answer>"
    #                 #     # print(f"Answer: {answer_string}")
    #                 #     answers.append(answer_string)
    #                 # # --- 答案生成逻辑修改结束 ---

    #             # --- 生成 Conversation (逻辑不变，现在只处理一个 Q&A 对) ---
    #             conversations = []
    #             # zip(questions, answers) 现在只会包含一对
    #             for q, a in zip(questions, answers):
    #                 conv = get_default_conv_template(self.model_name).copy() #
    #                 if "qwen" in self.model_name:
    #                     user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}" #
    #                 else:
    #                     user_message = DEFAULT_IMAGE_TOKEN + "\n" + q #
    #                 conv.append_message(conv.roles[0], user_message) #
    #                 conv.append_message(conv.roles[1], a) #
    #                 conversations.append(conv.get_prompt()) #

    #             # --- Final SAM image preprocessing ---
    #             image_sam = self.preprocess(torch.from_numpy(image_sam_processed).permute(2, 0, 1).contiguous()) #

    #             # --- 准备 Mask 和 Label (始终生成) ---
    #             masks = np.stack(sampled_masks, axis=0) # sampled_masks 现在只有一个元素，shape [1, H, W]
    #             masks = torch.from_numpy(masks) #
    #             label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label #

    #             # --- 返回单目标样本 ---
    #             return (
    #                 image_path,
    #                 image_sam,
    #                 image_vlm,
    #                 conversations, # List 长度为 1
    #                 masks,         # Tensor shape [1, H, W]
    #                 label,
    #                 resize,
    #                 questions,     # List 长度为 1
    #                 sampled_sents, # List 长度为 1
    #             )
    #         except Exception as e:
    #             print(f"Error processing sample idx {idx}, image {image_path}: {e}. Retrying...")
    #             # 继续循环以重试

                
        ###################################################################原始代码，取只取单个目标进行分割################################################################
        # images, jsons = self.reason_seg_data
        # idx = random.randint(0, len(images) - 1)
        # image_path = images[idx]
        # json_path = jsons[idx]

        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ori_size = image.shape[:2]

        # # Preprocess image for VLM
        # if "qwen" in self.model_name:
        #     image_vlm = Image.fromarray(image)  # PIL Image for Qwen2.5-VL

        # mask, sents, is_sentence = get_mask_from_json(json_path, image)
        # if len(sents) >= self.num_classes_per_sample:
        #     sampled_inds = random.sample(range(len(sents)), self.num_classes_per_sample)
        # else:
        #     sampled_inds = list(range(len(sents)))
        # sampled_sents = [sents[i] for i in sampled_inds]
        # sampled_masks = [(mask == 1).astype(np.float32) for _ in range(len(sampled_inds))]

        # image = self.transform.apply_image(image) 
        # resize = image.shape[:2]

        # questions = []
        # answers = []
        # for text in sampled_sents:
        #     # 1. Create the question (standard for all samples)
        #     if is_sentence:
        #         question_template = random.choice(self.long_question_list)
        #         questions.append(question_template.format(sent=text))
        #     else:
        #         question_template = random.choice(self.short_question_list)
        #         questions.append(question_template.format(class_name=text.lower()))

        #     img_name = image_path.split("/")[-1]

        #     if self.explanatory != -1 and img_name in self.img_to_explanation:
        #         answers.append(self.img_to_explanation[img_name]["outputs"])
        #     else:
        #         answers.append(random.choice(self.answer_list))

        # conversations = []
        # for q, a in zip(questions, answers):
        #     conv = get_default_conv_template(self.model_name).copy()
        #     if "qwen" in self.model_name:
        #         user_message = f"<|vision_start|><|image_pad|><|vision_end>\n{q}"
        #     else:
        #         user_message = "<image>" + "\n" + q
        #     conv.append_message(conv.roles[0], user_message)
        #     conv.append_message(conv.roles[1], a)
        #     conversations.append(conv.get_prompt())

        # image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # masks = np.stack(sampled_masks, axis=0)
        # masks = torch.from_numpy(masks)
        # label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label


        # return (
        #     image_path,
        #     image,
        #     image_vlm,
        #     conversations,
        #     masks,
        #     label,
        #     resize,
        #     questions,
        #     sampled_sents,
        # )



####################################################把test加进去训练########################################################

# import glob
# import json
# import os
# import random

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image

# from .conversation import get_default_conv_template  
# from model.segment_anything.utils.transforms import ResizeLongestSide

# from .data_processing import get_mask_from_json
# from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
#                     EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
#                     SHORT_QUESTION_LIST)


# def get_mask_center(mask_np):
#     """
#     Calculate the centroid (mean center) of the foreground mask.
#     Ensures the returned point lies inside the mask.
#     """
#     mask_np = (mask_np > 0).astype(np.uint8)
#     if mask_np.sum() == 0:
#         return [0, 0]

#     M = cv2.moments(mask_np)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#     else:
#         ys, xs = np.where(mask_np)
#         cx, cy = int(np.mean(xs)), int(np.mean(ys))

#     # 保证点在mask内部
#     if mask_np[cy, cx] == 0:
#         ys, xs = np.where(mask_np)
#         idx = len(xs) // 2
#         cx, cy = int(xs[idx]), int(ys[idx])

#     return [cx, cy]


# class ReasonSegDataset(torch.utils.data.Dataset):
#     pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
#     img_size = 1024
#     ignore_label = 255

#     def __init__(
#         self,
#         base_image_dir,
#         tokenizer,
#         samples_per_epoch=500 * 8 * 2 * 10,
#         precision: str = "fp32",
#         image_size: int = 224,
#         num_classes_per_sample: int = 2,
#         exclude_val=False,
#         reason_seg_data="ReasonSeg|train",
#         explanatory=0.1,
#         model_name="qwen_vl", 
#     ):
#         """
#         Initialize the ReasonSegDataset with dataset-specific configurations.
#         """
#         self.exclude_val = exclude_val
#         self.samples_per_epoch = samples_per_epoch
#         self.explanatory = explanatory
#         self.num_classes_per_sample = num_classes_per_sample

#         self.base_image_dir = base_image_dir
#         self.image_size = image_size
#         self.tokenizer = tokenizer
#         self.precision = precision
#         self.transform = ResizeLongestSide(image_size)
#         self.model_name = model_name.lower()

#         self.short_question_list = SHORT_QUESTION_LIST
#         self.long_question_list = LONG_QUESTION_LIST
#         self.answer_list = ANSWER_LIST

#         # --- 1. 解析数据集 split 并强制加入 test ---
#         dataset_name, splits_str = reason_seg_data.split("|")
#         splits = splits_str.split("_")

#         # 如果包含 train 但不包含 test，强制添加 test
#         if "train" in splits and "test" not in splits:
#             splits.append("test")
#             print(f"--> ReasonSegDataset Info: 'test' split auto-added. Current splits: {splits}")

#         images = []
#         for split in splits:
#             split_path = os.path.join(base_image_dir, dataset_name, split, "*.jpg")
#             images_split = glob.glob(split_path)
#             print(f"Loaded {len(images_split)} images from split: {split}")
#             images.extend(images_split)

#         jsons = [path.replace(".jpg", ".json") for path in images]
#         self.reason_seg_data = (images, jsons)
        
#         # 保存 dataset_name 供后面加载 explanatory 使用
#         self.dataset_name = dataset_name 

#         print("Total number of reason_seg samples (combined): ", len(images))

#         # --- 2. 加载 Explanatory 数据 (支持 train 和 test) ---
#         if explanatory != -1:
#             self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
#             self.img_to_explanation = {}
            
#             # 定义需要加载的文件列表
#             # train对应 updated_train_w_point.json
#             # test 对应 test_w_point.json (如果存在)
#             files_to_check = []
            
#             # 根据当前加载的 split 决定加载哪些 json
#             if "train" in splits:
#                 files_to_check.append("updated_train_w_point.json")
#             if "test" in splits:
#                 files_to_check.append("test_w_point.json")
            
#             # 如果splits只有val或其他，默认不加载或者你可以根据需求添加

#             for file_name in files_to_check:
#                 file_path = os.path.join(
#                     base_image_dir,
#                     self.dataset_name,
#                     "explanatory",
#                     file_name,
#                 )
                
#                 if os.path.exists(file_path):
#                     print(f"Loading explanatory data from: {file_name}")
#                     try:
#                         with open(file_path, 'r') as f:
#                             items = json.load(f)
#                         for item in items:
#                             img_name = item["image"]
#                             self.img_to_explanation[img_name] = {
#                                 "query": item["query"],
#                                 "outputs": item["outputs"],
#                             }
#                     except Exception as e:
#                         print(f"Error reading {file_name}: {e}")
#                 else:
#                     print(f"Warning: Explanatory file not found: {file_path}. Ensure you have generated it if needed.")

#             print("Total number of explanatory samples loaded: ", len(self.img_to_explanation))

#     def __len__(self):
#         return self.samples_per_epoch

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input for SAM."""
#         x = (x - self.pixel_mean) / self.pixel_std
#         h, w = x.shape[-2:]
#         padh = self.img_size - h
#         padw = self.img_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x

#     def __getitem__(self, idx):
#         """Fetch a sample with dataset-specific logic."""

#         images, jsons = self.reason_seg_data
        
#         # Loop to ensure a valid sample is returned
#         while True:
#             try:
#                 idx = random.randint(0, len(images) - 1)
#                 image_path = images[idx]
#                 json_path = jsons[idx]

#                 image = cv2.imread(image_path)
#                 if image is None:
#                     print(f"Warning: Failed to load image {image_path}. Retrying...")
#                     continue
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
#                 # Preprocess image for VLM (Qwen expects PIL)
#                 if "qwen" in self.model_name:
#                     image_vlm = Image.fromarray(image)
#                 else:
#                     image_vlm = None 

#                 # --- 获取 mask 和 sentences ---
#                 mask, sents, is_sentence = get_mask_from_json(json_path, image)
#                 if not sents:
#                     print(f"Warning: No sentences found in {json_path}. Retrying...")
#                     continue

#                 # --- 强制单目标采样 ---
#                 sampled_ind = random.choice(range(len(sents))) 
#                 sampled_sents = [sents[sampled_ind]] 
                
#                 # --- 获取掩码并计算中心点 ---
#                 final_mask_np = (mask == 1).astype(np.uint8)
#                 sampled_masks = [final_mask_np.astype(np.float32)] 
#                 center_point = get_mask_center(final_mask_np)
#                 center_x, center_y = center_point

#                 # --- Image transformation for SAM ---
#                 image_sam_processed = self.transform.apply_image(image)
#                 resize = image_sam_processed.shape[:2]

#                 # --- 生成 Question 和 Answer ---
#                 questions = []
#                 answers = []

#                 for text in sampled_sents:
#                     img_name = image_path.split("/")[-1]
                    
#                     # 1. 创建 Question
#                     if is_sentence:
#                         question_template = random.choice(self.long_question_list)
#                         questions.append(question_template.format(sent=text))

#                         # 2.A (句子): 
#                         # 注意：这里依然注释了 img_to_explanation 的调用，避免 Key Error
#                         # 如果未来需要使用，请确保 test_w_point.json 已经生成并正确加载
#                         # answers.append(self.img_to_explanation.get(img_name, {}).get("outputs", ""))
                        
#                         answer_string = f"<think></think><answer>So the mask is [{center_x}, {center_y}].<SEG></answer>"
#                         answers.append(answer_string)

#                     else:
#                         question_template = random.choice(self.short_question_list)
#                         questions.append(question_template.format(class_name=text.lower()))
                        
#                         # 2.B (非句子): 
#                         answer_string = f"<think></think><answer>So the mask is [{center_x}, {center_y}].<SEG></answer>"
#                         answers.append(answer_string)

#                 # --- 生成 Conversation ---
#                 conversations = []
#                 for q, a in zip(questions, answers):
#                     conv = get_default_conv_template(self.model_name).copy()
#                     if "qwen" in self.model_name:
#                         user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}"
#                     else:
#                         user_message = DEFAULT_IMAGE_TOKEN + "\n" + q
#                     conv.append_message(conv.roles[0], user_message)
#                     conv.append_message(conv.roles[1], a)
#                     conversations.append(conv.get_prompt())

#                 # --- Final SAM image preprocessing ---
#                 image_sam = self.preprocess(torch.from_numpy(image_sam_processed).permute(2, 0, 1).contiguous())

#                 # --- 准备 Mask 和 Label ---
#                 masks = np.stack(sampled_masks, axis=0) # shape [1, H, W]
#                 masks = torch.from_numpy(masks)
#                 label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

#                 return (
#                     image_path,
#                     image_sam,
#                     image_vlm,
#                     conversations, 
#                     masks,         
#                     label,
#                     resize,
#                     questions,     
#                     sampled_sents, 
#                 )
#             except Exception as e:
#                 print(f"Error processing sample idx {idx}: {e}. Retrying...")
#                 continue