import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from PIL import Image

from .conversation import get_default_conv_template  
from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

def get_mask_center(mask_np):
    """
    Calculate the centroid (mean center) of the foreground mask.
    Ensures the returned point lies inside the mask.
    """
    mask_np = (mask_np > 0).astype(np.uint8)
    if mask_np.sum() == 0:
        return [0, 0]

    M = cv2.moments(mask_np)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        ys, xs = np.where(mask_np)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

    # 保证点在mask内部
    if mask_np[cy, cx] == 0:
        ys, xs = np.where(mask_np)
        idx = len(xs) // 2
        cx, cy = int(xs[idx]), int(ys[idx])

    return [cx, cy]


def get_mask_points(mask_np, num_points=1):
    """
    从mask中获取固定数量的点。
    第一个点总是重心(center)，其余点从mask前景中随机采样。
    """
    center = get_mask_center(mask_np)
    points = [center]

    if num_points <= 1:
        return points

    ys, xs = np.where(mask_np > 0)
    
    # 如果mask为空，返回重复的中心点（通常是[0,0]）
    if len(xs) == 0:
        return points * num_points

    all_coords = list(zip(xs, ys))
    
    # 随机采样剩余的点
    num_to_sample = num_points - 1
    
    # 过滤掉中心点（防止重复），如果前景点太少就不强求过滤
    candidates = [p for p in all_coords if not (p[0] == center[0] and p[1] == center[1])]
    
    if len(candidates) >= num_to_sample:
        extras = random.sample(candidates, num_to_sample)
        points.extend(extras)
    else:
        # 如果前景点不够，先用 candidates 补，还不够就随机重复
        points.extend(candidates)
        while len(points) < num_points:
            points.append(random.choice(all_coords))

    return points


class ReferSegDataset(torch.utils.data.Dataset):
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
        num_classes_per_sample: int = 1,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        # refer_seg_data="refcoco||refcoco+||refcocog",
        model_name="qwen_vl",  
        num_points=1,           # <--- 新增：默认采样3个点
        use_SEG_token=True,
        normalize_coords=True, # <--- 新增：是否归一化坐标 (Qwen3需设为True)
    ):
        """
        Initialize the ReferSegDataset with dataset-specific configurations.

        Args:
            base_image_dir (str): Base directory for dataset files.
            tokenizer: Tokenizer for text processing.
            samples_per_epoch (int): Number of samples per epoch.
            precision (str): Data precision ("fp32" or "fp16").
            image_size (int): Target image size for resizing.
            num_classes_per_sample (int): Number of classes to sample per image.
            exclude_val (bool): Whether to exclude validation data.
            refer_seg_data (str): Datasets to use, separated by "||".
            model_name (str): Model name ("llava" or "qwen_vl").
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.model_name = model_name.lower()

        # 保存新参数
        self.num_points = num_points
        self.use_SEG_token = use_SEG_token
        self.normalize_coords = normalize_coords

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        DATA_DIR = base_image_dir
        self.refer_seg_ds_list = refer_seg_data.split("||")  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            splitBy = "umd" if ds == "refcocog" else "unc"
            refer_api = G_REFER(DATA_DIR, ds, splitBy) if ds == "grefcoco" else REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {
                "images": [],
                "annotations": refer_api.Anns,
                "img2refs": {}
            }

            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(self.base_image_dir, "refclef", "saiapr_tc-12", item["file_name"])
                else:
                    item["file_name"] = os.path.join(self.base_image_dir, "train2014", item["file_name"])
                refer_seg_ds["images"].append(item)

            for ref in refs_train:
                image_id = ref["image_id"]
                refer_seg_ds["img2refs"][image_id] = refer_seg_ds["img2refs"].get(image_id, []) + [ref]

            print(f"Dataset {ds} (refs {splitBy}) (train split) has {len(refer_seg_ds['images'])} images and {len(refer_seg_ds['annotations'])} annotations.")
            self.refer_seg_data[ds] = refer_seg_ds

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
        """Fetch a sample with dataset-specific logic."""
        ds = random.choice(self.refer_seg_ds_list)
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs.get(image_id) # 使用 .get 防止报错
        if not refs:
            return self.__getitem__(0)


        # 强制只采样一个指代表达
        # 1. 收集所有句子及其对应的 ann_id
        all_sents_anns = []
        for ref in refs:
            ann_id = ref["ann_id"] # 获取这个 ref 对应的 ann_id
            for sent in ref["sentences"]:
                all_sents_anns.append({'sent': sent["sent"], 'ann_id': ann_id})

        if not all_sents_anns: # 如果所有 ref 都没有句子，重新采样
             # print(f"Warning: Refs for image {image_id} in dataset {ds} have no sentences. Resampling.")
             return self.__getitem__(0)

        # 2. 从所有句子中随机选择一个
        sampled_choice = random.choice(all_sents_anns)
        sampled_sents = [sampled_choice['sent']] # 列表只包含一个句子
        sampled_ann_ids = [sampled_choice['ann_id']] # 列表只包含一个 ann_id
        sampled_classes = sampled_sents

        # --- 加载和预处理图像 ---
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}. Resampling.")
            return self.__getitem__(0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #

        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)  

        # image = self.transform.apply_image(image)
        # resize = image.shape[:2]
        image_sam_processed = self.transform.apply_image(image) # SAM 使用的大小调整
        resize = image_sam_processed.shape[:2] # 记录 resize 后的大小


        ######################################################加入point#############################################################
        masks_list = []
        ann_id = sampled_ann_ids[0] # 获取唯一的 ann_id
        final_mask_np = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)

        if isinstance(ann_id, list): 
            m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            for ann_id_i in ann_id:
                ann = annotations.get(ann_id_i)
                if ann is None or not ann.get("segmentation"): 
                     print(f"Warning: Annotation {ann_id_i} not found or invalid for image {image_id}. Using zero mask part.")
                     m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    if isinstance(ann["segmentation"][0], list):  # polygon
                        rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                    else: # RLE
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode() 
                    m = mask.decode(rle) 
                    if m.ndim > 2: 
                            m = np.sum(m, axis=2).astype(np.uint8) 
                    else:
                            m = m.astype(np.uint8)
                m_final = m_final | m 
            final_mask_np = m_final
        else: 
            ann = annotations.get(ann_id) 
            if ann is None or not ann.get("segmentation"):
                 print(f"Warning: Annotation {ann_id} not found or invalid for image {image_id}. Using zero mask.")
                 m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            else:
                if isinstance(ann["segmentation"][0], list):  # polygon
                    rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"]) 
                else: # RLE
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode() 
                m = mask.decode(rle) 
                if m.ndim > 2: 
                        m = np.sum(m, axis=2).astype(np.uint8) 
                else:
                        m = m.astype(np.uint8)
            final_mask_np = m
        
        masks_list.append(final_mask_np)

        if final_mask_np.sum() == 0: # 简单的检查 mask 是否有效
             return self.__getitem__(0)

        # --- *MODIFIED*: 计算多个点并根据参数归一化 ---
        
        # 1. 获取点列表
        sampled_points = get_mask_points(final_mask_np, num_points=self.num_points)

        # 2. 准备 Question
        questions = []
        answers = []
        sampled_sent = sampled_sents[0]
        question_template = random.choice(self.short_question_list) 
        questions.append(question_template.format(class_name=sampled_sent.lower()))

        # 3. 处理坐标和生成 Answer
        height, width = image.shape[:2]
        points_strs = []

        for pt in sampled_points:
            abs_x, abs_y = pt
            
            # 根据 self.normalize_coords 决定处理方式
            if self.normalize_coords:
                # Qwen3 风格: 归一化到 [0, 1000]
                norm_x = int(round(abs_x / width * 1000))
                norm_y = int(round(abs_y / height * 1000))
                norm_x = max(0, min(1000, norm_x))
                norm_y = max(0, min(1000, norm_y))
                points_strs.append(f"[{norm_x}, {norm_y}]")
            else:
                # Qwen2.5 风格: 原始绝对坐标
                points_strs.append(f"[{abs_x}, {abs_y}]")

        points_string = ", ".join(points_strs)
        if self.use_SEG_token:
            answer_string = f"<think></think><answer>The mask is {points_string}.<SEG></answer>"
            # answer_string = "<SEG>"
        else:
            answer_string = f"<think></think><answer>The mask is {points_string}.</answer>"
        answers.append(answer_string)

        # --- 生成 Conversations ---
        conversations = []
        q = questions[0]
        a = answers[0]
        conv = get_default_conv_template(self.model_name).copy() 
        if "qwen" in self.model_name:
            user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}" 
        else:
            user_message = "<image>" + "\n" + q 
        conv.append_message(conv.roles[0], user_message) 
        conv.append_message(conv.roles[1], a) 
        conversations.append(conv.get_prompt()) 

        # --- SAM 输入图像预处理 ---
        image_sam = self.preprocess(torch.from_numpy(image_sam_processed).permute(2, 0, 1).contiguous()) 
        
        # 如果不是 qwen，在这里赋值 image_vlm (对应前面可能的 else 分支)
        if "qwen" not in self.model_name:
             image_vlm = image_sam

        # --- 准备 Masks ---
        masks = torch.from_numpy(np.stack(masks_list, axis=0)) 
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label 

        return (
            image_path,
            image_sam,
            image_vlm,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )

            ################################################采用固定的格式################################################
        # # --- 生成 Question 和自定义 Answer ---
        # questions = []
        # answers = []

        # sampled_sent = sampled_sents[0]
        # # Question: 使用 SHORT_QUESTION_LIST 模板，将 {class_name} 替换为指代语句
        # question_template = random.choice(self.short_question_list) #
        # # 注意：这里用完整的指代语句替换了原先的 class_name
        # questions.append(question_template.format(class_name=sampled_sent.lower()))

        # # print(f"Question: {question_template.format(class_name=sampled_sent.lower())}")

        # # Answer: 使用你的自定义模板
        # sampled_sent_lower = sampled_sent.lower()
        # answer_string = f"<think>The target is the {sampled_sent_lower}.</think><answer>Target Object: <pos><SEG></pos></answer>"
        # # print(f"Answer: {answer_string}")
        # answers.append(answer_string)

        # # --- 生成 Conversations ---
        # conversations = []
        # # 因为只有一个 Q&A 对
        # q = questions[0]
        # a = answers[0]
        # conv = get_default_conv_template(self.model_name).copy() #
        # # 格式化用户消息
        # if "qwen" in self.model_name:
        #     user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}" #
        # else:
        #     user_message = "<image>" + "\n" + q #
        # conv.append_message(conv.roles[0], user_message) #
        # conv.append_message(conv.roles[1], a) #
        # conversations.append(conv.get_prompt()) #

        # # --- SAM 输入图像预处理 ---
        # image_sam = self.preprocess(torch.from_numpy(image_sam_processed).permute(2, 0, 1).contiguous()) # 标准化 + Padding

        # # --- 准备 Masks ---
        # masks = []
        # ann_id = sampled_ann_ids[0] # 获取唯一的 ann_id

        # # --- 加载单个掩码，逻辑与原代码类似但只处理一个 ann_id ---
        # if isinstance(ann_id, list): # 处理 ann_id 是列表的情况
        #     m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #     for ann_id_i in ann_id:
        #         ann = annotations.get(ann_id_i)
        #         if ann is None or not ann.get("segmentation"): # 检查标注是否存在且有效
        #              # 如果 ann_id 列表中的某个标注无效，可以选择跳过或填充零掩码
        #              print(f"Warning: Annotation {ann_id_i} not found or invalid for image {image_id}. Using zero mask part.")
        #              m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #         else:
        #             # try:
        #             if isinstance(ann["segmentation"][0], list):  # polygon
        #                 rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
        #             else: # RLE
        #                 rle = ann["segmentation"]
        #                 # 确保 counts 是 bytes
        #                 for i in range(len(rle)):
        #                     if not isinstance(rle[i]["counts"], bytes):
        #                         rle[i]["counts"] = rle[i]["counts"].encode() #
        #             m = mask.decode(rle) #
        #             if m.ndim > 2: # 处理可能的额外维度
        #                     m = np.sum(m, axis=2).astype(np.uint8) #
        #             else:
        #                     m = m.astype(np.uint8)
        #             # except Exception as e:
        #             #      print(f"Error decoding mask for ann_id {ann_id_i}, image {image_id}: {e}. Using zero mask part.")
        #             #      m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #         m_final = m_final | m # 合并掩码
        #     masks.append(m_final)
        # else: # 处理 ann_id 是单个 ID 的情况
        #     ann = annotations.get(ann_id) # 使用 .get()
        #     if ann is None or not ann.get("segmentation"):
        #          print(f"Warning: Annotation {ann_id} not found or invalid for image {image_id}. Using zero mask.")
        #          m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #     else:
        #         # try:
        #         if isinstance(ann["segmentation"][0], list):  # polygon
        #             rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"]) #
        #         else: # RLE
        #             rle = ann["segmentation"]
        #             # 确保 counts 是 bytes
        #             for i in range(len(rle)):
        #                 if not isinstance(rle[i]["counts"], bytes):
        #                     rle[i]["counts"] = rle[i]["counts"].encode() #
        #         m = mask.decode(rle) #
        #         if m.ndim > 2: # 处理可能的额外维度
        #                 m = np.sum(m, axis=2).astype(np.uint8) #
        #         else:
        #                 m = m.astype(np.uint8)
        #         # except Exception as e:
        #         #      print(f"Error decoding mask for ann_id {ann_id}, image {image_id}: {e}. Using zero mask.")
        #         #      m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #     masks.append(m)
        # # --- 掩码加载结束 ---

        # # 确保至少有一个掩码被添加，否则重新采样
        # if not masks:
        #     print(f"Warning: No valid masks generated for image {image_id}. Resampling.")
        #     return self.__getitem__(0)

        # masks = torch.from_numpy(np.stack(masks, axis=0)) #
        # label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label # 生成忽略标签图

        # return (
        #     image_path,
        #     image_sam,
        #     image_vlm,
        #     conversations,
        #     masks,
        #     label,
        #     resize,
        #     questions,
        #     sampled_classes,
        # )


        ##########################################采用随机的question和answer###############################################
        # questions = [random.choice(self.short_question_list).format(class_name=c.lower()) for c in sampled_classes]
        # answers = [random.choice(self.answer_list) for _ in sampled_classes]

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

        # masks = []
        # for ann_id in sampled_ann_ids:
        #     if isinstance(ann_id, list):
        #         m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #         for ann_id_i in ann_id:
        #             ann = annotations[ann_id_i]
        #             if not ann["segmentation"]:
        #                 m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #             else:
        #                 if isinstance(ann["segmentation"][0], list):  # polygon
        #                     rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
        #                 else:
        #                     rle = ann["segmentation"]
        #                     for i in range(len(rle)):
        #                         if not isinstance(rle[i]["counts"], bytes):
        #                             rle[i]["counts"] = rle[i]["counts"].encode()
        #                 m = mask.decode(rle)
        #                 m = np.sum(m, axis=2).astype(np.uint8)
        #             m_final = m_final | m
        #         masks.append(m_final)
        #     else:
        #         ann = annotations[ann_id]
        #         if not ann["segmentation"]:
        #             m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        #         else:
        #             if isinstance(ann["segmentation"][0], list):  # polygon
        #                 rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
        #             else:
        #                 rle = ann["segmentation"]
        #                 for i in range(len(rle)):
        #                     if not isinstance(rle[i]["counts"], bytes):
        #                         rle[i]["counts"] = rle[i]["counts"].encode()
        #             m = mask.decode(rle)
        #             m = np.sum(m, axis=2).astype(np.uint8)
        #         masks.append(m)

        # masks = torch.from_numpy(np.stack(masks, axis=0))
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
        #     sampled_classes,
        # )


#         )