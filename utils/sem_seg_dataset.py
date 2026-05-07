import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from .conversation import get_default_conv_template  
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, MULTI_CLASS_QUESTION_LIST, INST_ANSWER_LIST


def get_mask_center(mask_np):
    mask_np = (mask_np > 0).astype(np.uint8)
    if mask_np.sum() == 0:
        return [0, 0]

    # 1. 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    # 2. 如果有多个连通域，找到面积最大的那个（忽略背景 label 0）
    # stats[:, 4] 是面积
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # 从1开始找最大
        
        # 只保留最大连通域的 mask
        mask_largest = (labels == largest_label).astype(np.uint8)
        
        # 计算最大连通域的矩
        M = cv2.moments(mask_largest)
    else:
        # 只有一个连通域（或者是空的）
        M = cv2.moments(mask_np)

    # 3. 计算重心
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        ys, xs = np.where(mask_np)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

    # 4. 最后一道防线：确保点在 mask 内部
    # 注意：这里检查的是原始 mask_np，只要在任意前景里都行
    if mask_np[cy, cx] == 0:
        ys, xs = np.where(mask_np)
        # 这种情况下通常是因为形状极其怪异（如很细的C型），回退到取中间点
        idx = len(xs) // 2
        cx, cy = int(xs[idx]), int(ys[idx])

    return [cx, cy]


def get_mask_points(mask_np, num_points=1):
    """
    从mask中获取固定数量的点。
    第一个点总是重心(center)，其余点从mask前景中随机采样。
    """
    num_points = int(num_points)
    # 1. 获取重心作为第一个点
    center = get_mask_center(mask_np)
    points = [center]

    if num_points <= 1:
        return points

    # 2. 获取Mask所有前景点的坐标 (x, y)
    ys, xs = np.where(mask_np > 0)
    
    # 如果mask为空，返回中心点（其实是[0,0]）重复num_points次
    if len(xs) == 0:
        return points * num_points

    all_coords = list(zip(xs, ys))
    
    # 3. 随机采样剩余的点
    num_to_sample = num_points - 1
    
    # 过滤掉中心点（防止重复），如果前景点太少就不强求过滤
    candidates = [p for p in all_coords if not (p[0] == center[0] and p[1] == center[1])]
    
    if len(candidates) >= num_to_sample:
        extras = random.sample(candidates, num_to_sample)
        points.extend(extras)
    else:
        # 如果前景点不够（mask极小），允许重复采样
        # 先把所有的 candidates 加进去
        points.extend(candidates)
        # 剩下的缺口从 all_coords 里随机补（包含重复）
        while len(points) < num_points:
            points.append(random.choice(all_coords))

    return points



# Initialization functions for each dataset

def init_mapillary(base_image_dir):
    """Initialize Mapillary dataset."""
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    """Initialize ADE20K dataset."""
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = [x[:-4] for x in image_ids if x.endswith(".jpg")]
    ade20k_images = [
        os.path.join(base_image_dir, "ade20k", "images", "training", f"{image_id}.jpg")
        for image_id in ade20k_image_ids
    ]
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    """Initialize COCO-Stuff dataset."""
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "COCO_Stuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("COCO_Stuff", "COCO")
        for x in cocostuff_labels
    ]
    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    """Initialize PACO-LVIS dataset."""
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    """Initialize Pascal-Part dataset."""
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        sem_seg_p=[1.0, 0.0, 0.0],
        model_name="qwen_vl",
        num_points=1,  # <--- 新增参数：固定采样点数量，默认为3
        use_SEG_token=True,
        normalize_coords=True, # <--- 新增参数：是否归一化坐标到 [0, 1000], Qwen3为True，2.5为False
    ):
        """
        Initialize the SemSegDataset with dataset-specific configurations.
        
        Args:
            base_image_dir (str): Base directory for dataset files.
            tokenizer: Tokenizer for text processing.
            samples_per_epoch (int): Number of samples per epoch.
            precision (str): Data precision ("fp32" or "fp16").
            image_size (int): Target image size for resizing.
            num_classes_per_sample (int): Number of classes to sample per image.
            exclude_val (bool): Whether to exclude validation data.
            sem_seg_data (str): Datasets to use, separated by "||".
            sem_seg_p (list of float): Probabilities for sampling 1, 2 or 3 classes.
            model_name (str): Model identifier ("llava" or "qwen_vl").
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.model_name = model_name.lower()
        self.sem_seg_p = sem_seg_p

        self.num_points = num_points            # 保存点数设置
        self.use_SEG_token = use_SEG_token
        self.normalize_coords = normalize_coords # 保存坐标归一化设置

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.multi_class_question_list = MULTI_CLASS_QUESTION_LIST
        self.inst_answer_list = INST_ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval(f"init_{ds}")(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }
        print("sem_seg_p: ", sem_seg_p)

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
    

    


#####################################################带有point的get_item###############################################
    def __getitem__(self, idx):
        """Fetch a sample with dataset-specific logic."""
        ds = random.choice(self.sem_seg_datas)
        
        sampled_classes = []
        sampled_points = []
        masks_list = []
        class_ids_list = []
        label_map = None 
        
        target_num_points = self.num_points 

        # ---------------------------------------------------------------------
        # Part 1: 数据加载与采样
        # ---------------------------------------------------------------------
        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join("VOCdevkit", "VOC2010", "JPEGImages", file_name)
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "COCO", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if not anns:
                return self.__getitem__(0)
            
            sampled_ann = random.choice(anns)
            sampled_cls_info = class_map[sampled_ann["category_id"]]
            if isinstance(sampled_cls_info, tuple):
                obj, part = sampled_cls_info
                name = random.choice([f"{obj} {part}", f"the {part} of the {obj}"])
            else:
                name = sampled_cls_info
            sampled_classes.append(name) 

            try:
                mask_np = coco_api.annToMask(sampled_ann)
                masks_list.append(mask_np)
                sampled_points = get_mask_points(mask_np, num_points=target_num_points)
            except Exception as e:
                print(e)
                return self.__getitem__(0)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image_list, label_list = self.data2list[ds]
            idx = random.randint(0, len(image_list) - 1)
            image_path = image_list[idx]
            label_path = label_list[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = np.array(Image.open(label_path))

            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255

            unique_label = [l for l in np.unique(label) if l != 255]
            if not unique_label:
                return self.__getitem__(0)
            
            classes = [self.data2classes[ds][l] for l in unique_label]
            sampled_cls = random.choice(classes)
            sampled_classes.append(sampled_cls)
            
            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids_list.append(class_id)
            mask_np = (label == class_id).astype(np.uint8)
            
            sampled_points = get_mask_points(mask_np, num_points=target_num_points)
            label_map = torch.from_numpy(label).long()

        if not sampled_classes:
            return self.__getitem__(0)

        # ---------------------------------------------------------------------
        # Part 2: 生成 Question 和 Answer
        # ---------------------------------------------------------------------
        questions = []
        answers = []
        sampled_cls = sampled_classes[0]

        question_template = random.choice(self.short_question_list)
        questions.append(question_template.format(class_name=sampled_cls.lower()))

        height, width = image.shape[:2]
        points_strs = []
        
        for pt in sampled_points:
            abs_x, abs_y = pt
            
            # --- 使用参数 self.normalize_coords 控制逻辑 ---
            if self.normalize_coords:
                # Qwen3 风格: 归一化到 [0, 1000]
                norm_x = int(round(abs_x / width * 1000))
                norm_y = int(round(abs_y / height * 1000))
                norm_x = max(0, min(1000, norm_x))
                norm_y = max(0, min(1000, norm_y))
                points_strs.append(f"[{norm_x}, {norm_y}]")
            else:
                # Qwen2.5 风格: 使用原始绝对坐标
                points_strs.append(f"[{abs_x}, {abs_y}]")
        
        points_string = ", ".join(points_strs)

        if self.use_SEG_token:
            answer_string = f"<think></think><answer>The mask is {points_string}.<SEG></answer>"
            # answer_string = "<SEG>"
        else:
            answer_string = f"<think></think><answer>The mask is {points_string}.</answer>"
            
        answers.append(answer_string)

        # ---------------------------------------------------------------------
        # Part 3: 构造 Conversation 和 Tensor
        # ---------------------------------------------------------------------
        conversations = []
        for q, a in zip(questions, answers):
            conv = get_default_conv_template(self.model_name).copy()
            if "qwen" in self.model_name:
                user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}"
            else:
                user_message = "<image>" + "\n" + q
            conv.append_message(conv.roles[0], user_message)
            conv.append_message(conv.roles[1], a)
            conversations.append(conv.get_prompt())

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image_sam = self.preprocess(image_tensor)
        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)  
        else:
            image_vlm = image_sam

        resize = image.shape[:2]

        if ds in ["paco_lvis", "pascal_part"]:
            masks = np.stack(masks_list, axis=0)
            masks = torch.from_numpy(masks)
            label_map = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        else:
            masks = torch.stack([label_map == cid for cid in class_ids_list], dim=0)
            
        return (
            image_path,
            image_sam,
            image_vlm,
            conversations,
            masks,
            label_map,
            resize,
            questions,
            sampled_classes,
        )

    ##########################################################不带有point################################################################
    # def __getitem__(self, idx):
    #     """Fetch a sample with dataset-specific logic."""
    #     ds = random.choice(self.sem_seg_datas)

    #     if ds in ["paco_lvis", "pascal_part"]:
    #         class_map = self.data2classes[ds]
    #         img_ids, coco_api = self.data2list[ds]
    #         idx = random.randint(0, len(img_ids) - 1)
    #         img_id = img_ids[idx]
    #         image_info = coco_api.loadImgs([img_id])[0]
    #         file_name = image_info["file_name"]
    #         if ds == "pascal_part":
    #             file_name = os.path.join("VOCdevkit", "VOC2010", "JPEGImages", file_name)
    #             image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
    #         elif ds == "paco_lvis":
    #             image_path = os.path.join(self.base_image_dir, "COCO", file_name)
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         annIds = coco_api.getAnnIds(imgIds=image_info["id"])
    #         anns = coco_api.loadAnns(annIds)
    #         if not anns: # 如果没有标注，重新采样
    #             return self.__getitem__(0)
            
    #         # --- 强制只采样一个标注 ---
    #         sampled_anns = random.sample(anns, 1) # 强制只选一个 ann
    #         sampled_classes = []
    #         for ann in sampled_anns:
    #             sampled_cls_info = class_map[ann["category_id"]]
    #             if isinstance(sampled_cls_info, tuple): # 处理部位标注 ('person', 'head')
    #                 obj, part = sampled_cls_info
    #                 # 随机选择一种表述方式
    #                 name = random.choice([f"{obj} {part}", f"the {part} of the {obj}"])
    #             else: # 处理普通物体标注
    #                 name = sampled_cls_info
    #             sampled_classes.append(name) # sampled_classes 现在只会有一个元素


    #     elif ds in ["ade20k", "cocostuff", "mapillary"]:
    #         image_list, label_list = self.data2list[ds]
    #         idx = random.randint(0, len(image_list) - 1)
    #         image_path = image_list[idx]
    #         label_path = label_list[idx]
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         label = np.array(Image.open(label_path))

    #         if ds == "ade20k":
    #             label[label == 0] = 255
    #             label -= 1
    #             label[label == 254] = 255
    #         elif ds == "cocostuff":
    #             for c, i in self.cocostuff_class2index.items():
    #                 if "-" in c:
    #                     label[label == i] = 255

    #         unique_label = [l for l in np.unique(label) if l != 255]
    #         if not unique_label:
    #             return self.__getitem__(0)
    #         classes = [self.data2classes[ds][l] for l in unique_label]
    #         # 强制只采样一个类别
    #         sampled_classes = random.sample(classes, 1)
    #         # sampled_classes = random.sample(classes, min(self.num_classes_per_sample, len(classes)))

    #     questions = []
    #     answers = []
    #     class_ids = []

    #     # 单目标分割采样的代码
    #     if sampled_classes: # 确保 sampled_classes 不为空
    #         sampled_cls = sampled_classes[0] # 获取唯一的类别/描述

    #         # --- 生成 Question ---
    #         question_template = random.choice(self.short_question_list) # 从单目标模板中选择
    #         questions.append(question_template.format(class_name=sampled_cls.lower()))

    #         # print(f"Question: {question_template.format(class_name=sampled_cls.lower())}")

    #         # --- 生成自定义 Answer ---
    #         sampled_cls_lower = sampled_cls.lower()
    #         answer_string = f"<think>The target is the {sampled_cls_lower}.</think><answer>Target Object: <pos><SEG></pos></answer>"
    #         # print(f"Aanswer: {answer_string}")
    #         answers.append(answer_string)

    #         # 记录类别 ID (如果适用)
    #         if ds not in ["paco_lvis", "pascal_part"]:
    #             class_id = self.data2classes[ds].tolist().index(sampled_cls) #
    #             class_ids.append(class_id)

    #     # 如果 sampled_classes 为空，则重新采样
    #     else:
    #         print(f"sampled_classes为空，重新采样")
    #         return self.__getitem__(0)
    #     # # 多类别的代码
    #     # i = 0
    #     # while i < len(sampled_classes):
    #     #     number = np.random.choice([1, 2, 3], p=self.sem_seg_p)
    #     #     number = min(len(sampled_classes) - i, number)

    #     #     if number == 1:
    #     #         sampled_cls = sampled_classes[i]
    #     #         question_template = random.choice(self.short_question_list)
    #     #         questions.append(question_template.format(class_name=sampled_cls.lower()))
    #     #         answers.append(random.choice(self.answer_list))
    #     #         if ds not in ["paco_lvis", "pascal_part"]:
    #     #             class_id = self.data2classes[ds].tolist().index(sampled_cls)
    #     #             class_ids.append(class_id)
    #     #     else:
    #     #         text = "the "
    #     #         for idx2, c in enumerate(sampled_classes[i:i + number]):
    #     #             text += c
    #     #             if idx2 < number - 2:
    #     #                 text += ", "
    #     #             elif idx2 == number - 2:
    #     #                 text += " and " if idx2 == 0 else ", and "
    #     #         question_template = random.choice(self.multi_class_question_list)
    #     #         questions.append(question_template.format(classes=text.lower()))

    #     #         seg_tokens = ""
    #     #         for idx2 in range(number):
    #     #             seg_tokens += "[SEG]"
    #     #             if idx2 < number - 2:
    #     #                 seg_tokens += ", "
    #     #             elif idx2 == number - 2:
    #     #                 seg_tokens += " and " if idx2 == 0 else ", and "
    #     #         answer_template = random.choice(self.inst_answer_list)
    #     #         answers.append(answer_template.format(seg_tokens=seg_tokens))
    #     #         if ds not in ["paco_lvis", "pascal_part"]:
    #     #             for c in sampled_classes[i:i + number]:
    #     #                 class_id = self.data2classes[ds].tolist().index(c)
    #     #                 class_ids.append(class_id)
    #     #     i += number

    #     conversations = []
    #     for q, a in zip(questions, answers):
    #         conv = get_default_conv_template(self.model_name).copy()
    #         if "qwen" in self.model_name:
    #             user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}"
    #         else:
    #             user_message = "<image>" + "\n" + q
    #         conv.append_message(conv.roles[0], user_message)
    #         conv.append_message(conv.roles[1], a)
    #         conversations.append(conv.get_prompt())

    #     image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    #     image_sam = self.preprocess(image_tensor)
    #     if "qwen" in self.model_name:
    #         image_vlm = Image.fromarray(image)  
    #     else:
    #         image_vlm = image_sam

    #     resize = image.shape[:2]

    #     if ds in ["paco_lvis", "pascal_part"]:
    #         masks = []
    #         for ann in sampled_anns:
    #             try:
    #                 masks.append(coco_api.annToMask(ann))
    #             except Exception as e:
    #                 print(e)
    #                 return self.__getitem__(0)
    #         masks = np.stack(masks, axis=0)
    #         masks = torch.from_numpy(masks)
    #         label_map = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
    #     else:
    #         label_map = torch.from_numpy(label).long()
    #         masks = torch.stack([label_map == cid for cid in class_ids], dim=0)
            
    #     return (
    #         image_path,
    #         image_sam,
    #         image_vlm,
    #         conversations,
    #         masks,
    #         label_map,
    #         resize,
    #         questions,
    #         sampled_classes,
    #     )


