import logging
from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from model.segment_anything import build_sam_vit_h
from scipy.optimize import linear_sum_assignment
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Initializing model...")


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
    inputs = inputs.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class VlmSamSegModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(VlmSamSegModel, self).__init__()
        self.config = config

        if not kwargs["train_mask_decoder"]:
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_vlmSamSeg_modules(self.config, kwargs)

    def initialize_vlmSamSeg_modules(self, config, kwargs):

        # Build SAM visual model
        self.visual_model = build_sam_vit_h(self.vision_pretrained).to(kwargs["torch_dtype"])

        ########################################把text当做point处理#############################################
        # # 初始化text_embeddings为point_embeddings的值
        # if hasattr(self.visual_model.prompt_encoder, 'text_embeddings'):
        #     print("Copying pre-trained point_embeddings to text_embeddings...")
        #     with torch.no_grad():
        #         # 将 负 text_embed (索引0) 初始化为 负 point_embed (索引0)
        #         self.visual_model.prompt_encoder.text_embeddings[0].weight.data.copy_(
        #             self.visual_model.prompt_encoder.point_embeddings[0].weight.data
        #         )
        #         # 将 正 text_embed (索引1) 初始化为 正 point_embed (索引1)
        #         self.visual_model.prompt_encoder.text_embeddings[1].weight.data.copy_(
        #             self.visual_model.prompt_encoder.point_embeddings[1].weight.data
        #         )
        # else:
        #     print("WARNING: Could not find custom text_embeddings to initialize.")
        
        ########################################把text当做point处理#############################################
        
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if kwargs["train_mask_decoder"]:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        
        # ===== 修改这里 =====
        # Qwen3VL 使用复合配置，需要从 text_config 中获取 hidden_size
        if hasattr(config, 'text_config'):
            text_config = config.text_config
            if hasattr(text_config, 'hidden_size'):
                in_dim = text_config.hidden_size
            elif hasattr(text_config, 'd_model'):
                in_dim = text_config.d_model
            else:
                raise ValueError("Cannot find hidden_size in text_config")
        elif hasattr(config, 'hidden_size'):
            in_dim = config.hidden_size
        else:
            raise ValueError("Cannot determine hidden size from config")
        
        print(f"Using in_dim from config: {in_dim}")
        # ===== 修改结束 =====

        # Projection layer for text embeddings
        # ######################################Qwen2.5-VL的hidden_size######################################
        # in_dim = config.hidden_size
        out_dim = kwargs["out_dim"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class VlmSamSegForCausalLM(nn.Module):
    def __init__(self, config, **kwargs):
        super(VlmSamSegForCausalLM, self).__init__()
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.neg_seg_token_idx = kwargs.pop("neg_seg_token_idx")
        self.use_SEG_token = kwargs.pop("use_SEG_token")
        # self.vlm_point_mode = "qwen3"   # or "qwen25"

        tokenizer_len = kwargs.pop("tokenizer_len", None)

        # self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     kwargs["model"],
        #     torch_dtype=kwargs["torch_dtype"],
        #     attn_implementation=kwargs["attention"]
        # )

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            kwargs["model"],
            torch_dtype=kwargs["torch_dtype"],
            attn_implementation=kwargs["attention"]
        )

        # 更新词表长度
        if tokenizer_len is not None:
            self.vlm.resize_token_embeddings(tokenizer_len)
            print(f"Resized VLM token embeddings to {tokenizer_len}")

        self.model_vlmSamSeg = VlmSamSegModel(config, **kwargs).to(kwargs["torch_dtype"])
        self.processor = AutoProcessor.from_pretrained(kwargs["model"])
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

        # 编译正则表达式
        self.point_regex = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
        # 捕获负样本点块
        self.neg_block_regex = re.compile(r"Negative objects are:\s*(.*)", re.IGNORECASE)

    @staticmethod
    def adjust_indices_order(pred_indices: np.ndarray, gt_indices: np.ndarray):
        adjusted_gt_indices = np.empty_like(gt_indices)
        sorted_pred_indices = np.argsort(pred_indices)
        for i, sorted_idx in enumerate(sorted_pred_indices):
            adjusted_gt_indices[i] = gt_indices[sorted_idx]
        return np.arange(len(pred_indices)), adjusted_gt_indices

    def hungarian_matcher(self, pred_masks: List[torch.Tensor], gt_masks: List[torch.Tensor]):
        pred_masks = torch.stack([m.squeeze(0) for m in pred_masks]).flatten(1)
        gt_masks = torch.stack([m.squeeze(0) for m in gt_masks]).flatten(1)
        dice_loss_cur = batch_dice_loss(pred_masks, gt_masks)
        sigmoid_ce_loss_cur = batch_sigmoid_ce_loss(pred_masks, gt_masks)
        cost_matrix = dice_loss_cur + sigmoid_ce_loss_cur
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.detach().cpu())
        adjust_pred_indices, adjust_gt_indices = self.adjust_indices_order(pred_indices, gt_indices)
        return adjust_pred_indices, adjust_gt_indices

    def hungarian_matcher_batch(self, pred_masks: List[List[torch.Tensor]], gt_masks: List[List[torch.Tensor]], change_list: List[List[int]]):
        reordered_gt_masks = []
        for batch_idx, groups in enumerate(change_list):
            batch_pred_masks = pred_masks[batch_idx]
            batch_gt_masks = gt_masks[batch_idx]
            reordered_batch_gt_masks = batch_gt_masks.clone()
            for group in groups:
                group_pred_masks = batch_pred_masks[group].unsqueeze(1).flatten(1)
                group_gt_masks = batch_gt_masks[group].unsqueeze(1).flatten(1)
                _, group_gt_indices = self.hungarian_matcher(group_pred_masks, group_gt_masks)
                for idx, gt_idx in enumerate(group_gt_indices):
                    reordered_batch_gt_masks[group[idx]] = batch_gt_masks[group[gt_idx]]
            reordered_gt_masks.append(reordered_batch_gt_masks)
        return reordered_gt_masks


    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        
        CHUNK_SIZE = 2 
        
        embeddings_list = []
        B = pixel_values.shape[0]
        
        with torch.no_grad():
            for i in range(0, B, CHUNK_SIZE):
                # 1. 切片
                chunk = pixel_values[i : i + CHUNK_SIZE]
                
                # 2. 编码这一小块
                chunk_emb = self.model_vlmSamSeg.visual_model.image_encoder(chunk)
                embeddings_list.append(chunk_emb)
                
                # 3. 清理不需要的中间变量 (可选)
                del chunk
                # torch.cuda.empty_cache() # 如果显存极度紧张可以打开，但会稍微变慢

        # 4. 拼接回 (B, 256, 64, 64)
        return torch.cat(embeddings_list, dim=0)
    

    def _points_from_answer_region(
        self,
        text_region: str,
        resize_hw: tuple,
        original_hw: tuple,
        mode: str = "qwen3",
        device=None,
    ):
        """
        Parse points from a string region using self.point_regex.findall(region).

        Returns:
            pts_tensor: (N,2) in SAM input pixel coords, or None if no points.
        Coordinate modes:
        - qwen3: points are normalized in [0,1000], map to resized_w/h from resize_list[i]
        - qwen25: points are in original pixel coords, scale to SAM target_size=1024 based on long side
        """
        matches = self.point_regex.findall(text_region)
        if len(matches) == 0:
            return None

        if device is None:
            device = next(self.parameters()).device

        pts = []
        m = (mode or "qwen3").lower()

        if m in ["qwen3", "qwen-3", "qwen3-vl", "qwen_3"]:
            resized_h, resized_w = resize_hw
            for x_str, y_str in matches:
                x_norm, y_norm = float(x_str), float(y_str)  # [0,1000]
                x_abs = (x_norm / 1000.0) * float(resized_w)
                y_abs = (y_norm / 1000.0) * float(resized_h)
                pts.append(torch.tensor([x_abs, y_abs], device=device))
        else:
            # Qwen2.5-VL: treat (x,y) as original pixel coords, then scale to target_size=1024 by long side
            H_orig, W_orig = original_hw
            target_size = 1024.0
            original_long_side = float(max(H_orig, W_orig))
            scale = target_size / original_long_side

            for x_str, y_str in matches:
                x, y = float(x_str), float(y_str)
                pts.append(torch.tensor([x * scale, y * scale], device=device))

        return torch.stack(pts, dim=0)  # (N,2)



    # 修改这里采取不同的方式进行训练
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        
        if self.use_SEG_token:
            # 用SEG token和positive, negative point一起处理
            return self.model_forward_with_pos_neg_points(**kwargs)
        else:
            # 消融实验，只使用point进行处理
            return self.model_forward_with_pos_points_only(**kwargs)

        # # 正常SEG处理
        # return self.model_forward(**kwargs)

        # # 用SEG token和pos point一起处理
        # return self.model_forward_with_pos_points(**kwargs)

        # # # # 用SEG token和positive, negative point一起处理
        # return self.model_forward_with_pos_neg_points(**kwargs)

        # # 消融实验，只使用point进行处理
        # return self.model_forward_with_pos_points_only(**kwargs)

        # # 消融实验：只用 SEG token，不用 point
        # return self.model_forward_seg_only(**kwargs)

        # 把text当做point处理
        # return self.model_forward_new(**kwargs)
    
    # --- 用SEG token和pos point一起处理 ---
    def model_forward_with_pos_neg_points(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list,
        label_list,
        resize_list,
        conversation_list,
        inference: bool = False,
        change_list=[],
        **kwargs
    ):
        # -------------------------
        # 1) SAM visual embeddings (保持不变)
        # -------------------------
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        # -------------------------
        # 2) VLM forward (保持不变)
        # -------------------------
        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )

        output_hidden_states = output.hidden_states[-2].detach().requires_grad_(True)

        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        last_hidden_state = self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states)

        # -------------------------
        # 3) Get <SEG> token embedding (保持不变)
        # -------------------------
        seg_token_mask = (input_ids == self.seg_token_idx)
        first_seg_token_indices = torch.argmax(seg_token_mask.int(), dim=1) 
        idx_gather = first_seg_token_indices.unsqueeze(-1).unsqueeze(-1)
        idx_gather = idx_gather.expand(-1, 1, last_hidden_state.shape[-1])
        seg_embeddings = torch.gather(last_hidden_state, dim=1, index=idx_gather).squeeze(1)

        seg_embeddings_list = []
        for i in range(batch_size):
            s, e = offset[i].item(), offset[i + 1].item()
            seg_embeddings_list.append(seg_embeddings[s:e])

        # -------------------------
        # 4) Helper: Point Parsing (支持解析多个点)
        # -------------------------
        import re
        device = images.device
        
        def parse_points_from_text(text_segment, resize_hw, original_hw, device):
            # 正则匹配所有 [x, y]
            pattern = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
            matches = re.findall(pattern, text_segment)
            
            if not matches:
                return None
            
            points = []
            h_new, w_new = resize_hw
            h_orig, w_orig = original_hw
            
            for x_str, y_str in matches:
                x, y = float(x_str), float(y_str)
                # 假设 Qwen3 归一化 (0-1000)
                if getattr(self, "vlm_point_mode", "qwen3") == "qwen3":
                    x = x / 1000.0 * w_new
                    y = y / 1000.0 * h_new
                else:
                    x = x * (w_new / w_orig)
                    y = y * (h_new / h_orig)
                points.append([x, y])
            
            return torch.tensor(points, device=device, dtype=torch.float32)

        # -------------------------
        # 5) Parse Points & Prepare Labels (合并 Pos 和 Neg)
        # -------------------------
        points_per_image = []
        labels_per_image = []
        
        # 临时列表用于 Padding 计算
        unpadded_pts_list = []
        unpadded_lbl_list = []
        max_points_in_batch = 1 # 整个 Batch 中点最多的数量

        for i in range(batch_size):
            start, end = offset[i].item(), offset[i + 1].item()

            pts_list_for_image = []
            lbl_list_for_image = []
            
            # 注意：如果一个图有多个 mask (start -> end > 1)，这里会处理每个 mask
            for j in range(start, end):
                full_answer = conversation_list[j].split("<answer>")[-1].split("</answer>")[0]
                clean_answer = full_answer.replace("<SEG>", "")
                
                # ==================== 【修改开始】自动判断顺序逻辑 ====================
                mask_marker = "The mask is"
                inter_marker = "The interference is"
                
                mask_idx = clean_answer.find(mask_marker)
                inter_idx = clean_answer.find(inter_marker)
                
                pos_text = ""
                neg_text = ""
                
                # 情况 1: 两个标记都存在 (关键逻辑：比较 index 大小)
                if mask_idx != -1 and inter_idx != -1:
                    if mask_idx < inter_idx:
                        # Mask 在前: ... The mask is [...] ... The interference is [...]
                        # 截取 mask_marker 之后，inter_marker 之前
                        pos_text = clean_answer[mask_idx + len(mask_marker):inter_idx]
                        # 截取 inter_marker 之后
                        neg_text = clean_answer[inter_idx + len(inter_marker):]
                    else:
                        # Interference 在前: ... The interference is [...] ... The mask is [...]
                        # 截取 inter_marker 之后，mask_marker 之前
                        neg_text = clean_answer[inter_idx + len(inter_marker):mask_idx]
                        # 截取 mask_marker 之后
                        pos_text = clean_answer[mask_idx + len(mask_marker):]
                        
                # 情况 2: 只有 Mask 标记
                elif mask_idx != -1:
                    pos_text = clean_answer[mask_idx + len(mask_marker):]
                    
                # 情况 3: 只有 Interference 标记 (虽然少见)
                elif inter_idx != -1:
                    neg_text = clean_answer[inter_idx + len(inter_marker):]
                    
                # 情况 4: 都没有，默认全是正样本 (兼容旧数据或纯Mask数据)
                else:
                    pos_text = clean_answer
                # ==================== 【修改结束】 ====================

                # B. 解析坐标
                pos_pts_tensor = parse_points_from_text(pos_text, resize_list[i], label_list[i].shape, device)
                neg_pts_tensor = parse_points_from_text(neg_text, resize_list[i], label_list[i].shape, device)

                # C. 构造当前样本的点和标签列表
                current_sample_pts = []
                current_sample_lbls = []

                # --- 处理 Positive Points (Label = 1) ---
                if pos_pts_tensor is not None:
                    current_sample_pts.append(pos_pts_tensor)
                    current_sample_lbls.append(torch.ones(len(pos_pts_tensor), device=device, dtype=torch.long))
                
                # --- 处理 Negative Points (Label = 0) ---
                if neg_pts_tensor is not None:
                    current_sample_pts.append(neg_pts_tensor)
                    current_sample_lbls.append(torch.zeros(len(neg_pts_tensor), device=device, dtype=torch.long))

                # D. 合并 & 异常处理
                if len(current_sample_pts) > 0:
                    # 将正点和负点拼接成一个 Tensor
                    combined_pts = torch.cat(current_sample_pts, dim=0)
                    combined_lbls = torch.cat(current_sample_lbls, dim=0)
                else:
                    # 极其罕见情况：既没正点也没负点 -> 给一个 Dummy 点，Label -1 (Ignore)
                    combined_pts = torch.tensor([[0.0, 0.0]], device=device)
                    combined_lbls = torch.tensor([-1], device=device, dtype=torch.long)

                pts_list_for_image.append(combined_pts)
                lbl_list_for_image.append(combined_lbls)

                # 更新最大长度以便 Padding
                max_points_in_batch = max(max_points_in_batch, combined_pts.shape[0])

            unpadded_pts_list.append(pts_list_for_image)
            unpadded_lbl_list.append(lbl_list_for_image)

        # -------------------------
        # 6) Padding Logic (统一 Pad 到 max_points_in_batch)
        # -------------------------
        for i in range(batch_size):
            padded_pts_image = []
            padded_lbl_image = []
            
            pts_list = unpadded_pts_list[i]
            lbl_list = unpadded_lbl_list[i]
            
            for pts, lbls in zip(pts_list, lbl_list):
                num_to_pad = max_points_in_batch - pts.shape[0]
                if num_to_pad > 0:
                    # Pad points with (0,0)
                    pad_p = torch.zeros((num_to_pad, 2), device=device, dtype=pts.dtype)
                    pts = torch.cat([pts, pad_p], dim=0)
                    
                    # Pad labels with -1 (Ignore)
                    pad_l = torch.full((num_to_pad,), -1, device=device, dtype=lbls.dtype)
                    lbls = torch.cat([lbls, pad_l], dim=0)
                
                padded_pts_image.append(pts)
                padded_lbl_image.append(lbls)
            
            # Stack: (N_masks_in_image, max_points_in_batch, 2)
            points_per_image.append(torch.stack(padded_pts_image, dim=0))
            labels_per_image.append(torch.stack(padded_lbl_image, dim=0))

        # -------------------------
        # 7) Predict Masks (单分支结构)
        # -------------------------
        pred_masks = []
        for i, seg_embeds in enumerate(seg_embeddings_list):
            if seg_embeds.numel() == 0:
                pred_masks.append(torch.empty(0, *label_list[i].shape, device=device))
                continue
            
            # 直接把包含了正(1)和负(0)的 points/labels 传进去
            # text_embeds 直接使用 SEG token embedding
            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=(points_per_image[i], labels_per_image[i]),
                boxes=None,
                masks=None,
                text_embeds=seg_embeds.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(image_embeddings.dtype)

            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
        
        if inference:
            # 构造一个 dummy loss 防止报错，device 取 images 的 device
            dummy_loss = torch.tensor(0.0, device=images.device, requires_grad=False)
            
            return {
                "pred_masks": pred_masks,  # 预测出的 masks
                "gt_masks": masks_list,    # 把输入的 GT 传回去，方便评测脚本计算 IoU
                "loss": dummy_loss,
                "ce_loss": dummy_loss,
                "mask_bce_loss": dummy_loss,
                "mask_dice_loss": dummy_loss,
                "mask_loss": dummy_loss,
            }

        # -------------------------
        # 8) Loss Calculation (保持不变)
        # -------------------------
        gt_masks = masks_list
        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0.0
        mask_dice_loss = 0.0
        num_masks = 0.0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            if gt_mask.shape[0] != pred_mask.shape[0]:
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }


    # --- 用SEG token和pos point一起处理 ---
    def model_forward_with_pos_points(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list,
        label_list,
        resize_list,
        conversation_list,
        inference: bool = False,
        change_list=[],
        **kwargs
    ):
        # -------------------------
        # 1) SAM visual embeddings
        # -------------------------
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        # -------------------------
        # 2) VLM forward
        # -------------------------
        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )

        # keep grad on text projector input (same style as your other forward funcs)
        output_hidden_states = output.hidden_states[-2].detach().requires_grad_(True)

        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        last_hidden_state = self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states)

        # -------------------------
        # 3) take FIRST <SEG> token embedding per conversation
        # -------------------------
        seg_token_mask = (input_ids == self.seg_token_idx)
        first_seg_token_indices = torch.argmax(seg_token_mask.int(), dim=1)  # (total_convs,)
        idx_gather = first_seg_token_indices.unsqueeze(-1).unsqueeze(-1)
        idx_gather = idx_gather.expand(-1, 1, last_hidden_state.shape[-1])
        pred_embeddings = torch.gather(last_hidden_state, dim=1, index=idx_gather).squeeze(1)  # (total_convs,C)

        pred_embeddings_list = []
        for i in range(batch_size):
            s, e = offset[i].item(), offset[i + 1].item()
            pred_embeddings_list.append(pred_embeddings[s:e])  # (N_masks_i,C)

        # -------------------------
        # 4) parse MULTI positive points (pad to maxP per image)
        # -------------------------
        device = images.device
        mode = getattr(self, "vlm_point_mode", "qwen3")

        points_per_image = []
        points_labels_per_image = []

        for i in range(batch_size):
            start, end = offset[i].item(), offset[i + 1].item()

            pts_list_for_image = []
            lbl_list_for_image = []
            max_points_in_image = 1

            for j in range(start, end):
                answer = conversation_list[j].split("<answer>")[-1]

                # If there is a negative block, only take points BEFORE it as positive points
                neg_block_match = self.neg_block_regex.search(answer)
                pos_region = answer if neg_block_match is None else answer[:neg_block_match.start()]

                pts_tensor = self._points_from_answer_region(
                    pos_region,
                    resize_hw=resize_list[i],
                    original_hw=label_list[i].shape,
                    mode=mode,
                    device=device,
                )

                if pts_tensor is None:
                    # no pos points -> ignore by label -1
                    pts_list_for_image.append(torch.tensor([[0.0, 0.0]], device=device))
                    lbl_list_for_image.append(torch.tensor([-1], device=device, dtype=torch.long))
                    max_points_in_image = max(max_points_in_image, 1)
                else:
                    pts_list_for_image.append(pts_tensor)  # (Npos,2)
                    lbl_list_for_image.append(torch.ones((pts_tensor.shape[0],), device=device, dtype=torch.long))
                    max_points_in_image = max(max_points_in_image, pts_tensor.shape[0])

            # pad to max_points_in_image
            padded_pts, padded_lbls = [], []
            for pts, lbls in zip(pts_list_for_image, lbl_list_for_image):
                num_to_pad = max_points_in_image - pts.shape[0]
                if num_to_pad > 0:
                    pad_pts = torch.zeros((num_to_pad, 2), device=device, dtype=pts.dtype)
                    pts = torch.cat([pts, pad_pts], dim=0)

                    pad_lbls = torch.full((num_to_pad,), -1, device=device, dtype=lbls.dtype)
                    lbls = torch.cat([lbls, pad_lbls], dim=0)

                padded_pts.append(pts)
                padded_lbls.append(lbls)

            points_per_image.append(torch.stack(padded_pts, dim=0))          # (N_masks,maxP,2)
            points_labels_per_image.append(torch.stack(padded_lbls, dim=0))  # (N_masks,maxP)

        # -------------------------
        # 5) predict masks: points + SEG token
        # -------------------------
        pred_masks = []
        for i, text_embeds in enumerate(pred_embeddings_list):
            if text_embeds.numel() == 0:
                pred_masks.append(torch.empty(0, *label_list[i].shape, device=device))
                continue

            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=(points_per_image[i], points_labels_per_image[i]),
                boxes=None,
                masks=None,
                text_embeds=text_embeds.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)

            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        # -------------------------
        # 6) loss (same style)
        # -------------------------
        gt_masks = masks_list

        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0.0
        mask_dice_loss = 0.0
        num_masks = 0.0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            if gt_mask.shape[0] != pred_mask.shape[0]:
                logger.warning(f"Batch {batch_idx}: gt_mask.shape[0] != pred_mask.shape[0]")
                num_masks += 1
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }




    def model_forward_seg_only(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        change_list: List[torch.Tensor] = [],
        **kwargs
    ):
        """
        Ablation: only use <SEG> token embeddings to prompt SAM.
        No points / boxes / masks prompts.
        """

        # === 1) SAM image embeddings ===
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        # === 2) VLM forward (CE loss + hidden states) ===
        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )

        # 取倒数第二层 hidden states 做投影
        output_hidden_states = output.hidden_states[-2]
        output_hidden_states = output_hidden_states.detach().requires_grad_(True)

        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        last_hidden_state = self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states)

        # === 3) extract <SEG> token embeddings ===
        seg_token_mask = input_ids == self.seg_token_idx
        pred_embeddings_all = last_hidden_state[seg_token_mask]  # (total_seg_tokens, C)

        # 每个样本里 seg 的数量 -> offset 切片
        seg_token_counts = seg_token_mask.int().sum(-1)          # (total_convs,)
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, device=seg_token_offset.device, dtype=torch.long), seg_token_offset],
            dim=0
        )
        seg_token_offset = seg_token_offset[offset]              # (batch+1,)

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            s, e = seg_token_offset[i].item(), seg_token_offset[i + 1].item()
            pred_embeddings_list.append(pred_embeddings_all[s:e])  # (num_seg_i, C)

        # === 4) SAM mask prediction (NO points) ===
        pred_masks = []
        for i, text_embeds in enumerate(pred_embeddings_list):
            if text_embeds.numel() == 0:
                pred_masks.append(torch.empty(0, *label_list[i].shape, device=images.device))
                continue

            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=text_embeds.unsqueeze(1),  # (num_seg_i, 1, C)
            )
            sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)

            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        # === 5) loss (same as your other branches) ===
        gt_masks = masks_list
        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0.0
        mask_dice_loss = 0.0
        num_masks = 0.0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if gt_mask.shape[0] != pred_mask.shape[0]:
                logger.warning(f"Batch {batch_idx}: gt_mask.shape[0] != pred_mask.shape[0]")
                num_masks += 1
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }



    def model_forward_with_pos_points_only(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        conversation_list: List[str],  # 必须传入对话文本
        inference: bool = False,
        change_list: List[torch.Tensor] = [],
        **kwargs
    ):
        """
        [修改版]：
        1. 仅使用 Point (Positive + Negative) 进行分割
        2. 移除了 SEG token (text_embeds=None)
        3. 增加了对 Negative Point (Label 0) 的支持
        """
        
        # === 1. 提取图像特征 ===
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        # === 2. VLM forward（仅用于计算CE loss，不提取SEG token）===
        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        
        # 【注意】这里我们不再提取 hidden_states 和 SEG token embedding

        # === 3. 辅助函数：解析点坐标 ===
        # 直接使用你第二段代码中的解析逻辑
        device = images.device
        
        def parse_points_from_text(text_segment, resize_hw, original_hw, device):
            # 正则匹配所有 [x, y]
            pattern = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
            matches = re.findall(pattern, text_segment)
            
            if not matches:
                return None
            
            points = []
            h_new, w_new = resize_hw
            h_orig, w_orig = original_hw
            
            for x_str, y_str in matches:
                x, y = float(x_str), float(y_str)
                # 假设 Qwen3 归一化 (0-1000)
                # 如果你的 scale 逻辑不同，请保留你原本的 scale 方式
                if getattr(self, "vlm_point_mode", "qwen3") == "qwen3":
                    x = x / 1000.0 * w_new
                    y = y / 1000.0 * h_new
                else:
                    x = x * (w_new / w_orig)
                    y = y * (h_new / h_orig)
                points.append([x, y])
            
            return torch.tensor(points, device=device, dtype=torch.float32)

        # === 4. 解析 Positive / Negative 点并构建 Batch ===
        points_per_image = []
        labels_per_image = []
        
        # 临时列表用于 Padding 计算
        unpadded_pts_list = []
        unpadded_lbl_list = []
        max_points_in_batch = 1 

        for i in range(batch_size):
            start, end = offset[i].item(), offset[i + 1].item()
            pts_list_for_image = []
            lbl_list_for_image = []
            
            # 这里的 resized_h/w 用于坐标转换
            resized_h, resized_w = resize_list[i]

            for j in range(start, end):
                full_answer = conversation_list[j].split("<answer>")[-1].split("</answer>")[0]
                clean_answer = full_answer.replace("<SEG>", "")

                # ==================== 【修改开始】自动判断顺序逻辑 ====================
                mask_marker = "The mask is"
                inter_marker = "The interference is"
                
                mask_idx = clean_answer.find(mask_marker)
                inter_idx = clean_answer.find(inter_marker)
                
                pos_text = ""
                neg_text = ""
                
                # 情况 1: 两个标记都存在 (关键逻辑：比较 index 大小)
                if mask_idx != -1 and inter_idx != -1:
                    if mask_idx < inter_idx:
                        # Mask 在前: ... The mask is [...] ... The interference is [...]
                        # 截取 mask_marker 之后，inter_marker 之前
                        pos_text = clean_answer[mask_idx + len(mask_marker):inter_idx]
                        # 截取 inter_marker 之后
                        neg_text = clean_answer[inter_idx + len(inter_marker):]
                    else:
                        # Interference 在前: ... The interference is [...] ... The mask is [...]
                        # 截取 inter_marker 之后，mask_marker 之前
                        neg_text = clean_answer[inter_idx + len(inter_marker):mask_idx]
                        # 截取 mask_marker 之后
                        pos_text = clean_answer[mask_idx + len(mask_marker):]
                        
                # 情况 2: 只有 Mask 标记
                elif mask_idx != -1:
                    pos_text = clean_answer[mask_idx + len(mask_marker):]
                    
                # 情况 3: 只有 Interference 标记 (虽然少见)
                elif inter_idx != -1:
                    neg_text = clean_answer[inter_idx + len(inter_marker):]
                    
                # 情况 4: 都没有，默认全是正样本 (兼容旧数据或纯Mask数据)
                else:
                    pos_text = clean_answer
                # ==================== 【修改结束】 ====================
                
                # # ==================== interference在后 ====================
                # # # A. 分割正负文本区域
                # # # 数据格式: The interference is [...]. The mask is [...].
                # # if "The interference is" in clean_answer:
                # #     parts = clean_answer.split("The interference is")
                # #     pos_text = parts[0]
                # #     neg_text = parts[1]
                # # else:
                # #     pos_text = clean_answer
                # #     neg_text = ""
                # # ==================== interference在后 ====================


                # # ==================== interference在前 ====================
                # # A. 分割正负文本区域
                # # 数据格式: The interference is [...]. The mask is [...].
                # if "The mask is" in clean_answer:
                #     parts = clean_answer.split("The mask is")
                #     # parts[0] 在 "The mask is" 之前 -> Interference (Neg)
                #     neg_text = parts[0] 
                #     # parts[1] 在 "The mask is" 之后 -> Mask (Pos)
                #     pos_text = parts[1]
                # else:
                #     # 兼容情况：如果没有 Interference，可能只有 Mask 数据
                #     pos_text = clean_answer
                #     neg_text = ""
                # # ==================== interference在前 ====================

                # --- B. 解析坐标 ---
                # 注意：传入正确的 resize_list[i]
                pos_pts_tensor = parse_points_from_text(pos_text, resize_list[i], label_list[i].shape, device)
                neg_pts_tensor = parse_points_from_text(neg_text, resize_list[i], label_list[i].shape, device)

                # --- C. 构建当前样本的 Points 和 Labels ---
                current_sample_pts = []
                current_sample_lbls = []

                # Positive Points -> Label 1
                if pos_pts_tensor is not None:
                    current_sample_pts.append(pos_pts_tensor)
                    current_sample_lbls.append(torch.ones(len(pos_pts_tensor), device=device, dtype=torch.long))
                
                # Negative Points -> Label 0 (这是原本代码缺失的部分)
                if neg_pts_tensor is not None:
                    current_sample_pts.append(neg_pts_tensor)
                    current_sample_lbls.append(torch.zeros(len(neg_pts_tensor), device=device, dtype=torch.long))

                # 合并
                if len(current_sample_pts) > 0:
                    combined_pts = torch.cat(current_sample_pts, dim=0)
                    combined_lbls = torch.cat(current_sample_lbls, dim=0)
                else:
                    # 兜底：如果没有点，给一个 (0,0) 并标记为 ignore (-1)
                    combined_pts = torch.tensor([[0.0, 0.0]], device=device)
                    combined_lbls = torch.tensor([-1], device=device, dtype=torch.long)

                pts_list_for_image.append(combined_pts)
                lbl_list_for_image.append(combined_lbls)
                
                max_points_in_batch = max(max_points_in_batch, combined_pts.shape[0])

            unpadded_pts_list.append(pts_list_for_image)
            unpadded_lbl_list.append(lbl_list_for_image)

        # === 5. Padding Logic (必须做，否则 SAM 无法批处理) ===
        for i in range(batch_size):
            padded_pts_image = []
            padded_lbl_image = []
            
            pts_list = unpadded_pts_list[i]
            lbl_list = unpadded_lbl_list[i]
            
            for pts, lbls in zip(pts_list, lbl_list):
                num_to_pad = max_points_in_batch - pts.shape[0]
                if num_to_pad > 0:
                    pad_p = torch.zeros((num_to_pad, 2), device=device, dtype=pts.dtype)
                    pts = torch.cat([pts, pad_p], dim=0)
                    
                    pad_l = torch.full((num_to_pad,), -1, device=device, dtype=lbls.dtype)
                    lbls = torch.cat([lbls, pad_l], dim=0)
                
                padded_pts_image.append(pts)
                padded_lbl_image.append(lbls)
            
            points_per_image.append(torch.stack(padded_pts_image, dim=0))
            labels_per_image.append(torch.stack(padded_lbl_image, dim=0))

        # === 6. Mask 预测 (核心修改处) ===
        pred_masks = []
        for i in range(batch_size):
            if points_per_image[i].numel() == 0:
                pred_masks.append(torch.empty(0, *label_list[i].shape, device=device))
                continue

            # 调用 Prompt Encoder
            # points 包含了 Pos(1) 和 Neg(0)
            # text_embeds 显式设置为 None，实现 "No SEG Token"
            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=(points_per_image[i], labels_per_image[i]),
                boxes=None,
                masks=None,
                text_embeds=None,  # <--- 关键点：不使用文本 Embedding
            )
            
            sparse_embeddings = sparse_embeddings.to(image_embeddings.dtype)
            
            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
        
        if inference:
            # 构造一个 dummy loss 防止报错，device 取 images 的 device
            dummy_loss = torch.tensor(0.0, device=images.device, requires_grad=False)
            
            return {
                "pred_masks": pred_masks,  # 预测出的 masks
                "gt_masks": masks_list,    # 把输入的 GT 传回去，方便评测脚本计算 IoU
                "loss": dummy_loss,
                "ce_loss": dummy_loss,
                "mask_bce_loss": dummy_loss,
                "mask_dice_loss": dummy_loss,
                "mask_loss": dummy_loss,
            }

        # === 7. 损失计算 (保持不变) ===
        # ... (这部分代码与原来完全一致，省略以节省空间) ...
        # 复制原本的 Loss 计算代码即可
        
        gt_masks = masks_list
        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        logits = output.logits
        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if gt_mask.shape[0] != pred_mask.shape[0]:
                num_masks += 1
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }




    # SEG token
    def model_forward(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        change_list: List[torch.Tensor] = [],
        **kwargs
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )


        output_hidden_states = output.hidden_states[-2]
        # .requires_grad_(True) 告诉PyTorch，我们需要从这个点开始计算新的梯度
        output_hidden_states = output_hidden_states.detach().requires_grad_(True)
        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = []
        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)
        # print(f"seg_token_counts: {seg_token_counts}")
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_offset.device).long(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_list

        pred_masks = []
        for i, text_embeds in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=text_embeds.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)
            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list
        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        logits = output.logits
        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            # # 添加这行打印来验证
            print(f"Batch {batch_idx}: gt_mask shape: {gt_mask.shape}, pred_mask shape: {pred_mask.shape}")
            if gt_mask.shape[0] != pred_mask.shape[0]:
                print(f"Wrong! gt_mask.shape[0] != pred_mask.shape[0]")
                num_masks += 1
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "loss": total_loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    
    # SEG token 和 neg_SEG token
    def model_forward_new(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        change_list: List[torch.Tensor] = [],
        **kwargs
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )


        output_hidden_states = output.hidden_states[-2]
        # .requires_grad_(True) 告诉PyTorch，我们需要从这个点开始计算新的梯度
        output_hidden_states = output_hidden_states.detach().requires_grad_(True)

        # 获取两个token的掩码
        seg_token_mask = input_ids == self.seg_token_idx
        neg_seg_token_mask = input_ids == self.neg_seg_token_idx

        # 合并掩码以查找所有相关 token
        combined_seg_mask = seg_token_mask | neg_seg_token_mask

        hidden_states = []
        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        # 提取所有相关 token 的嵌入
        pred_embeddings_all = last_hidden_state[combined_seg_mask] # Shape: (total_all_seg_tokens, C)

        # 4. 为所有提取的 token 创建标签 (0 或 1)
        # 创建一个-1的标签张量
        labels_tensor = torch.full_like(input_ids, -1)
        labels_tensor[seg_token_mask] = 1     # 正 token 标签为 1
        labels_tensor[neg_seg_token_mask] = 0 # 负 token 标签为 0

        # 使用组合掩码提取标签
        pred_labels_all = labels_tensor[combined_seg_mask] # Shape: (total_all_seg_tokens)

        # 5. 更新 token 计数和偏移量，以计算所有 token (正 + 负)
        seg_token_counts = combined_seg_mask.int().sum(-1) # (修改为使用 combined_seg_mask)
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_offset.device).long(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]

        # 6. 按批次拆分嵌入和标签
        pred_embeddings_list = []
        pred_labels_list = [] # 新增
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings_all[start_i:end_i])
            pred_labels_list.append(pred_labels_all[start_i:end_i]) # 新增

        pred_masks = []
        for i in range(len(pred_embeddings_list)): # 按批次索引循环

            text_embeds = pred_embeddings_list[i]   # Shape: (N_i, C) -> N_i 是此样本中的 token 总数
            text_labels = pred_labels_list[i]   # Shape: (N_i)
            
            # 保持与原始代码一致的形状 (B, N, C)，其中 B=N_i, N=1
            text_embeds = text_embeds.unsqueeze(1)  # Shape: (N_i, 1, C)
            text_labels = text_labels.unsqueeze(1)  # Shape: (N_i, 1)
            print(f"text_labels: {text_labels}")

            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=text_embeds,
                text_labels=text_labels, # <-- 传递新标签
            )

            sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)
            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list
        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        logits = output.logits
        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            # # 添加这行打印来验证
            print(f"Batch {batch_idx}: gt_mask shape: {gt_mask.shape}, pred_mask shape: {pred_mask.shape}")
            if gt_mask.shape[0] != pred_mask.shape[0]:
                print(f"Wrong! gt_mask.shape[0] != pred_mask.shape[0]")
                num_masks += 1
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "loss": total_loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        inputs_qwen: dict,
        image_classical: torch.FloatTensor,
        resize_list: List[tuple],
        original_size_list: List[tuple],
        max_new_tokens: int = 2048,
        tokenizer=None
    ):
        with torch.no_grad():
            outputs = self.vlm.generate(
                **inputs_qwen,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            output_ids = outputs.sequences
            # decoded = self.processor.batch_decode(
            #     output_ids,
            #     skip_special_tokens=False,
            #     clean_up_tokenization_spaces=False
            # )
            # logger.info("Decoded Output: %s", decoded)

            output_hidden_states = outputs.hidden_states[-1][-2]
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx

            hidden_states = []
            assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
            hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)


            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

            pred_embeddings_list = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_list.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_list

            image_embeddings = self.get_visual_embs(image_classical)
            pred_masks = []
            for i, text_embeds in enumerate(pred_embeddings):
                sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=text_embeds.unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)
                low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        torch.cuda.empty_cache()
        return output_ids, pred_masks
