import argparse
import os
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import subprocess

try:
    import wandb
except ImportError:
    wandb = None

from model.vlmsam import VlmSamSegForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args):
    parser = argparse.ArgumentParser(description="VlmSamSeg Model Training with DDP and Resumption")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Pretrained Qwen3-VL HuggingFace model id or local path")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=32768, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dataset", default="sem_seg||refer_seg||ReasonSeg||overlap_reasonseg", type=str,
                        help="Datasets to train on, joined by '||'")
    parser.add_argument("--sample_rates", default="9,9,1,3", type=str,
                        help="Sampling rate per dataset, comma-separated, must match --dataset length")
    parser.add_argument("--sem_seg_data", default="ade20k||cocostuff", type=str)
    parser.add_argument("--refer_seg_data", default="refcoco||refcoco+||refcocog", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--cot_data", default="caption||cot||conversation", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str,
                        help="Root directory containing all datasets (ade20k, COCO, refcoco, etc.)")
    parser.add_argument("--overlap_json_path", default="", type=str,
                        help="Path to MechSeg-Bench train JSON; required when 'overlap_reasonseg' is in --dataset")
    parser.add_argument("--log_base_dir", default="./runs/psrs_main", type=str)
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--num_points", default=1, type=int,
                        help="Number of points predicted per object (for ablation)")
    parser.add_argument("--use_SEG_token", default=True, type=lambda x: str(x).lower() == "true",
                        help="Use the <SEG> token (True) or only points (False)")
    parser.add_argument("--exp_name", default="vlmsamseg", type=str)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--steps_per_epoch", default=5000, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accumulation_steps", default=5, type=int)
    parser.add_argument("--val_batch_size", default=4, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.5, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=1, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="./weights/sam_vit_h_4b8939.pth", type=str,
                        help="Path to SAM ViT-H pretrained weights")
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch (overridden if resuming)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--val_freq", default=10000000, type=int, help="Validation frequency in optimization steps")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging (requires `wandb login` or WANDB_API_KEY env var)")
    return parser.parse_args(args)

def setup_ddp():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return torch.device(f'cuda:{rank}'), world_size, rank

# def setup_ddp():
#     if 'SLURM_PROCID' in os.environ:
#         node_list = os.environ['SLURM_NODELIST']
#         master_addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
#         os.environ['MASTER_ADDR'] = master_addr
#         os.environ['MASTER_PORT'] = '29500'
#         os.environ['RANK'] = os.environ['SLURM_PROCID']
#         os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
#         os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']

#     dist.init_process_group(backend='nccl')
    
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     # ！！！你缺的就是下面这部分！！！
#     local_rank = int(os.environ.get('LOCAL_RANK', 0))
#     device = torch.device(f"cuda:{local_rank}")
#     torch.cuda.set_device(device)
    
#     return device, world_size, rank


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, cur_ciou, args, rank):
    if rank == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "cur_ciou": cur_ciou,
        }
        checkpoint_path = os.path.join(args.log_base_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    dist.barrier()  # Synchronize all processes after saving

# Validation Function
def validate(val_loader, model, global_optim_step, writer, args, rank, device):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    val_loss_meter = AverageMeter("ValLoss", ":.4f")
    ce_loss_meter = AverageMeter("CeLoss", ":.4f")
    mask_bce_loss_meter = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_loss_meter = AverageMeter("MaskDICELoss", ":.4f")
    mask_loss_meter = AverageMeter("MaskLoss", ":.4f")

    model.eval()
    with torch.no_grad():
        for input_dict in val_loader:
            input_dict = dict_to_cuda(input_dict)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32):
                output_dict = model(**input_dict)
                loss = output_dict["loss"].item()
            val_loss_meter.update(loss, input_dict["images"].size(0))
            ce_loss_meter.update(output_dict["ce_loss"].item(), input_dict["images"].size(0))
            # 删掉 .item() 即可
            mask_bce_loss_meter.update(output_dict["mask_bce_loss"], input_dict["images"].size(0))
            mask_dice_loss_meter.update(output_dict["mask_dice_loss"].item(), input_dict["images"].size(0))
            mask_loss_meter.update(output_dict["mask_loss"].item(), input_dict["images"].size(0))

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()

            intersection, union, acc_iou_sum = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                iou_i = intersection_i / (union_i + 1e-5)
                iou_i[union_i == 0] = 1.0  
                acc_iou_sum += iou_i

            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            acc_iou_meter.update(acc_iou_sum.cpu().numpy(), n=masks_list.shape[0])

    intersection_sum = torch.tensor(intersection_meter.sum, device=device)
    union_sum = torch.tensor(union_meter.sum, device=device)
    dist.all_reduce(intersection_sum)
    dist.all_reduce(union_sum)
    intersection_meter.sum = intersection_sum.cpu().numpy()
    union_meter.sum = union_sum.cpu().numpy()

    acc_iou_sum_total = torch.tensor(acc_iou_meter.sum, device=device)
    acc_iou_count_total = torch.tensor(acc_iou_meter.count, device=device)
    dist.all_reduce(acc_iou_sum_total)
    dist.all_reduce(acc_iou_count_total)

    # Loss meters
    val_loss_sum = torch.tensor(val_loss_meter.sum, device=device)
    val_loss_count = torch.tensor(val_loss_meter.count, device=device)
    dist.all_reduce(val_loss_sum)
    dist.all_reduce(val_loss_count)
    val_loss_avg = val_loss_sum / val_loss_count if val_loss_count > 0 else 0

    ce_loss_sum = torch.tensor(ce_loss_meter.sum, device=device)
    ce_loss_count = torch.tensor(ce_loss_meter.count, device=device)
    dist.all_reduce(ce_loss_sum)
    dist.all_reduce(ce_loss_count)
    ce_loss_avg = ce_loss_sum / ce_loss_count if ce_loss_count > 0 else 0

    mask_bce_loss_sum = torch.tensor(mask_bce_loss_meter.sum, device=device)
    mask_bce_loss_count = torch.tensor(mask_bce_loss_meter.count, device=device)
    dist.all_reduce(mask_bce_loss_sum)
    dist.all_reduce(mask_bce_loss_count)
    mask_bce_loss_avg = mask_bce_loss_sum / mask_bce_loss_count if mask_bce_loss_count > 0 else 0

    mask_dice_loss_sum = torch.tensor(mask_dice_loss_meter.sum, device=device)
    mask_dice_loss_count = torch.tensor(mask_dice_loss_meter.count, device=device)
    dist.all_reduce(mask_dice_loss_sum)
    dist.all_reduce(mask_dice_loss_count)
    mask_dice_loss_avg = mask_dice_loss_sum / mask_dice_loss_count if mask_dice_loss_count > 0 else 0

    mask_loss_sum = torch.tensor(mask_loss_meter.sum, device=device)
    mask_loss_count = torch.tensor(mask_loss_meter.count, device=device)
    dist.all_reduce(mask_loss_sum)
    dist.all_reduce(mask_loss_count)
    mask_loss_avg = mask_loss_sum / mask_loss_count if mask_loss_count > 0 else 0

    # ADD THESE DEBUG LINES
    if rank == 0:
        print("------------------- DEBUG INFO -------------------")
        print(f"intersection_meter.sum type: {type(intersection_meter.sum)}")
        print(f"intersection_meter.sum shape: {getattr(intersection_meter.sum, 'shape', 'N/A')}")
        print(f"intersection_meter.sum value: {intersection_meter.sum}")
        print(f"union_meter.sum type: {type(union_meter.sum)}")
        print(f"union_meter.sum shape: {getattr(union_meter.sum, 'shape', 'N/A')}")
        print(f"union_meter.sum value: {union_meter.sum}")
        print("--------------------------------------------------")

    # Compute final metrics
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = (acc_iou_sum_total / acc_iou_count_total)[1].item() if acc_iou_count_total > 0 else 0

    if rank == 0:
        print(f"global_optim_step: {global_optim_step}, giou: {giou:.4f}, ciou: {ciou:.4f}, val_loss: {val_loss_avg:.4f}")
        if wandb is not None and args.use_wandb:
            wandb.log({
                "val/loss": val_loss_avg,
                "val/ce_loss": ce_loss_avg,
                "val/mask_bce_loss": mask_bce_loss_avg,
                "val/mask_dice_loss": mask_dice_loss_avg,
                "val/mask_loss": mask_loss_avg,
                "val/giou": giou,
                "val/ciou": ciou,
                "global_optim_step": global_optim_step
            })
        if writer:
            writer.add_scalar("val/loss", val_loss_avg, global_optim_step)
            writer.add_scalar("val/giou", giou, global_optim_step)
            writer.add_scalar("val/ciou", ciou, global_optim_step)

    return giou, ciou

def train(train_loader, val_loader, model, tokenizer, optimizer, scheduler, epoch, writer, args, rank, best_score, cur_ciou, device):
    optim_steps_per_epoch = args.steps_per_epoch // args.grad_accumulation_steps
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        optim_steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses, mask_bce_losses, mask_dice_losses],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    end = time.time()
    optim_step = 0
    for step, input_dict in enumerate(train_loader):

        # ================================================================
        #
        # ### 新增的终极调试代码块 ###
        #
        # 在主进程(rank 0)打印每一步加载到的图像路径，以观察DataLoader的行为
        if rank == 0:
            # 使用 .get() 以防 "image_paths" 键不存在
            current_paths = input_dict.get("image_paths", ["'image_paths' key not found in input_dict."])
            print(f"[DEBUG DATALOADER] Step: {step}, Image Paths in Batch: {current_paths}")
        #
        # ================================================================

        if step >= args.steps_per_epoch:
            break
        input_dict = dict_to_cuda(input_dict)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32):
            output_dict = model(**input_dict)
            loss = output_dict["loss"] / args.grad_accumulation_steps
        loss.backward()
        

        if (step + 1) % args.grad_accumulation_steps == 0:

            if rank == 0:
                print("\n" + "="*60)
                print(f"--- Inspecting text_hidden_fcs @ Global Step {epoch * optim_steps_per_epoch + optim_step} ---")
                try:
                    # 通过 model.module 访问DDP包装下的原始模型
                    # text_hidden_fcs 是一个 ModuleList, 我们检查其中的第一个 Sequential 模块
                    mlp_module = model.module.model_vlmSamSeg.text_hidden_fcs[0]
                    
                    # 以MLP的第一个线性层为例
                    first_linear_layer = mlp_module[0]
                    second_linear_layer = mlp_module[2]
                    
                    # 1. 检查权重 (Weights) 本身
                    if first_linear_layer.weight is not None:
                        weights = first_linear_layer.weight.data
                        print(f"  Weights (Layer 1): "
                                f"mean={weights.mean():.8f}, std={weights.std():.8f}, "
                                f"max={weights.max():.8f}, min={weights.min():.8f}")

                    # 2. 检查梯度 (Gradients) - 这是诊断梯度爆炸最关键的部分
                    if first_linear_layer.weight.grad is not None:
                        grads = first_linear_layer.weight.grad.data
                        # 计算梯度的L2范数 (grad_norm)，这是衡量梯度大小最常用的指标
                        grad_norm = torch.linalg.norm(grads)
                        print(f"  GRADS (Layer 1):   "
                                f"mean={grads.mean():.8f}, std={grads.std():.8f}, "
                                f"max={grads.max():.8f}, min={grads.min():.8f}, "
                                f"norm={grad_norm:.8f}")
                    else:
                        print("  GRADS (Layer 1): No gradients found for this layer.")

                    # 1. 检查权重 (Weights) 本身
                    if second_linear_layer.weight is not None:
                        weights = second_linear_layer.weight.data
                        print(f"  Weights (Layer 2): "
                                f"mean={weights.mean():.8f}, std={weights.std():.8f}, "
                                f"max={weights.max():.8f}, min={weights.min():.8f}")

                    # 2. 检查梯度 (Gradients) - 这是诊断梯度爆炸最关键的部分
                    if second_linear_layer.weight.grad is not None:
                        grads = second_linear_layer.weight.grad.data
                        # 计算梯度的L2范数 (grad_norm)，这是衡量梯度大小最常用的指标
                        grad_norm = torch.linalg.norm(grads)
                        print(f"  GRADS (Layer 2):   "
                                f"mean={grads.mean():.8f}, std={grads.std():.8f}, "
                                f"max={grads.max():.8f}, min={grads.min():.8f}, "
                                f"norm={grad_norm:.8f}")
                    else:
                        print("  GRADS (Layer 2): No gradients found for this layer.")

                except Exception as e:
                    print(f"  Could not inspect parameters: {e}")
                print("="*60 + "\n")


            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            optim_step += 1
            global_optim_step = epoch * optim_steps_per_epoch + optim_step

            # ### START: 新增的保存Mask图像代码块 ###
            #
            # 只在主进程(rank 0)且每隔50步执行一次
            if rank == 0 and global_optim_step % 200 == 0:
                print(f"\n--- Saving mask visualizations @ Step {global_optim_step} ---")
                
                vis_dir = os.path.join(args.log_base_dir, "training_vis")
                os.makedirs(vis_dir, exist_ok=True)
                
                try:
                    pred_masks_batch = output_dict["pred_masks"]
                    gt_masks_batch = output_dict["gt_masks"]
                    image_paths_batch = input_dict["image_paths"]
                    
                    # ### 新增循环: 遍历批次中的每一个样本 ###
                    # 为了避免保存过多图片，这里我们只保存最多2个样本
                    num_samples_to_save = min(len(image_paths_batch), 2) 
                    
                    for i in range(num_samples_to_save):
                        image_path = image_paths_batch[i]
                        print(f"  Visualizing image path: {image_path}") # <-- 添加这行
                        original_image = cv2.imread(image_path)
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                        # 处理并保存第 i 个样本的第一个预测mask
                        if pred_masks_batch and len(pred_masks_batch[i]) > 0:
                            pred_mask_tensor = pred_masks_batch[i][0] # [H, W]
                            pred_mask_np = (pred_mask_tensor > 0).detach().cpu().numpy()
                            
                            gt_mask_tensor = gt_masks_batch[i][0] # [H, W] or [1, H, W]
                            gt_mask_np = gt_mask_tensor.detach().cpu().numpy().astype(bool)

                            # 在文件名中加入样本索引 i
                            save_name = f"epoch{epoch}_step{global_optim_step}_sample{i}"
                            
                            # 保存纯黑白的预测mask
                            pred_save_path = os.path.join(vis_dir, f"{save_name}_pred.png")
                            cv2.imwrite(pred_save_path, pred_mask_np.astype(np.uint8) * 255)
                            
                            # 保存纯黑白的GT mask (用于对比)
                            gt_save_path = os.path.join(vis_dir, f"{save_name}_gt.png")
                            cv2.imwrite(gt_save_path, gt_mask_np.astype(np.uint8) * 255)
                            
                            # 保存叠加后的可视化结果
                            overlay_img = original_image.copy()
                            # 确保 pred_mask_np 是布尔类型以进行索引
                            pred_mask_bool = pred_mask_np.astype(bool)
                            overlay_img[pred_mask_bool] = (
                                overlay_img * 0.5 + pred_mask_bool[:, :, None] * np.array([255, 0, 0]) * 0.5
                            )[pred_mask_bool]
                            overlay_save_path = os.path.join(vis_dir, f"{save_name}_overlay.png")
                            cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                            
                            print(f"  Saved visualization for sample {i} to {vis_dir} with prefix {save_name}")

                except Exception as e:
                    print(f"  Could not save visualization: {e}")
            #
            # ### END: 调试代码块结束 ###
            #
            # ================================================================

            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(output_dict["loss"].item(), input_dict["images"].size(0))
            ce_losses.update(output_dict["ce_loss"].item(), input_dict["images"].size(0))
            # mask_bce_losses.update(output_dict["mask_bce_loss"].item(), input_dict["images"].size(0))
            bce_loss_value = output_dict["mask_bce_loss"]
            if isinstance(bce_loss_value, torch.Tensor):
                bce_loss_value = bce_loss_value.item()
            mask_bce_losses.update(bce_loss_value, input_dict["images"].size(0))

            dice_loss_value = output_dict["mask_dice_loss"]
            if isinstance(dice_loss_value, torch.Tensor):
                dice_loss_value = dice_loss_value.item()
            mask_dice_losses.update(dice_loss_value, input_dict["images"].size(0))

            # mask_dice_losses.update(output_dict["mask_dice_loss"].item(), input_dict["images"].size(0))

            loss_value = output_dict["mask_loss"]
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            mask_losses.update(loss_value, input_dict["images"].size(0))

            # mask_losses.update(output_dict["mask_loss"].item(), input_dict["images"].size(0))

            if rank == 0 and global_optim_step % args.print_freq == 0:
                progress.display(global_optim_step)
                if wandb is not None and args.use_wandb:
                    wandb.log({
                        "train/loss": losses.avg,
                        "train/ce_loss": ce_losses.avg,
                        "train/mask_bce_loss": mask_bce_losses.avg,
                        "train/mask_dice_loss": mask_dice_losses.avg,
                        "train/mask_loss": mask_losses.avg,
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_optim_step": global_optim_step
                    })
                if writer:
                    writer.add_scalar("train/loss", losses.avg, global_optim_step)
            
            # 检查是否处于最后1个epoch，并且当前全局优化步数是1000的倍数
            is_last_two_epochs = (epoch >= args.epochs - 1)
            is_checkpoint_step = (global_optim_step % 1000 == 0)

            # 仅在rank 0进程、最后两个epoch、每1000步且非第0步时保存
            if rank == 0 and is_last_two_epochs and is_checkpoint_step and global_optim_step > 0:
                print(f"\n--- Saving periodic checkpoint at Epoch {epoch}, Step {global_optim_step} ---")
                
                # 1. 创建 checkpoint 字典
                checkpoint = {
                    "epoch": epoch,
                    "global_optim_step": global_optim_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_score": best_score[0], # 保存当前的 best_score
                    "cur_ciou": cur_ciou[0],   # 保存当前的 cur_ciou
                }
                
                # 2. 定义保存路径
                #    使用 global_optim_step 来确保文件名的唯一性
                save_path = os.path.join(args.log_base_dir, f"checkpoint_epoch{epoch}_step{global_optim_step}.pth")
                
                # 3. 保存完整的 checkpoint
                torch.save(checkpoint, save_path)
                print(f"Periodic checkpoint saved at {save_path}")

            if val_loader is not None and global_optim_step % args.val_freq == 0:
                giou, ciou = validate(val_loader, model, global_optim_step, writer, args, rank, device)
                if rank == 0 and giou > best_score[0]:
                    best_score[0] = giou
                    cur_ciou[0] = ciou
                    # torch.save(model.state_dict(), os.path.join(args.log_base_dir, f"best_model_epoch{epoch}_step{global_optim_step}_giou{giou:.3f}_ciou{ciou:.3f}.pth"))

                    # --- 开始修改 ---
                    # 1. 创建完整的 checkpoint 字典
                    # (这需要 optimizer, scheduler, epoch 变量在当前作用域可用)
                    checkpoint = {
                        "epoch": epoch,
                        "global_optim_step": global_optim_step, # 保存当前的step也很有用
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_score": best_score[0], # 保存更新后的最佳分数
                        "cur_ciou": cur_ciou[0],   # 保存对应的ciou
                    }

                    # 2. 定义保存路径
                    save_path = os.path.join(args.log_base_dir, f"best_checkpoint_epoch{epoch}_step{global_optim_step}_giou{giou:.3f}_ciou{ciou:.3f}.pth")
                    
                    # 3. 保存完整的 checkpoint
                    torch.save(checkpoint, save_path)
                    print(f"New best checkpoint saved at {save_path}")
                    # --- 结束修改 ---

def main(args):
    device, world_size, rank = setup_ddp()

    if rank == 0:
        if wandb is not None and args.use_wandb:
            wandb.init(project="vlmsamseg_training", config=vars(args))
        os.makedirs(args.log_base_dir, exist_ok=True)
        writer = SummaryWriter(args.log_base_dir)
    else:
        writer = None
    dist.barrier()  

    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    tokenizer = processor.tokenizer


    # num_added_tokens = tokenizer.add_tokens("<SEG>")
    # args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    # print(f"seg_token_idx: {args.seg_token_idx}")

    new_tokens = ["<SEG>", "<neg_SEG>"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_tokens} new tokens: {new_tokens}")

    # 2. 获取两个 token 的 ID
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    args.neg_seg_token_idx = tokenizer("<neg_SEG>", add_special_tokens=False).input_ids[0]
    
    print(f"seg_token_idx: {args.seg_token_idx}")
    print(f"neg_seg_token_idx: {args.neg_seg_token_idx}")

    # 3. 获取新的 tokenizer 词汇表示大小
    args.tokenizer_len = len(tokenizer)

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "model": args.version,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "neg_seg_token_idx": args.neg_seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
        "torch_dtype": torch_dtype,
        "attention": args.attention,
        "tokenizer_len": args.tokenizer_len,
        "use_SEG_token": args.use_SEG_token,
    }
    config = transformers.AutoConfig.from_pretrained(args.version)
    model = VlmSamSegForCausalLM(config, **model_args).to(device)

    version_str = args.version.lower()

    if "qwen3" in version_str:
        model.vlm_point_mode = "qwen3"
    elif "qwen2.5" in version_str or "qwen25" in version_str:
        model.vlm_point_mode = "qwen25"
    else:
        # safe default (most of your current data seems qwen3-style)
        model.vlm_point_mode = "qwen3"

    print(f"[INFO] vlm_point_mode set to: {model.vlm_point_mode}")

    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id
    model.vlm.enable_input_require_grads()
    for p in model.vlm.visual.parameters():
        p.requires_grad = False

    if args.lora_r > 0:
        lora_target_modules = args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.vlm = get_peft_model(model.vlm, lora_config)
        model.vlm.print_trainable_parameters()

    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "text_hidden_fcs"]) or \
           "mask_decoder" in n:
            p.requires_grad = True
    model.vlm.resize_token_embeddings(len(tokenizer))

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=0.0
    )
    optim_steps_per_epoch = args.steps_per_epoch // args.grad_accumulation_steps
    total_optim_steps = args.epochs * optim_steps_per_epoch
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_optim_steps
    )

    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.overlap_json_path,
        samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        cot_data=args.cot_data,
        explanatory=args.explanatory,
        sem_seg_p=[1.0, 0.0, 0.0],
        num_points=args.num_points,
        use_SEG_token=args.use_SEG_token,
        )
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            local_rank=rank,
        ),
    )

    if not args.no_eval or args.eval_only:
        val_dataset = ValDataset(args.dataset_dir, tokenizer, args.val_dataset, args.image_size)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                processor=processor,
                model_name="qwen_vl",
                local_rank=rank,
            ),
        )
        print(f"Training with {len(train_dataset)} examples, validating with {len(val_dataset)}.")
    else:
        val_loader = None
        print(f"Training with {len(train_dataset)} examples.")

    best_score = [0.0]
    cur_ciou = [0.0]

    if args.resume and not args.eval_only:  
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            args.start_epoch = checkpoint["epoch"] + 1  
            best_score[0] = checkpoint["best_score"]
            cur_ciou[0] = checkpoint["cur_ciou"]
            print(f"Resumed from epoch {checkpoint['epoch']}, best_score: {best_score[0]}, cur_ciou: {cur_ciou[0]}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch.")
    elif not args.eval_only:
        print("No resume checkpoint specified, starting from scratch.")

    if args.eval_only:
        if not args.resume:
            raise ValueError("For eval-only, please specify a checkpoint to load using --resume")
        checkpoint_path = args.resume
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)  
        model.load_state_dict(state_dict)
        model.eval()
        if val_loader is not None:
            giou, ciou = validate(val_loader, model, 0, writer, args, rank, device)
            if rank == 0:
                print(f"Evaluation results: giou={giou:.4f}, ciou={ciou:.4f}")
        else:
            print("No validation dataset specified.")
        if rank == 0:
            if writer:
                writer.close()
            if wandb is not None and args.use_wandb:
                wandb.finish()
        dist.destroy_process_group()
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, val_loader, model, tokenizer, optimizer, scheduler, epoch, writer, args, rank, best_score, cur_ciou, device)
        save_checkpoint(model, optimizer, scheduler, epoch, best_score[0], cur_ciou[0], args, rank)

    if rank == 0:
        if writer:
            writer.close()
        if wandb is not None and args.use_wandb:
            wandb.finish()
    dist.destroy_process_group()
    
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)