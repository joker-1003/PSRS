import argparse
import os
import torch
import transformers
from model.vlmsam import VlmSamSegForCausalLM
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a LoRA-adapted checkpoint, merge the adapters into the base model, and save the merged checkpoint."
    )
    parser.add_argument("--version", default="Qwen/Qwen3-VL-4B-Instruct", type=str,
                        help="Pretrained Qwen3-VL HuggingFace id or local path")
    parser.add_argument("--resume", required=True, type=str,
                        help="Path to the training checkpoint to load")
    parser.add_argument("--save_path", required=True, type=str,
                        help="Path to save the merged model state dict")
    parser.add_argument("--use_SEG_token", default=True,
                        type=lambda x: str(x).lower() == "true")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    return parser.parse_args()

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    args = parse_args()

    # --- 1. 初始化模型结构 (与训练时保持一致) ---
    print("Initializing model architecture...")
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32
    config = transformers.AutoConfig.from_pretrained(args.version)
    
    # 确保这里的参数与训练时使用的参数一致
    model_args = {
        "train_mask_decoder": True,
        "model": args.version,
        "out_dim": 256,
        # "seg_token_idx": 151851, # 请确保这个ID是正确的
        "seg_token_idx": 151665,
        "neg_seg_token_idx": 151666,
        "torch_dtype": torch_dtype,
        "attention": "flash_attention_2",
        "use_SEG_token": args.use_SEG_token,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VlmSamSegForCausalLM(config, **model_args).to(device)

    # --- 2. 应用LoRA配置，使模型结构准备好接收LoRA权重 ---
    lora_target_modules = args.lora_target_modules.split(",")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.vlm = get_peft_model(model.vlm, lora_config)
    
    # 调整token embedding大小以匹配tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.version)
    tokenizer.add_tokens("<SEG>")
    tokenizer.add_tokens("<neg_SEG>")
    model.vlm.resize_token_embeddings(len(tokenizer))

    # --- 3. 加载训练好的checkpoint ---
    print(f"Loading checkpoint from {args.resume} ...")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = remove_module_prefix(state_dict)

    # --- 4. 严格加载权重并进行检查 (这是关键修正) ---
    print("Loading state dict with strict checking...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print("\n--- Sanity Check Report ---")
    print(f"Unexpected Keys: {unexpected_keys}")
    print(f"Missing Keys: {missing_keys}")

    # 关键检查：确保分割头 (MLP和Mask Decoder) 的权重没有出现在 "missing_keys" 中
    is_mlp_missing = any("text_hidden_fcs" in key for key in missing_keys)
    is_decoder_missing = any("mask_decoder" in key for key in missing_keys)

    if is_mlp_missing or is_decoder_missing:
        print("\n\nFATAL ERROR: The checkpoint is missing crucial weights for the segmentation head!")
        print("This is the reason for incorrect mask generation.")
        if is_mlp_missing:
            print("- The trained MLP weights ('text_hidden_fcs') were not found.")
        if is_decoder_missing:
            print("- The trained Mask Decoder weights ('mask_decoder') were not found.")
        print("Please check your training and checkpoint saving process.")
        return # 终止脚本
    
    print("\n✅ Sanity check passed. All critical weights are present in the checkpoint.")
    print("---------------------------\n")

    # --- 5. 合并LoRA权重 ---
    print("Merging LoRA weights into the base model...")
    model.vlm = model.vlm.merge_and_unload()
    
    # --- 6. 保存完整的、合并后的模型 ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Merged and fully trained model saved at {args.save_path}")

if __name__ == "__main__":
    main()