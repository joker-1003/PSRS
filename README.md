# PSRS: Point-Supervised Reasoning Segmentation with Qwen3-VL and SAM

PSRS combines a vision-language model (**Qwen3-VL-4B-Instruct**) with the **Segment Anything Model (SAM ViT-H)** to perform reasoning-driven referring segmentation. Given an image and a textual prompt, the VLM emits a `<SEG>` token plus point coordinates that condition SAM to produce a fine-grained mask. The model is trained end-to-end with LoRA on the language side and a trainable SAM mask decoder.

This repository provides the training, inference, and benchmark evaluation code. Pretrained checkpoints are released on the Hugging Face Hub:

> 🤗 **Checkpoints**: <https://huggingface.co/gudongxixixi/PSRS-Checkpoints>

---

## Table of Contents
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Architecture

```
                            ┌────────────────────────┐
   image ─┐                 │                        │
          │   ┌──────────►  │   Qwen3-VL-4B-Instruct │ ── tokens incl. <SEG>, (x,y) points
   prompt ┘   │             │      (LoRA on q,v)     │              │
              │             └────────────────────────┘              │
              │                          │                          │
              │                          │ <SEG> hidden state       │
              │                          ▼                          │
              │             ┌────────────────────────┐              │
              └──────────►  │      SAM ViT-H         │ ◄────────────┘
                            │  (frozen encoder +     │
                            │   trainable decoder)   │
                            └────────────────────────┘
                                          │
                                          ▼
                                  segmentation mask
```

- **VLM**: Qwen3-VL-4B-Instruct with LoRA (r=16) on `q_proj`, `v_proj`. Two new tokens are added to the tokenizer: `<SEG>` and `<neg_SEG>`.
- **Mask predictor**: SAM ViT-H. The encoder is frozen; the prompt encoder is fed projected `<SEG>` hidden states plus the predicted points; the mask decoder is fine-tuned.
- **Losses**: cross-entropy on language tokens (1.0×) + BCE (2.0×) + Dice (0.5×) on masks.
- **Training framework**: PyTorch native DDP (`torchrun`), bf16 autocast, manual gradient accumulation. No DeepSpeed / FSDP / Accelerate.

---

## Results

Numbers below are produced by the released checkpoint `PSRS.pth` with `use_SEG_token=True, num_points=1`.

### ReasonSeg (val, 200 samples)

| Subset                | gIoU   | cIoU   | #samples |
|-----------------------|--------|--------|----------|
| Original total        | 0.6580 | 0.6412 | 200      |
| Filtered total        | 0.6956 | 0.6837 | 133      |

### MechSeg-Bench (full, 1027 samples)

| Subset             | gIoU   | cIoU   | #samples |
|--------------------|--------|--------|----------|
| Total              | 0.6769 | 0.6369 | 1027     |
| Causal & Temporal  | 0.6673 | 0.5984 | 406      |
| Function & State   | 0.6905 | 0.6776 | 503      |
| Spatial & Topology | 0.6518 | 0.5644 | 118      |

### RefCOCO / RefCOCO+ / RefCOCOg (100 samples per split)

| Split           | gIoU   | cIoU   |
|-----------------|--------|--------|
| refcoco_val     | 0.8374 | 0.8470 |
| refcoco_testA   | 0.8747 | 0.8843 |
| refcoco_testB   | 0.7956 | 0.8033 |
| refcoco+_val    | 0.8505 | 0.8430 |
| refcoco+_testA  | 0.8350 | 0.7900 |
| refcoco+_testB  | 0.7056 | 0.6917 |
| refcocog_val    | 0.6996 | 0.6316 |
| refcocog_test   | 0.7887 | 0.7455 |

> RefCOCO numbers are from a 100-per-split sanity check; full-set evaluation is left to the user.

---

## Installation

PSRS is tested with **Python 3.10** and **CUDA 12.x**.

```bash
git clone https://github.com/joker-1003/<this-repo>.git
cd <this-repo>

# (Recommended) create a fresh conda environment
conda create -n psrs python=3.10 -y
conda activate psrs

# Install a CUDA-matched PyTorch first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Optional but recommended**: install `flash-attn` for faster attention.
```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

If you need to reproduce the exact environment we used for the released numbers, see `requirements-frozen.txt` (227 packages, pip-freeze snapshot).

---

## Pretrained Weights

1. **PSRS checkpoints** — download from Hugging Face:
   ```bash
   huggingface-cli download gudongxixixi/PSRS-Checkpoints --local-dir ./checkpoints
   ```
   The released checkpoints are LoRA-merged state dicts that can be loaded directly by `inference_*_evaluate.py` and `inference.py`.

2. **SAM ViT-H weights** — download from Meta's official SAM release and place at `./weights/sam_vit_h_4b8939.pth`:
   ```bash
   mkdir -p weights
   wget -O weights/sam_vit_h_4b8939.pth \
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

3. **Qwen3-VL-4B-Instruct** is pulled automatically by `transformers` from the Hugging Face Hub on first use. Override with `--version /path/to/local/Qwen3-VL-4B-Instruct` if you have a local copy.

---

## Dataset Preparation

PSRS trains on a mixture of public datasets. Place all datasets under a single `--dataset_dir` (default `./dataset`):

```text
./dataset/
├── ade20k/                  # ADE20K semantic segmentation
│   ├── images/
│   └── annotations/
├── COCO_Stuff/              # COCOStuff labels
│   └── train2017/
├── COCO/
│   └── train2017/           # COCO 2017 images (used by COCOStuff & MechSeg-Bench)
├── train2014/               # COCO 2014 images (used by RefCOCO/+/g)
├── refcoco/                 # RefCOCO annotations (LISA-style)
├── refcoco+/
├── refcocog/
├── ReasonSeg/               # ReasonSeg val/test JSONL + image folders
│   ├── reasonseg_val_fixed.jsonl
│   ├── reasonseg_val_fixed_filtered.jsonl
│   ├── val/
│   ├── reasonseg_test_fixed.jsonl
│   └── test/
├── refcoco_eval/            # 8 evaluation JSONLs for RefCOCO benchmarks
│   ├── refcoco_val.jsonl
│   ├── refcoco_testA.jsonl
│   └── ...
└── MechSeg-Bench/
    └── new_final_test.json  # MechSeg-Bench evaluation JSON
```

**Source pointers**:
- ADE20K: <https://groups.csail.mit.edu/vision/datasets/ADE20K/>
- COCO 2014 / 2017: <https://cocodataset.org>
- COCOStuff: <https://github.com/nightrome/cocostuff>
- RefCOCO/+/g: <https://github.com/lichengunc/refer> (use the LISA preprocessing for training)
- ReasonSeg: <https://github.com/dvlab-research/LISA> (originally proposed by LISA)
- MechSeg-Bench: dataset accompanying this work; see the HF checkpoint repo for download instructions.

For training, only `--dataset_dir` and (optionally) `--overlap_json_path` for the MechSeg-Bench training split are required.

---

## Training

A ready-to-run script that mirrors the configuration of the released checkpoint:

```bash
# Single GPU
bash scripts/run_train.sh

# Multi-GPU (e.g. 8)
bash scripts/run_train.sh 8

# Resume from a checkpoint
bash scripts/run_train.sh 8 ./runs/psrs_main_<TS>/checkpoint_epochN.pth

# Train including overlap_reasonseg (MechSeg-Bench)
OVERLAP_JSON=./dataset/MechSeg-Bench/train_singlepoint_pos_neg.json \
    bash scripts/run_train.sh 8
```

Or invoke `train_ddp.py` directly for full control:

```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --vision_pretrained ./weights/sam_vit_h_4b8939.pth \
    --dataset_dir ./dataset \
    --dataset "sem_seg||refer_seg||ReasonSeg||overlap_reasonseg" \
    --sample_rates "9,9,1,3" \
    --epochs 30 --batch_size 2 --grad_accumulation_steps 5 --lr 4e-5 \
    --lora_r 16 --image_size 1024 \
    --use_SEG_token True --num_points 1 \
    --log_base_dir ./runs/psrs_main \
    --use_wandb            # optional
```

Run `python train_ddp.py --help` to see every flag.

### After training: merge LoRA into the base model

```bash
python transform_weight.py \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --resume  ./runs/psrs_main/checkpoint_epoch29.pth \
    --save_path ./checkpoints/PSRS_merged.pth \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules q_proj,v_proj
```

The merged checkpoint is what the inference / evaluation scripts load via `--resume`.

---

## Evaluation

Three benchmarks, one script each. Each script writes results to `./evaluate_results/<benchmark>_results/`.

```bash
# ReasonSeg (val by default; pass reasonseg_test for the test split)
bash scripts/run_eval_reasonseg.sh ./checkpoints/PSRS.pth reasonseg_val

# MechSeg-Bench
bash scripts/run_eval_mechseg.sh ./checkpoints/PSRS.pth

# RefCOCO / RefCOCO+ / RefCOCOg (all 8 splits by default)
bash scripts/run_eval_refcoco.sh ./checkpoints/PSRS.pth
```

Each script forwards to one of `inference_reasonseg_evaluate.py`, `inference_MechSeg_Bench_evaluate.py`, `inference_refcoco_evaluate.py`. Run any of them with `--help` for the full argument list.

---

## Inference

`inference.py` runs the model on a curated split (ReasonSeg / RefCOCO) and saves visualisations alongside the predicted masks:

```bash
python inference.py \
    --resume ./checkpoints/PSRS.pth \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --dataset reasonseg_val \
    --reasonseg_root ./dataset/ReasonSeg \
    --output_dir ./evaluate_results/ReasonSeg_results \
    --save_vis
```

---

## Repository Structure

```text
.
├── train_ddp.py                          # DDP training entry
├── inference.py                          # Batch inference + visualisation
├── inference_reasonseg_evaluate.py       # ReasonSeg gIoU/cIoU evaluator
├── inference_MechSeg_Bench_evaluate.py   # MechSeg-Bench evaluator
├── inference_refcoco_evaluate.py         # RefCOCO/+/g evaluator
├── transform_weight.py                   # Merge LoRA adapters into base weights
├── scripts/
│   ├── run_train.sh
│   ├── run_eval_reasonseg.sh
│   ├── run_eval_mechseg.sh
│   └── run_eval_refcoco.sh
├── model/
│   ├── vlmsam.py                         # VlmSamSegForCausalLM definition
│   └── segment_anything/                 # Meta SAM source (Apache-2.0)
├── utils/
│   ├── dataset.py                        # HybridDataset, ValDataset, collate_fn
│   ├── conversation.py
│   ├── data_processing.py
│   ├── reason_seg_dataset.py
│   ├── overlap_reasonseg_dataset.py
│   ├── refer_seg_dataset.py
│   ├── sem_seg_dataset.py
│   ├── cot_dataset.py
│   ├── vqa_dataset.py
│   ├── refer.py / grefer.py / grefcoco.py
│   ├── ade20k_classes.json
│   ├── cocostuff_classes.txt
│   └── utils.py
├── requirements.txt                      # Curated, minimal dependency list
├── requirements-frozen.txt               # Full pip freeze of a known-good env
├── LICENSE                               # Apache 2.0
└── README.md
```

---

## Acknowledgements

PSRS builds on the shoulders of:

- **[LISA](https://github.com/dvlab-research/LISA)** — the original "Reasoning Segmentation via LLM" framework, whose dataset loaders and training recipe heavily inspired this codebase.
- **[Qwen-VL](https://github.com/QwenLM/Qwen3-VL)** — the vision-language model we fine-tune.
- **[Segment Anything](https://github.com/facebookresearch/segment-anything)** — Meta AI's promptable segmentation backbone.
- **[ReasonSeg](https://github.com/dvlab-research/LISA)** — the reasoning segmentation benchmark.

We thank the authors of all of the above for releasing their work openly.

---

## Citation

If you use PSRS in your research, please cite:

```bibtex
@misc{psrs2026,
    title={PSRS: Point-Supervised Reasoning Segmentation with Qwen3-VL and SAM},
    author={<authors>},
    year={2026},
    howpublished={\url{https://github.com/joker-1003/<this-repo>}}
}
```

(BibTeX will be updated once a paper / preprint is available.)

---

## License

This project is released under the [Apache License 2.0](LICENSE). Note that the bundled `model/segment_anything/` directory is also licensed under Apache 2.0 by Meta. Pretrained weights from Hugging Face inherit the licenses of their respective base models (Qwen3-VL, SAM).
