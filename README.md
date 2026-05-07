# PSRS: Reasoning-induced Segmentation via Progressive Signal Strengthening

**PSRS** (Progressive Signal Reasoning Segmentation) is a unified framework for *reasoning segmentation* — producing pixel-level masks from instructions whose targets are governed by world mechanisms (intrinsic attributes, spatial topology, causal-temporal dependencies).

Existing modular MLLM+segmenter approaches rely on **(a)** sparse coordinate prompts, **(b)** text-augmented sparse prompts, or **(c)** a single latent token. All three under-constrain *near-miss* candidates. PSRS instead builds a **progressive signal stack** that strengthens conditioning from coarse to fine:

> **Stage 1** Discrete Textual Commitments (SP-CoT) → **Stage 2** Topology-Aware Spatial Anchors (P⁺ / P⁻) → **Stage 3** Dense Latent Token (`<SEG>`)

Together with the proposed **MechSeg-Bench**, PSRS establishes a new state-of-the-art on reasoning segmentation while staying parameter-efficient (4B backbone outperforming 7B baselines).

> 🤗 **Pretrained checkpoints**: <https://huggingface.co/gudongxixixi/PSRS-Checkpoints>

---

## Table of Contents
- [Method](#method)
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

## Method

### Architecture

PSRS has three modules — a **Reasoning Core** (MLLM), a lightweight **Signal Projector** (MLP), and a **Segmentation Actuator** (SAM).

```
                ┌────────────────────────────────────────────┐
   image  ────► │            Qwen3-VL-4B-Instruct            │
   query  ────► │       (LoRA-tuned q_proj, v_proj)          │
                └─────────────┬──────────────────────────────┘
                              │
                              ▼
        ┌─────────────── adaptive output sequence ───────────────┐
        │  Stage 1: SP-CoT             Stage 2: Spatial anchors  │
        │  <think> Perception →        [x₁,y₁]⁺  (mandatory P⁺)  │
        │          Reasoning  →        [x₂,y₂]⁻  (optional P⁻)   │
        │          Prediction </think>                           │
        │                              Stage 3: <SEG> token      │
        └─────────────┬──────────────────────┬───────────────────┘
                      │                      │
                      │                      ▼  hidden state
                      │              ┌─────────────────┐
                      │              │   MLP Projector │
                      │              └────────┬────────┘
                      │                       │ E_dense (d_SAM)
                      ▼                       ▼
           Fourier positional       ┌────────────────────────┐
              embeddings   ───────► │   SAM ViT-H Decoder    │
                                    │  (frozen encoder +     │
                                    │   trainable decoder)   │
                                    └────────────┬───────────┘
                                                 ▼
                                       segmentation mask
```

### Adaptive signal generation: Direct Mode vs Reasoning Mode

The MLLM autonomously decides which mode to use based on query complexity:

- **Direct Mode** — explicit instructions (semantic segmentation, RefCOCO-style referring). The `<think>` block is skipped; the model emits anchors and `<SEG>` directly for inference efficiency.
- **Reasoning Mode** — implicit physical / causal queries. The full SP-CoT chain is activated:
  1. **Perception** — map visual entities to semantic concepts.
  2. **Reasoning** — apply mechanism constraints, identify distractors.
  3. **Prediction** — commit to the target.
  
  Then the model emits `[x₁,y₁]⁺ ... ([x₂,y₂]⁻)? ... <SEG>`. Negative anchors P⁻ are produced **only** when the SP-CoT detects near-miss candidates.

### Training objective

PSRS is trained end-to-end with a multi-task loss:

```
L = λ_gen · L_gen + λ_mask · L_mask
L_gen  = autoregressive CE over the full token sequence (SP-CoT + anchors + <SEG>)
L_mask = λ_bce · L_bce + λ_dice · L_dice          # default λ_bce=2.0, λ_dice=0.5
```

The SAM image encoder is **frozen**; the SAM decoder, the MLP projector, the LoRA adapters on the Qwen3-VL `q_proj`/`v_proj`, and the embeddings of the new tokens (`<SEG>`, `<neg_SEG>`) are trained.

Optimization: AdamW with cosine learning-rate schedule, bf16 mixed precision, PyTorch native DDP via `torchrun`. Training data is a mixture of ADE20K, COCOStuff, RefCOCO/+/g, ReasonSeg, and the MechSeg training set (see [Dataset Preparation](#dataset-preparation)).

### MechSeg-Bench

A new mechanism-aware reasoning-segmentation benchmark with **~1,000 samples** balanced across three reasoning dimensions:

| Dimension | Focus |
|---|---|
| **Intrinsic Semantic** | function / state (e.g., "an apple vs unpeeled fruit for *eaten immediately*") |
| **Spatial-Topological** | 3D structural symmetry & dynamic accessibility (e.g., the *first furniture encountered* along a route) |
| **Causal-Temporal** | counterfactuals & future consequences (e.g., the object to *set upright* to resume travel) |

MechSeg-Bench is built atop AS-V2 with a 4-step Generate-then-Filter pipeline:
1. **Generate** queries with GPT-5 under mechanism-specific prompts.
2. **Filter** with a probe model (Qwen2.5-VL-7B): retain samples with `IoU_Direct < 0.5` *and* `IoU_CoT > 0.7` (the "Reasoning Gap" — intuition-hard but logically verifiable). This yields a ~5% retention rate.
3. **Annotate** SP-CoT and P⁺/P⁻ via Qwen2.5-VL-72B with geometry-consistency calibration.
4. **Generate masks** with SAM, refined by a human-in-the-loop protocol.

---

## Results

All numbers below are from the paper. Metric definitions:
- **gIoU**: per-sample IoU averaged over the dataset (favored for small instances).
- **cIoU**: cumulative intersection / cumulative union (sensitive to large objects).

### Reasoning segmentation (Table 1)

| Method | ReasonSeg val | ReasonSeg test | MechSeg Intrinsic | MechSeg Spatial | MechSeg Causal | MechSeg Overall |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| | gIoU / cIoU | gIoU / cIoU | gIoU / cIoU | gIoU / cIoU | gIoU / cIoU | gIoU / cIoU |
| LISA-7B | 52.9 / 54.0 | 47.3 / 48.4 | 39.3 / 37.9 | 33.5 / 25.1 | 34.5 / – | 33.7 / 36.7 |
| SegZero-7B | 62.6 / 62.0 | – / – | 55.2 / 48.0 | 51.3 / 40.0 | 62.8 / – | 54.0 / 46.5 |
| READ | 59.8 / 67.6 | 57.2 / 58.0 | 52.4 / 48.6 | 42.8 / 28.8 | 47.3 / – | 47.7 / 43.5 |
| **PSRS (ours, 4B)** | **66.0 / 64.4** | **59.9 / 58.4** | **69.4 / 67.4** | **65.7 / 56.9** | **67.5 / 61.0** | **68.2 / 64.0** |

PSRS improves over LISA-7B by **+13.1 gIoU** on ReasonSeg val and over SegZero-7B by **+3.4 gIoU**, despite using a 4B (vs 7B) backbone. On MechSeg-Bench it leads by large margins on every reasoning dimension.

### Referring expression segmentation — RefCOCO/+/g (Table 2, cIoU)

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val | test | Avg |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LISA-7B | 74.1 | 76.5 | 71.1 | 62.4 | 67.4 | 56.5 | 66.4 | 68.5 | 67.9 |
| GSVA-7B | 76.4 | 77.4 | 72.8 | 64.5 | 67.7 | 58.6 | 71.1 | 72.0 | 70.1 |
| OMG-LLaVA | 78.0 | 80.3 | 74.1 | 69.1 | 73.1 | 63.0 | 72.9 | 72.9 | 72.9 |
| SegLLM | 80.2 | 81.5 | 75.4 | 70.3 | 73.0 | 62.5 | 72.6 | 73.6 | 73.6 |
| **PSRS (ours)** | 78.4 | **80.9** | 74.2 | **73.0** | **78.0** | **66.2** | **74.1** | **74.7** | **74.9** |

PSRS is jointly optimized for both reasoning and explicit referring tasks; nevertheless it achieves the best average cIoU.

### Ablation (MechSeg-Bench, Table 3)

| #P⁺ | `<SEG>` | P⁻ | gIoU | cIoU |
|:-:|:-:|:-:|:-:|:-:|
| 1 | ✗ | ✓ | 66.3 | 57.4 |
| 1 | ✓ | ✗ | 63.5 | 56.1 |
| 3 | ✓ | ✓ | 66.6 | 60.4 |
| **1** | **✓** | **✓** | **68.2** | **64.0** |

Removing P⁻ causes the largest drop (−4.7 gIoU); a single positive anchor with `<SEG>` and adaptive P⁻ is optimal — more positives crowd out the dense `<SEG>` and exclusion logic.

---

## Installation

PSRS is tested with **Python 3.10** and **CUDA 12.x**.

```bash
git clone https://github.com/joker-1003/PSRS.git
cd PSRS

conda create -n psrs python=3.10 -y
conda activate psrs

# Install a CUDA-matched PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Optional — Flash Attention** (recommended for training):
```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For the exact pin set used to reproduce the paper's numbers, see `requirements-frozen.txt`.

---

## Pretrained Weights

1. **PSRS checkpoints** — already LoRA-merged, ready for evaluation:
   ```bash
   huggingface-cli download gudongxixixi/PSRS-Checkpoints --local-dir ./checkpoints
   ```

2. **SAM ViT-H weights** (Meta's official release):
   ```bash
   mkdir -p weights
   wget -O weights/sam_vit_h_4b8939.pth \
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

3. **Qwen3-VL-4B-Instruct** is pulled automatically from the Hugging Face Hub on first use. Pass `--version /path/to/local/Qwen3-VL-4B-Instruct` to override.

---

## Dataset Preparation

PSRS is trained on a mixture of public datasets plus the MechSeg training set. Place them under a single `--dataset_dir` (default `./dataset`):

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
├── refcoco/                 # RefCOCO-series annotations (LISA-style)
├── refcoco+/
├── refcocog/
├── ReasonSeg/
│   ├── reasonseg_val_fixed.jsonl
│   ├── reasonseg_val_fixed_filtered.jsonl   # ReasonSeg-Clean diagnostic subset
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
- RefCOCO/+/g: <https://github.com/lichengunc/refer> (LISA preprocessing)
- ReasonSeg: <https://github.com/dvlab-research/LISA>
- MechSeg train + bench JSONs: see the HF checkpoint repo

For training, only `--dataset_dir` and (when `overlap_reasonseg` is in the data mix) `--overlap_json_path` are required.

---

## Training

A ready-to-run script that mirrors the paper's setting:

```bash
# Single GPU
bash scripts/run_train.sh

# Multi-GPU (e.g. 8 × A100)
bash scripts/run_train.sh 8

# Resume
bash scripts/run_train.sh 8 ./runs/psrs_main_<TS>/checkpoint_epochN.pth

# Include MechSeg training data (overlap_reasonseg)
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

The reference configuration uses **`num_points=1`** with **`use_SEG_token=True`** — the optimal setting from Table 3.

Run `python train_ddp.py --help` to see every flag.

### After training: merge LoRA into the base model

The released checkpoints have LoRA already merged. If you finish your own run, merge it before evaluation:

```bash
python transform_weight.py \
    --version "Qwen/Qwen3-VL-4B-Instruct" \
    --resume  ./runs/psrs_main/checkpoint_epoch29.pth \
    --save_path ./checkpoints/PSRS_merged.pth \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules q_proj,v_proj
```

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
│   ├── vlmsam.py                         # VlmSamSegForCausalLM (Reasoning Core + Projector + Actuator)
│   └── segment_anything/                 # Meta SAM source (Apache-2.0)
├── utils/
│   ├── dataset.py                        # HybridDataset, ValDataset, collate_fn
│   ├── conversation.py                   # Conversation templates incl. SP-CoT format
│   ├── data_processing.py
│   ├── reason_seg_dataset.py
│   ├── overlap_reasonseg_dataset.py      # MechSeg training-data loader
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

PSRS builds on the shoulders of several open-source efforts:

- **[LISA](https://github.com/dvlab-research/LISA)** — the original Reasoning Segmentation framework that proposed `<SEG>` token conditioning and the ReasonSeg benchmark; our dataset loaders and training recipe are derived from LISA.
- **[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)** — the vision-language reasoning core.
- **[Segment Anything](https://github.com/facebookresearch/segment-anything)** — Meta AI's promptable segmentation backbone.
- **[Seg-Zero](https://github.com/dvlab-research/Seg-Zero)** and **[READ](https://github.com/whatakitakai/READ)** — strong recent baselines we compare against.

We thank the authors of all of the above.

---

## Citation

If you use PSRS in your research, please cite:

```bibtex
@inproceedings{psrs2026,
  title     = {PSRS: Reasoning-induced Segmentation via Progressive Signal Strengthening},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Under review}
}
```

(Citation will be updated once the paper is published.)

---

## License

This project is released under the [Apache License 2.0](LICENSE). The bundled `model/segment_anything/` directory is also Apache-2.0 (Meta AI). Pretrained weights from Hugging Face inherit the licenses of their respective base models (Qwen3-VL, SAM).
