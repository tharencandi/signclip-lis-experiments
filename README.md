# SignCLIP LIS Experiments

Evaluation of [SignCLIP](https://arxiv.org/abs/2407.10925) on Italian Sign Language (LIS) across zero-shot, few-shot, and fine-tuning paradigms. Datasets: **A3LIS-147** (balanced, lab-recorded) and **SignIT** (naturalistic, long-tailed). Adapted from the original MMPT/fairseq implementation with fairseq dependencies removed.

repo based on J22Melody/fairseq

## Structure

```
signclip-lis-experiments/
├── signclip/                  # Core package (no fairseq deps)
├── configs/                   # Model configs (signclip_v1_1, ablations, ...)
├── dataset/
│   ├── embeddings/            # Precomputed .npy pose embeddings by model variant
│   ├── A3LIS_dataset_poses/
│   └── SignIT_dataset_poses/
├── src/                       # Evaluation and analysis scripts
│   ├── few_shot.py            # linear-probe etc 
│   ├── zero_shot.py           # Zero-shot video-text retrieval
│   ├── precompute_embeddings.py 
├── scripts/                   # Training / batch eval shell scripts
├── runs/                      # Experiment outputs and CSVs
├── pretrained_models/         # Checkpoint weights
├── docs/                      # Report (Typst)
├── environment.yml
└── requirements.txt
```

## Installation

```bash
conda env create -f environment.yml
conda activate signclip
pip install -e .
python setup_local_config.py   # creates required dirs and local config
```

## Key Workflows

**1. Precompute embeddings**
```bash
python src/precompute_embeddings.py \
  --config configs/signclip_v1_1/baseline_temporal.yaml \
  --pose_dir dataset/A3LIS_dataset_poses \
  --output_dir dataset/embeddings/a3lis_baseline
```

**2. Few-shot evaluation** (prototypical / linear-probe / KNN)
```bash
python src/few_shot.py \
  --pose_embeddings_dir dataset/embeddings/a3lis_baseline \
  --method prototypical --label_language english
```

**3. Zero-shot retrieval**
```bash
python src/zero_shot.py \
  --pose_embeddings_dir dataset/embeddings/a3lis_baseline \
  --label_language english
```

## Fine-tuning

Loss functions evaluated: InfoNCE, SupCon, Cross-Entropy, DHN-NCE, **GlobalNCE** (best), **ProLIP** (parameter-efficient). All configs are in `configs/`.

**A3LIS**
```bash
python scripts/run_finetune.py --config configs/signclip_v1_1/a3lis_finetune.yaml
```

**SignIT**
```bash
python scripts/run_signit_finetune.py --config configs/signclip_v1_1/signit_finetune.yaml
```

Results are written to `runs/`.

## Reference

This project builds on SignCLIP. Please cite the original work:

```bibtex
@inproceedings{jiang2024signclip,
  title     = {{S}ign{CLIP}: Connecting Text and Sign Language by Contrastive Learning},
  author    = {Jiang, Zifan and Sant, Gerard and Moryossef, Amit and
               M{\"u}ller, Mathias and Sennrich, Rico and Ebling, Sarah},
  booktitle = {Proceedings of EMNLP 2024},
  year      = {2024},
}
```

