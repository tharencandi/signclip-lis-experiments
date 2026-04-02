# SignCLIP Experiments

Extracted from J22Medoly/fairseq

## Directory Structure
```
signclip-lis-experiments/
├── signclip/              # Core package (no fairseq deps!)
│   ├── models/            # Model architectures
│   ├── tasks/             # Training tasks
│   ├── processors/        # Data processors
│   ├── datasets/          # Dataset classes
│   ├── evaluators/        # Evaluation metrics
│   ├── losses/            # Loss functions
│   ├── modules/           # Model modules
│   └── utils/             # Utilities
├── configs/               # Model configs (all variants)
├── scripts/               # Your training/eval scripts
├── upstream_reference/    # Original SignCLIP scripts (reference only)
│   ├── demos/             # Original demo scripts
│   ├── tests/             # Original test scripts  
│   └── analysis/          # Original results/analysis scripts
├── pretrained_models/     # Downloaded checkpoint weights
├── tests/                 # Your test suite
├── environment.yml        # Conda environment
└── requirements.txt       # Pip dependencies
```

## Installation

### Local Install
```bash
pip install -e .
```

## Colab Setup w/ Konda
```
!pip install konda
import konda; konda.install()

!git clone https://github.com/yourusername/signclip-experiments.git
%cd signclip-experiments

!konda env create -f environment.yml -n signclip -y
!konda activate signclip

# Install package in dev mode
!konda run "pip install -e ."
```

## Quick Start

Inference
```python 
from signclip.models import MMPTModel

# Load pretrained model
model, tokenizer, aligner = MMPTModel.from_pretrained(
    "configs/signclip_v1_1/baseline_temporal.yaml"
)

# Run inference on pose file
embeddings = model.embed_pose("path/to/sign.pose")
```

fine-tuning
```python
from signclip.models import MMPTModel
from scripts.train_simple import simple_train

# Load pretrained
model, tokenizer, aligner = MMPTModel.from_pretrained("configs/signclip_v1_1/...")

# Your data
train_loader = DataLoader(your_dataset, batch_size=32)

# Train
simple_train(model, train_loader, epochs=10, lr=5e-5)
```

## Upstream Reference Scripts

The `upstream_reference/` folder contains **original scripts from the SignCLIP authors** for reference purposes. These include additional demos, tests, and analysis scripts from the paper. See [upstream_reference/README.md](upstream_reference/README.md) for details.

⚠️ These are provided as-is from the original fairseq/MMPT implementation and may have additional dependencies.

## Citation
    title = "{S}ign{CLIP}: Connecting Text and Sign Language by Contrastive Learning",
    author = {Jiang, Zifan and Sant, Gerard and Moryossef, Amit and 
              M{\"u}ller, Mathias and Sennrich, Rico and Ebling, Sarah},
    booktitle = "Proceedings of EMNLP 2024",
    year = "2024",
}
MIT License (adapted from original MMPT/fairseq)

