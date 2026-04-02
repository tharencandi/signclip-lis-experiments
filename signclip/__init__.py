"""SignCLIP: Sign Language-Text Contrastive Learning

A clean, standalone package for SignCLIP inference and fine-tuning.
Extracted from the MMPT framework, removing fairseq dependencies.
"""

__version__ = "0.1.0"

from .models.mmfusion import MMPTModel, MMFusionSeparate
from .losses.nce import MMContraLoss

__all__ = [
    "MMPTModel",
    "MMFusionSeparate", 
    "MMContraLoss",
]
