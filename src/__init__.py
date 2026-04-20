"""
SignCLIP Evaluation Tools

Modules for zero-shot and few-shot sign language recognition evaluation.
"""

from .pose_dataset import PoseDataset
from .demo_sign import embed_pose, embed_text, score_pose_and_text, score_pose_and_text_batch

__all__ = [
    'PoseDataset',
    'embed_pose',
    'embed_text',
    'score_pose_and_text',
    'score_pose_and_text_batch',
]

__version__ = '0.1.0'
