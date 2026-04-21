"""
SignCLIP Evaluation Tools

Modules for zero-shot and few-shot sign language recognition evaluation.
"""

from .demo_sign import embed_pose, embed_text, score_pose_and_text, score_pose_and_text_batch

__all__ = [
    'embed_pose',
    'embed_text',
    'score_pose_and_text',
    'score_pose_and_text_batch',
]

__version__ = '0.1.0'
