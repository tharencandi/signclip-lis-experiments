import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow C++ backend logs
os.environ['GLOG_minloglevel'] = '3'          # Suppress GLOG messages from XLA/CUDA

import sys
from pathlib import Path

# Add project root to path for signclip imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import warnings
# warnings.filterwarnings("ignore")           # Suppress Python warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow Python logs

# Optionally, if using Hugging Face Transformers, you can suppress its logs too:
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

import argparse
import torch
import numpy as np
from pose_format import Pose
from signclip.models import MMPTModel
from signclip.tasks.task import Task
from signclip.utils.load_config import load_config
from signclip.utils.pose_utils import (
    FACEMESH_CONTOURS_POINTS,
    MAX_FRAMES as MAX_FRAMES_DEFAULT,
    pose_normalization_info,
    pose_hide_legs,
    preprocess_pose,
)

# Model configurations - use projects/retri/ like original
# a3lis_finetune uses None because it loads from a checkpoint file, not a retri YAML
model_configs = [
    ("default", "signclip_v1_1/baseline_temporal"),
    ("asl_citizen", "signclip_asl/asl_citizen_finetune"),
    ("asl_finetune", "signclip_asl/asl_finetune"),
    ("suisse", "signclip_suisse/suisse_finetune"),
    ("a3lis_finetune", None),  # fine-tuned on A3LIS-147 LIS classes
]

# Checkpoint and base config for the A3LIS fine-tuned model.
# The fine-tuned weights are overlaid on the same architecture as 'default'.
A3LIS_FINETUNE_CHECKPOINT = "runs/signclip_a3lis_finetune/checkpoint_best.pt"
A3LIS_FINETUNE_BASE_CONFIG = "projects/retri/signclip_v1_1/baseline_temporal.yaml"

# Cache for models that have been lazily initialized.
models = {}


def _load_checkpoint_into_model(model, checkpoint_path, model_name):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = state.get('model_state_dict', state)
    if not any(k.startswith('model.') for k in sd):
        sd = {f'model.{k}': v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[{model_name}] Warning: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"[{model_name}] Warning: {len(unexpected)} unexpected keys in checkpoint")


def _build_model_from_config(config_path):
    config = load_config(config_file=config_path)
    mmtask = Task.config_task(config)
    model = MMPTModel(config, mmtask.build_model(), None)

    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    use_fast = OmegaConf.select(config.dataset, 'use_fast')
    if use_fast is None:
        use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.dataset.bert_name), use_fast=use_fast
    )

    from signclip.processors import Aligner
    aligner = Aligner(config.dataset)
    return model, tokenizer, aligner

def get_model(model_name, checkpoint_path=None):
    """
    Lazily load the requested model based on model_name.
    If the model is already loaded, return it.
    Otherwise, find its config, load it, and cache it.
    """
    cache_key = (model_name, str(checkpoint_path) if checkpoint_path else None)
    if cache_key in models:
        return models[cache_key]

    # Look up the configuration for the given model_name.
    config_path = None
    found = False
    for m_name, cfg in model_configs:
        if m_name == model_name:
            config_path = cfg
            found = True
            break
    if not found:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_name == 'a3lis_finetune' or checkpoint_path:
        # Load base architecture then overlay the requested checkpoint.
        if model_name == 'a3lis_finetune' and checkpoint_path is None:
            ckpt_path = Path(A3LIS_FINETUNE_CHECKPOINT)
        else:
            assert checkpoint_path is not None, "checkpoint_path is required when overriding the model checkpoint"
            ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned A3LIS checkpoint not found: {ckpt_path}\n"
                f"Run scripts/run_finetune.py first to produce the checkpoint."
            )
        model, tokenizer, aligner = _build_model_from_config(A3LIS_FINETUNE_BASE_CONFIG)
        _load_checkpoint_into_model(model, ckpt_path, model_name)
    else:
        # Standard YAML-based model loading.
        model, tokenizer, aligner = MMPTModel.from_pretrained(
            f"projects/retri/{config_path}.yaml",
            video_encoder='',
        )

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    models[cache_key] = {
        "model": model,
        "tokenizer": tokenizer,
        "aligner": aligner,
    }
    return models[cache_key]


def preprocess_text(text, model_name="default", checkpoint_path=None):
    model_info = get_model(model_name, checkpoint_path=checkpoint_path)
    aligner = model_info["aligner"]
    tokenizer = model_info["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_pose(pose, model_name='default', checkpoint_path=None):
    model_info = get_model(model_name, checkpoint_path=checkpoint_path)
    model = model_info['model']

    caps, cmasks = preprocess_text('', model_name, checkpoint_path=checkpoint_path)
    poses = pose if type(pose) == list else [pose]
    embeddings = []

    pose_frames_l = []
    for p in poses:
        # Truncate to MAX_FRAMES_DEFAULT (256) to match model config
        pose_frames = preprocess_pose(p, max_frames=MAX_FRAMES_DEFAULT)
        pose_frames_l.append(pose_frames)

    # Batch padding
    # 1) find the longest sequence
    max_len = max(pf.shape[1] for pf in pose_frames_l)

    # 2) pad each to max_len on the time‐axis
    padded = []
    for pf in pose_frames_l:
        pad_len = max_len - pf.shape[1]
        if pad_len > 0:
            # create zeros of shape (batch=1, pad_len, features)
            pad = pf.new_zeros((pf.shape[0], pad_len, pf.shape[2]))
            pf = torch.cat([pf, pad], dim=1)
        padded.append(pf)

    # 3) now you can concatenate without size mismatches
    pose_frames_l = torch.cat(padded, dim=0)

    batch_size = len(poses)

    with torch.no_grad():
        output = model(pose_frames_l,
                       caps.repeat(batch_size, 1),
                       cmasks.repeat(batch_size, 1),
                       return_score=False)
        embeddings.append(output['pooled_video'].cpu().numpy())

    return np.concatenate(embeddings)


def embed_text(text, model_name='default', checkpoint_path=None):
    model_info = get_model(model_name, checkpoint_path=checkpoint_path)
    model = model_info['model']
    
    # Determine the placeholder dimension based on the model_name.
    if model_name == 'lip':
        placeholder_dim = 1377
    elif model_name == 'lip_only':
        placeholder_dim = 768
    else:
        placeholder_dim = 609

    # Ensure texts is a list.
    texts = text if isinstance(text, list) else [text]
    batch_size = len(texts)

    # Preprocess each text individually and store the results.
    caps_list = []
    cmasks_list = []
    for t in texts:
        caps, cmasks = preprocess_text(t, model_name, checkpoint_path=checkpoint_path)
        caps_list.append(caps)   # Each should have shape (1, 128)
        cmasks_list.append(cmasks)

    # Concatenate the individual results along the batch dimension.
    caps_batch = torch.cat(caps_list, dim=0)
    cmasks_batch = torch.cat(cmasks_list, dim=0)

    # Create dummy pose_frames with shape (batch_size, 1, placeholder_dim).
    pose_frames = torch.randn(batch_size, 1, placeholder_dim)

    # Run the model forward pass only once with the full batch.
    with torch.no_grad():
        output = model(pose_frames, caps_batch, cmasks_batch, return_score=False)
    
    # Extract the pooled text embeddings and return as a NumPy array.
    embeddings = output['pooled_text'].cpu().numpy()
    return embeddings


def score_pose_and_text(pose, text, model_name="default", max_frames=None):
    model_info = get_model(model_name)
    model = model_info["model"]

    pose_frames = preprocess_pose(
        pose,
        MAX_FRAMES_DEFAULT if max_frames is None else max_frames,
    )
    caps, cmasks = preprocess_text(text, model_name)

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=True)

    return text, float(output["score"])  # dot-product


def score_pose_and_text_batch(pose, text, model_name='default'):
    pose_embedding = embed_pose(pose, model_name)
    text_embedding = embed_text(text, model_name)

    scores = np.matmul(pose_embedding, text_embedding.T)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose and text similarity using SignCLIP.")
    parser.add_argument(
        "--pose_path",
        default="./house.pose",
        type=Path,
        help="Path to the .pose file.",
    )
    parser.add_argument(
        "--max_frames",
        nargs="?",
        type=int,
        const=MAX_FRAMES_DEFAULT,
        default=None,
        help=f"If provided, pose sequences longer than this will be truncated, otherwise they will not. If provided without a value, will use {MAX_FRAMES_DEFAULT}, as SignCLIP can currently only support this many. If provided with a value, will use that value",
    )

    args = parser.parse_args()

    pose_path = args.pose_path
    max_frames = args.max_frames

    if not pose_path.is_file():
        print(f"Error: File {pose_path} does not exist.")
        return

    with open(pose_path, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

        print(score_pose_and_text(pose, "random text", max_frames=max_frames))
        print(score_pose_and_text(pose, "house", max_frames=max_frames))
        print(score_pose_and_text(pose, "<en> <ase> house", max_frames=max_frames))
        print(score_pose_and_text(pose, "<en> <gsg> house", max_frames=max_frames))
        print(score_pose_and_text(pose, "<en> <fsl> house", max_frames=max_frames))
        print(score_pose_and_text(pose, "<en> <ase> sun", max_frames=max_frames))
        print(score_pose_and_text(pose, "<en> <ase> police", max_frames=max_frames))
        print(score_pose_and_text(pose, "<en> <ase> how are you?", max_frames=max_frames))

        text_l = ["<en> <ase> house", "<en> <ase> police"]
        pose_l = [pose, pose]
        print(score_pose_and_text_batch(pose_l, text_l))
        
        print(score_pose_and_text_batch(pose_l, text_l, model_name='asl_finetune'))


if __name__ == "__main__":
    main()
