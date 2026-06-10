"""
Shared pose preprocessing utilities for SignCLIP.

Used by:
  - src/demo_sign.py       (inference / embedding)
  - signclip/tasks/a3lis_finetune.py  (fine-tuning collate_fn)

The preprocessing pipeline here matches the SignCLIP pretraining pipeline
exactly.  Any data path that feeds the model MUST go through preprocess_pose()
so that fine-tuning and inference see the same input distribution.
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
from pose_format import Pose
from typing import Optional
# ---------------------------------------------------------------------------
# Face mesh contour landmark indices (avoids importing mediapipe at runtime).
# Generated from: sorted(set(p for tup in mp_holistic.FACEMESH_CONTOURS for p in tup))
# ---------------------------------------------------------------------------
FACEMESH_CONTOURS_POINTS = [
    '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
    '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
    '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
    '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
    '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
    '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
    '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
    '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
]

# Maximum sequence length accepted by the model (frames).
MAX_FRAMES = 256


def pose_normalization_info(pose_header):
    """Return shoulder-based normalization info for the given pose header schema."""
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        )
    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder"),
        )
    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder"),
        )
    raise ValueError(
        f"Could not parse normalization info: pose_header.components[0].name is "
        f"'{pose_header.components[0].name}'. "
        f"Expected one of (POSE_LANDMARKS, BODY_135, pose_keypoints_2d)"
    )


def pose_hide_legs(pose):
    """Zero out lower-body keypoints (knees, ankles, heels, foot indices)."""
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(
    pose: Pose,
    max_frames: Optional[int] = None,
    augment: bool = False,
    **augment_kwargs,
) -> torch.Tensor:
    """Apply the full SignCLIP preprocessing pipeline to a raw Pose object.

    Steps (identical to pretraining):
      1. Filter to {POSE, FACE (contours only), LEFT_HAND, RIGHT_HAND} components
      2. Shoulder-based normalisation (D=1, mid-point = origin)
      3. Zero out lower-body keypoints
      4. NaN → 0
      5. Reshape to (T, 609)
      6. Optionally truncate to max_frames

    Args:
        pose:       A pose_format.Pose object.
        max_frames: If provided, truncate sequences longer than this value.
        augment:    Whether to apply data augmentation.
        **augment_kwargs: Keyword args forwarded to apply_augmentations(),
                          e.g. sigma_temporal, p_flip, sigma_spatial, sigma_noise.

    Returns:
        Float32 tensor of shape (1, T, 609) — batch dimension included so the
        output can be fed directly to the model or indexed with [0] to get (T, 609).
    """
    pose = pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )
    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)  # (T, 609)

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()  # (1, T, 609)

    if augment:
        pose_frames = apply_augmentations(pose_frames, **augment_kwargs)

    if max_frames is not None and pose_frames.size(1) > max_frames:
        print(
            f"pose sequence length too long ({pose_frames.size(1)}) "
            f"longer than {max_frames} frames. Truncating"
        )
        pose_frames = pose_frames[:, :max_frames, :]

    return pose_frames

def apply_augmentations(pose_frames, sigma_temporal=0.2, p_flip=0.2, sigma_spatial=0.2, sigma_noise=0.001):
    """
    Applies the full SignCLIP augmentation stack sequentially to a (1, T, 609) tensor.
    """
    # ---------------------------------------------------------
    # 1. Temporal Augmentation (Stretching/Compressing Time)
    # ---------------------------------------------------------
    scale_time = random.gauss(1.0, sigma_temporal)
    scale_time = max(0.5, min(2.0, scale_time)) # Bound to prevent crashes
    
    T = pose_frames.size(1)
    new_T = int(T * scale_time)
    
    if new_T > 1 and new_T != T:
        pose_permuted = pose_frames.permute(0, 2, 1) # [1, 609, T]
        pose_interpolated = F.interpolate(
            pose_permuted, size=new_T, mode='linear', align_corners=False
        )
        pose_frames = pose_interpolated.permute(0, 2, 1) # Back to [1, new_T, 609]

    # ---------------------------------------------------------
    # 2. Spatial 2D Augmentation (Scaling the skeleton size)
    # ---------------------------------------------------------
    # We draw a random scale factor and multiply all coordinates.
    # Because the body is centered at (0,0), this perfectly zooms the skeleton in/out.
    scale_spatial = random.gauss(1.0, sigma_spatial)
    scale_spatial = max(0.5, min(1.5, scale_spatial))
    
    pose_frames = pose_frames * scale_spatial

    # ---------------------------------------------------------
    # 3. Random Horizontal Flipping (p=0.2)
    # ---------------------------------------------------------
    if random.random() < p_flip:
        # Since X=0 is the center of the body, negating the X coordinates 
        # mirrors the skeleton perfectly across the vertical axis.
        # X coordinates are at indices 0, 3, 6, 9...
        pose_frames[:, :, 0::3] = pose_frames[:, :, 0::3] * -1
        
        # *Note on Anatomical Integrity:* # This mirrors the coordinates, but leaves the indices where they are.
        # The AI will see the "Right Hand" data physically moving on the left side of the screen. 
        # This acts as a massive regularization boost, forcing the AI to learn spatial invariance.

    # ---------------------------------------------------------
    # 4. Gaussian Keypoint Noise (Jitter)
    # ---------------------------------------------------------
    # Adds a tiny, random flutter to every single coordinate to prevent exact memorization.
    noise = torch.randn_like(pose_frames) * sigma_noise
    pose_frames = pose_frames + noise

    return pose_frames
