"""Kinematic chain definition and conversions for face + upper-body + hand keypoints.

Joint layout (55 joints selected from COCO-WholeBody 133):
    0  nose             1  left_eye         2  right_eye
    3  left_ear         4  right_ear
    5  left_shoulder    6  right_shoulder
    7  left_elbow       8  right_elbow
    9  left_wrist      10  right_wrist
   11  left_hip        12  right_hip
   13..33  left hand (wrist + 5 fingers × 4)
   34..54  right hand (wrist + 5 fingers × 4)

Root = mid-shoulder (average of joints 5 and 6).
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# COCO-WholeBody index selection
# ---------------------------------------------------------------------------

FACE = list(range(0, 5))          # nose, eyes, ears
UPPER_BODY = list(range(5, 13))   # shoulders, elbows, wrists, hips
LEFT_HAND = list(range(91, 112))  # 21 keypoints
RIGHT_HAND = list(range(112, 133))  # 21 keypoints
SELECTED_INDICES = FACE + UPPER_BODY + LEFT_HAND + RIGHT_HAND

NUM_JOINTS = len(SELECTED_INDICES)  # 55

# Indices of the two root-anchor joints within the local layout.
ROOT_LEFT = 5   # left shoulder
ROOT_RIGHT = 6  # right shoulder

# ---------------------------------------------------------------------------
# Kinematic parent map  (parent == -1  ⇒  child of root)
# ---------------------------------------------------------------------------

def _build_hand_parents(hand_root: int) -> list[int]:
    """21 joints: wrist + 5 fingers × 4 joints each."""
    parents = [-1]  # placeholder, overwritten by caller
    for finger in range(5):
        base = hand_root + 1 + finger * 4
        parents.append(hand_root)
        for j in range(1, 4):
            parents.append(base + j - 1)
    return parents


_LEFT_HAND_PARENTS = _build_hand_parents(13)
_LEFT_HAND_PARENTS[0] = 9    # left hand wrist  → left wrist

_RIGHT_HAND_PARENTS = _build_hand_parents(34)
_RIGHT_HAND_PARENTS[0] = 10  # right hand wrist → right wrist

PARENT_INDEX = np.array(
    [
        -1,  #  0  nose            → root (mid-shoulder)
        0,   #  1  left_eye        → nose
        0,   #  2  right_eye       → nose
        1,   #  3  left_ear        → left_eye
        2,   #  4  right_ear       → right_eye
        -1,  #  5  left_shoulder   → root (mid-shoulder)
        -1,  #  6  right_shoulder  → root (mid-shoulder)
        5,   #  7  left_elbow      → left_shoulder
        6,   #  8  right_elbow     → right_shoulder
        7,   #  9  left_wrist      → left_elbow
        8,   # 10  right_wrist     → right_elbow
        5,   # 11  left_hip        → left_shoulder
        6,   # 12  right_hip       → right_shoulder
    ]
    + _LEFT_HAND_PARENTS
    + _RIGHT_HAND_PARENTS,
    dtype=np.int64,
)

# ---------------------------------------------------------------------------
# Topological order (parents before children)
# ---------------------------------------------------------------------------

def _topo_order(parents: np.ndarray) -> np.ndarray:
    n = len(parents)
    order: list[int] = []
    visited: set[int] = set()

    def _visit(i: int) -> None:
        if i in visited or i < 0:
            return
        _visit(parents[i])
        visited.add(i)
        order.append(i)

    for i in range(n):
        _visit(i)
    return np.array(order, dtype=np.int64)


TOPO_ORDER = _topo_order(PARENT_INDEX)

# ---------------------------------------------------------------------------
# Skeleton connections for rendering (independent of kinematic parents)
# ---------------------------------------------------------------------------

FACE_CONNECTIONS = [
    (0, 1), (0, 2),   # nose — eyes
    (1, 3), (2, 4),   # eyes — ears
]

BODY_CONNECTIONS = [
    (5, 6),            # shoulder — shoulder
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), # shoulders — hips
    (11, 12),          # hip — hip
]

_HAND_CONNECTIONS_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

LEFT_HAND_CONNECTIONS = [(i + 13, j + 13) for i, j in _HAND_CONNECTIONS_LOCAL]
RIGHT_HAND_CONNECTIONS = [(i + 34, j + 34) for i, j in _HAND_CONNECTIONS_LOCAL]

SKELETON_COLORS = {
    "face": "#ffff00",
    "body": "#00ff00",
    "left_hand": "#ff4444",
    "right_hand": "#4488ff",
}

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def positions_to_offsets(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute positions → (root, kinematic offsets).

    Args:
        positions: (..., 55, 2) absolute normalised (y, x).

    Returns:
        root:    (..., 2)     mid-shoulder absolute position.
        offsets: (..., 55, 2) offset of each joint from its parent.
    """
    root = (positions[..., ROOT_LEFT, :] + positions[..., ROOT_RIGHT, :]) / 2.0

    offsets = np.empty_like(positions)
    for i in range(NUM_JOINTS):
        parent = PARENT_INDEX[i]
        if parent == -1:
            offsets[..., i, :] = positions[..., i, :] - root
        else:
            offsets[..., i, :] = positions[..., i, :] - positions[..., parent, :]

    return root, offsets


def offsets_to_positions(
    root: torch.Tensor | np.ndarray,
    offsets: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """(root, kinematic offsets) → absolute positions.

    Args:
        root:    (..., 2)     mid-shoulder absolute position.
        offsets: (..., 55, 2) kinematic offsets.

    Returns:
        positions: (..., 55, 2) absolute normalised (y, x).
    """
    is_torch = isinstance(offsets, torch.Tensor)
    positions = torch.empty_like(offsets) if is_torch else np.empty_like(offsets)

    for i in TOPO_ORDER:
        parent = PARENT_INDEX[i]
        if parent == -1:
            positions[..., i, :] = root + offsets[..., i, :]
        else:
            positions[..., i, :] = positions[..., parent, :] + offsets[..., i, :]

    return positions