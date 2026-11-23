import numpy as np

from .common import random_rotation

POSE_SEED = 20261210
NUM_PASSES = 4


def pass_rotations(row, pose):
    rng = np.random.default_rng([POSE_SEED, int(row), 0 if pose == "canon" else 1])
    return [random_rotation(rng) for _ in range(NUM_PASSES - 1)]


def passes(item):
    return ([(0, item["xyz"], None)] + [(k + 1, (item["xyz"] @ R).astype(np.float32), R) for k, R in enumerate(item["rotations"])])[:NUM_PASSES]
