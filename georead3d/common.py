import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise_points(x):
    x = x.astype(np.float64)
    x = x - x.mean(0)
    x /= np.linalg.norm(x, axis=1).max()
    return x.astype(np.float32)


def normalise_points64(x):
    x = x.astype(np.float64)
    x = x - x.mean(0)
    return x / np.linalg.norm(x, axis=1).max()


def random_rotation(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else np.c_[-q[:, 0], q[:, 1:]]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def lab_colour(rgb01):
    c = torch.where(rgb01 > 0.04045, ((rgb01 + 0.055) / 1.055) ** 2.4, rgb01 / 12.92)
    m = torch.tensor([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], device=rgb01.device)
    x = (c @ m) / torch.tensor([0.95047, 1.0, 1.08883], device=rgb01.device)
    f = torch.where(x > 0.008856, x.clamp_min(1e-12) ** (1 / 3), 7.787 * x + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1) / 100.


def estimate_normals(xyz):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=300))
    return np.asarray(pc.normals, dtype=np.float32)


def load_state(path, device="cuda"):
    return torch.load(path, map_location=device, weights_only=False)["model"]


def save_state(path, model, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), **extra}, tmp)
    tmp.replace(path)
