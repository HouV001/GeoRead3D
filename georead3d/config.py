import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _path(name, default):
    value = os.environ.get(name)
    return Path(value) if value else ROOT / default


COMPAT_ROOT = _path("GEOREAD3D_COMPAT_ROOT", "datasets/3DCoMPaT200")
PARTNEXT_ROOT = _path("GEOREAD3D_PARTNEXT_ROOT", "datasets/PartNeXt")
ARTIVERSE_ROOT = _path("GEOREAD3D_ARTIVERSE_ROOT", "datasets/Artiverse")
PARTNEXT_POINTS = _path("GEOREAD3D_PARTNEXT_POINTS", "prepared/partnext")
ARTIVERSE_POINTS = _path("GEOREAD3D_ARTIVERSE_POINTS", "prepared/artiverse")
POINTLLM_REPO = _path("GEOREAD3D_POINTLLM_REPO", "third_party/PointLLM")
POINTLLM_CKPT = _path("GEOREAD3D_POINTLLM_CKPT", "third_party/PointLLM_7B_v1.2")
UTONIA_REPO = _path("GEOREAD3D_UTONIA_REPO", "third_party/Utonia")
UTONIA_CKPT = os.environ.get("GEOREAD3D_UTONIA_CKPT", "")
CKPT_DIR = _path("GEOREAD3D_CKPT_DIR", "checkpoints")
CACHE_DIR = _path("GEOREAD3D_CACHE_DIR", "cache")
OUT_DIR = _path("GEOREAD3D_OUT_DIR", "outputs")
SHIMS = ROOT / "shims"

TRAIN_SHARD = "hdf5/train_coarse_2048_10.hdf5"
GATE_SEEDS = (20261301, 20261302, 20261303)
DIRECT_SEEDS = (20261901, 20261911, 20261921)


def ckpt(name):
    return CKPT_DIR / f"{name}.pt"
