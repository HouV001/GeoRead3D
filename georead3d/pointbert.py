import sys

import numpy as np
import torch

from . import config

if str(config.POINTLLM_REPO) not in sys.path:
    sys.path.insert(0, str(config.POINTLLM_REPO))

from pointllm.model.pointbert import dvae
from pointllm.model.pointbert.point_encoder import PointTransformer
from pointllm.utils import cfg_from_yaml_file

LAST_GROUPING = {}
YAML = "pointllm/model/pointbert/PointTransformer_8192point_2layer.yaml"


def _fps_index(pos, n):
    b, npts, _ = pos.shape
    dev = pos.device
    c = torch.zeros(b, n, dtype=torch.long, device=dev)
    dist = torch.ones(b, npts, device=dev) * 1e10
    far = torch.randint(0, npts, (b,), dtype=torch.long, device=dev)
    bi = torch.arange(b, device=dev)
    for i in range(n):
        c[:, i] = far
        cc = pos[bi, far, :].view(b, 1, 3)
        dist = torch.min(dist, torch.sum((pos - cc) ** 2, -1))
        far = torch.max(dist, -1)[1]
    return c


def _group_forward(self, xyz):
    b, n, ch = xyz.shape
    pos = xyz[:, :, :3] if ch > 3 else xyz
    rgb = xyz[:, :, 3:] if ch > 3 else None
    cidx = _fps_index(pos, self.num_group)
    center = pos.gather(1, cidx.unsqueeze(-1).expand(-1, -1, 3))
    idx = dvae.knn_point(self.group_size, pos, center)
    LAST_GROUPING["cidx"] = cidx.detach()
    LAST_GROUPING["idx"] = idx.detach()
    ib = torch.arange(b, device=pos.device).view(-1, 1, 1) * n
    flat = (idx + ib).view(-1)
    nx = pos.reshape(b * n, -1)[flat, :].view(b, self.num_group, self.group_size, 3).contiguous() - center.unsqueeze(2)
    if rgb is not None:
        nr = rgb.reshape(b * n, -1)[flat, :].view(b, self.num_group, self.group_size, -1).contiguous()
        return torch.cat((nx, nr), -1), center
    return nx, center


dvae.Group.forward = _group_forward


def build_encoder(weights=None):
    cfg = cfg_from_yaml_file(str(config.POINTLLM_REPO / YAML))
    cfg.model.point_dims = 6
    enc = PointTransformer(cfg.model, use_max_pool=getattr(cfg.model, "use_max_pool", False))
    state = torch.load(weights or config.ckpt("pointbert_encoder"), map_location="cpu", weights_only=False)
    enc.load_state_dict(state["model"], strict=True)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.cuda().to(torch.bfloat16).eval()


@torch.no_grad()
def encode(enc, xyz, rgb):
    pc = torch.from_numpy(np.concatenate([xyz, rgb], -1)).cuda().to(torch.bfloat16)
    tok = enc(pc)
    return tok, LAST_GROUPING["idx"], LAST_GROUPING["cidx"]


def group_inputs(xyz, rgb01, knn, cidx):
    from .common import lab_colour
    b = torch.arange(xyz.shape[0], device=xyz.device)[:, None]
    center = xyz[b, cidx]
    col = rgb01[b[:, :, None], knn].mean(2)
    return center, lab_colour(col)
