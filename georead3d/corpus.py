import zlib

import h5py
import numpy as np

from . import config
from .common import normalise_points, random_rotation

TAU = 0.05
STREAM_SEED = 20261013
NUM_VAL = 1000
MAX_PARTS = 4
BATCH = 32


def load_shard(path, compositions=1):
    with h5py.File(path, "r") as h:
        sid = np.array([s.decode() for s in h["shape_id"][:]])
        keep = []
        seen = {}
        for i, s in enumerate(sid):
            if s == "":
                continue
            k = seen.get(s, 0)
            if k < compositions:
                keep.append(i)
                seen[s] = k + 1
        keep = np.array(sorted(keep))
        if len(keep) and keep[-1] - keep[0] == len(keep) - 1:
            sl = slice(int(keep[0]), int(keep[-1]) + 1)
            pts = h["points"][sl]
            plab = h["points_part_labels"][sl]
            slab = h["shape_label"][sl].astype(np.int64)
        else:
            pts = h["points"][keep]
            plab = h["points_part_labels"][keep]
            slab = h["shape_label"][keep].astype(np.int64)
        sid = sid[keep]
    xyz = np.stack([normalise_points(p[:, :3]) for p in pts])
    rgb = (pts[:, :, 3:] / 255.).astype(np.float32)
    return xyz, rgb, plab, sid, slab


def eligible_parts(xyz, plab):
    out = []
    for i in range(len(xyz)):
        x = xyz[i].astype(np.float64)
        lab = plab[i]
        e = []
        for pid in np.unique(lab):
            m = lab == pid
            d = x[m].mean(0) - x.mean(0)
            if np.linalg.norm(d) >= TAU:
                e.append((int(pid), d))
        out.append(e)
    return out


def fold_of(sid):
    return zlib.crc32(sid.encode()) % 2


class Corpus:
    def __init__(self):
        self.X, self.R, self.L, self.S, self.C = load_shard(config.COMPAT_ROOT / config.TRAIN_SHARD, 1)
        self.E = eligible_parts(self.X, self.L)
        self.folds = np.array([fold_of(s) for s in self.S])
        rng = np.random.default_rng(STREAM_SEED)
        self.val = np.zeros(len(self.S), bool)
        self.val[rng.choice(len(self.S), NUM_VAL, replace=False)] = True
        self.pool = np.where(~self.val & np.array([len(e) > 0 for e in self.E]))[0]

    def views(self, ix, g, rotate):
        xyz = self.X[ix].copy()
        samples = []
        for j, i in enumerate(ix):
            if rotate(g):
                xyz[j] = (xyz[j] @ random_rotation(g)).astype(np.float32)
            parts = [p for p, _ in self.E[i]]
            parts = parts if len(parts) <= MAX_PARTS else list(g.choice(parts, MAX_PARTS, replace=False))
            for p in parts:
                m = self.L[i] == p
                samples.append((j, int(p), xyz[j][m].mean(0) - xyz[j].mean(0)))
        return xyz, samples

    def stream(self, step):
        g = np.random.default_rng([STREAM_SEED, step])
        ix = np.sort(g.choice(self.pool, BATCH, replace=False))
        xyz, samples = self.views(ix, g, lambda g_: g_.random() < 0.5)
        return ix, xyz, samples
