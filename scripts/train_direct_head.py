import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config
from georead3d.common import lab_colour, save_state
from georead3d.corpus import Corpus
from georead3d.heads import EvidenceHead, balanced_bce, point_readout, rel_sq_error
from georead3d.losses import direction_loss, kernel_quad, kernels, normalise, pair_index, slots
from georead3d.utonia_features import FEATURE_DIM

STREAM_SEED = 20261902
STEPS = 4000
BATCH = 32
MAX_PARTS = 4
LR = 1e-3
KERNEL_LENGTH = 0.2
DEFORMATION_SIGMA = 0.4
DIRECTION_WEIGHT = 0.3
RAMP = (1000, 1500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in config.DIRECT_SEEDS))
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cache = Path(a.cache_dir) if a.cache_dir else config.CACHE_DIR
    data = Corpus()
    index = json.load(open(cache / "utonia_train_index.json"))
    key = {(e["row"], e["k"]): e["id"] for e in index}
    Rm = {(e["row"], e["k"]): np.array(e["R"], np.float32) for e in index}
    rows = np.array(sorted({e["row"] for e in index}))
    store = np.load(cache / "utonia_train.npy", mmap_mode="r")
    readers = ThreadPoolExecutor(16)

    def draw_batch(step):
        g = np.random.default_rng([STREAM_SEED, step])
        ix = np.sort(g.choice(rows, BATCH, replace=False))
        ks = [0 if g.random() < 0.5 else int(g.integers(1, 3)) for _ in ix]
        return g, ix, ks

    def read(step):
        _, ix, ks = draw_batch(step)
        return np.stack(list(readers.map(lambda ik: np.array(store[key[(int(ik[0]), ik[1])]]), zip(ix, ks))))

    steps = a.max_steps or STEPS
    prefetch = ThreadPoolExecutor(1)
    pending = prefetch.submit(read, 0)
    heads = {}
    for sd in seeds:
        torch.manual_seed(sd)
        h = EvidenceHead(din=FEATURE_DIM).cuda()
        heads[sd] = (h, torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=0.01))
    t0 = time.time()
    for step in range(steps):
        g, ix, ks = draw_batch(step)
        samples = []
        for j, (i, k) in enumerate(zip(ix, ks)):
            parts = [p for p, _ in data.E[i]]
            parts = parts if len(parts) <= MAX_PARTS else list(g.choice(parts, MAX_PARTS, replace=False))
            for p in parts:
                d = next(dd for pp, dd in data.E[i] if pp == p)
                samples.append((j, int(p), (np.asarray(d, np.float64) @ Rm[(int(i), k)]).astype(np.float32)))
        with torch.no_grad():
            Xt = torch.from_numpy(np.stack([data.X[i] @ Rm[(int(i), k)] for i, k in zip(ix, ks)])).cuda().float()
            arr = pending.result()
            if step + 1 < steps:
                pending = prefetch.submit(read, step + 1)
            feat = torch.from_numpy(arr.astype(np.float32)).cuda()
            colp = lab_colour(torch.from_numpy(data.R[ix]).cuda())
            lab = torch.from_numpy(data.L[ix].astype(np.int64)).cuda()
            cat = torch.from_numpy(data.C[ix]).cuda()
            j = torch.tensor([s_[0] for s_ in samples], device="cuda")
            pid = torch.tensor([s_[1] for s_ in samples], device="cuda")
            t = torch.tensor(np.stack([s_[2] for s_ in samples]), device="cuda")
            y = (lab[j] == pid[:, None]).float()
            u = y / y.sum(-1, keepdim=True).clamp_min(1)
            Xn, rms = normalise(Xt)
            Kb = kernels(Xn, [KERNEL_LENGTH])
            sl = slots(j)
            tn2 = (t.pow(2).sum(-1) / rms[j].pow(2)).clamp_min(1e-12)
            pairs = pair_index(j, t)
        lr = LR * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(step / STEPS, 1)))
        ramp = min(1.0, max(0.0, (step - RAMP[0]) / (RAMP[1] - RAMP[0])))
        for sd, (h, opt) in heads.items():
            logit = h(feat[j], pid, cat[j], colp[j])
            w = torch.softmax(logit, -1)
            r = point_readout(w, Xt[j])
            comp = {"grs": rel_sq_error(r, t).mean(), "bce": balanced_bce(logit, y).mean(),
                    "kernel": (3 * DEFORMATION_SIGMA * DEFORMATION_SIGMA * kernel_quad(w - u, Kb[KERNEL_LENGTH], j, sl, MAX_PARTS) / tn2).mean()}
            mu = DIRECTION_WEIGHT * ramp
            if len(pairs) and mu > 0:
                comp["direction"] = mu * ((direction_loss(r[pairs[:, 0]], t[pairs[:, 0]]) + direction_loss(r[pairs[:, 1]], t[pairs[:, 1]])) / 2).mean()
            loss = sum(comp.values())
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
            opt.step()
            if step % 100 == 0 or a.max_steps:
                print(f"seed {sd} step {step}/{STEPS} " + " ".join(f"{k} {float(v):.6f}" for k, v in comp.items()), flush=True)
    out_dir = Path(a.out_dir) if a.out_dir else config.CKPT_DIR
    for sd, (h, _) in heads.items():
        save_state(out_dir / f"direct_{sd}.pt", h, step=steps)
    print(f"{len(heads)} heads: {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
