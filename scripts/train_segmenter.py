import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config, pointbert
from georead3d.common import random_rotation, save_state, seed_all
from georead3d.corpus import STREAM_SEED, Corpus
from georead3d.heads import SegHead
from georead3d.taxonomy import NPART

FULL_SEED = 20260929
EPOCHS = 10
BATCH = 32
LR = 1e-3


def soft_targets(L, ix, idx):
    lab = torch.from_numpy(L[ix].astype(np.int64)).cuda()
    gl = torch.gather(lab[:, None, :].expand(-1, idx.shape[1], -1), 2, idx)
    return Fn.one_hot(gl, NPART).float().mean(2)


def train_full(data, enc, max_steps):
    seed_all(FULL_SEED)
    head = SegHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    n = len(data.X)
    steps_ep = math.ceil(n / BATCH)
    total = steps_ep * EPOCHS
    step = 0
    for ep in range(EPOCHS):
        g = np.random.default_rng([FULL_SEED, ep])
        order = g.permutation(n)
        tot = 0.
        seen = 0
        for b in range(steps_ep):
            ix = order[b * BATCH:(b + 1) * BATCH]
            xyz = data.X[ix].copy()
            for j in range(len(ix)):
                if g.random() < 0.5:
                    xyz[j] = (xyz[j] @ random_rotation(g)).astype(np.float32)
            seed_all(FULL_SEED + 1000 * ep + b)
            tok, idx, _ = pointbert.encode(enc, xyz, data.R[ix])
            tgt = soft_targets(data.L, ix, idx)
            for q in opt.param_groups:
                q["lr"] = LR * 0.5 * (1 + math.cos(math.pi * min((ep * steps_ep + b) / total, 1.0)))
            loss = -(tgt * Fn.log_softmax(head(tok, torch.from_numpy(data.C[ix]).cuda()), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(ix)
            seen += len(ix)
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"full step {step}/{total} loss {float(loss):.6f}", flush=True)
            if max_steps and step >= max_steps:
                return head
        print(f"full epoch {ep + 1}/{EPOCHS} soft-CE {tot / seen:.4f}", flush=True)
    return head


def train_fold(data, enc, k, max_steps):
    rows = np.where(data.folds == k)[0]
    seed_all(STREAM_SEED + k)
    head = SegHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    steps_ep = math.ceil(len(rows) / BATCH)
    total = steps_ep * EPOCHS
    step = 0
    for ep in range(EPOCHS):
        g = np.random.default_rng([STREAM_SEED, k, ep])
        order = rows[g.permutation(len(rows))]
        for b in range(steps_ep):
            ix = np.sort(order[b * BATCH:(b + 1) * BATCH])
            xyz = data.X[ix].copy()
            for j in range(len(ix)):
                if g.random() < 0.5:
                    xyz[j] = (xyz[j] @ random_rotation(g)).astype(np.float32)
            seed_all(STREAM_SEED + 10000 * k + 1000 * ep + b)
            tok, idx, _ = pointbert.encode(enc, xyz, data.R[ix])
            tgt = soft_targets(data.L, ix, idx)
            for q in opt.param_groups:
                q["lr"] = LR * 0.5 * (1 + math.cos(math.pi * min((ep * steps_ep + b) / total, 1)))
            loss = -(tgt * Fn.log_softmax(head(tok, torch.from_numpy(data.C[ix]).cuda()), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"fold{k} step {step}/{total} loss {float(loss):.6f}", flush=True)
            if max_steps and step >= max_steps:
                return head
    head.eval()
    other = np.where(data.folds != k)[0][:500]
    correct = total_pts = 0
    with torch.no_grad():
        for b in range(0, len(other), BATCH):
            ix = other[b:b + BATCH]
            seed_all(STREAM_SEED + b)
            tok, idx, _ = pointbert.encode(enc, data.X[ix], data.R[ix])
            Pg = Fn.softmax(head(tok, torch.from_numpy(data.C[ix]).cuda()), -1)
            pts = torch.zeros(len(ix), data.X.shape[1], NPART, device="cuda").scatter_add(
                1, idx.reshape(len(ix), -1, 1).expand(-1, -1, NPART), Pg.repeat_interleave(idx.shape[2], 1))
            correct += int((pts.argmax(-1).cpu().numpy() == data.L[ix]).sum())
            total_pts += data.L[ix].size
    print(f"fold{k} out-of-fold point accuracy {correct / total_pts:.3f}", flush=True)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("full", "fold0", "fold1"), required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = Corpus()
    enc = pointbert.build_encoder()
    t0 = time.time()
    if a.which == "full":
        head = train_full(data, enc, a.max_steps)
        name = "segmenter"
    else:
        k = int(a.which[-1])
        head = train_fold(data, enc, k, a.max_steps)
        name = f"segmenter_fold{k}"
    out = Path(a.out) if a.out else config.ckpt(name)
    save_state(out, head)
    print(f"{name}: {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
