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
from georead3d.common import lab_colour, save_state, seed_all
from georead3d.corpus import STREAM_SEED, Corpus
from georead3d.heads import GATE_ARCH, QueryGate, centres, membership, point_readout, rel_sq_error, split_bce

STEPS = 4000
LR = 1e-3
BCE_WEIGHT = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = Corpus()
    enc = pointbert.build_encoder()
    torch.manual_seed(a.seed)
    head = QueryGate(**GATE_ARCH).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    steps = a.max_steps or STEPS
    for step in range(steps):
        ix, xyz, samples = data.stream(step)
        seed_all(STREAM_SEED + step)
        with torch.no_grad():
            tok, idx, cidx = pointbert.encode(enc, xyz, data.R[ix])
            Xt = torch.from_numpy(xyz).cuda().float()
            Mn = membership(idx, Xt, cidx)
            cen = centres(Xt, cidx)
            colp = lab_colour(torch.from_numpy(data.R[ix]).cuda())
            lab = torch.from_numpy(data.L[ix].astype(np.int64)).cuda()
            cat = torch.from_numpy(data.C[ix]).cuda()
        j = torch.tensor([s[0] for s in samples], device="cuda")
        pid = torch.tensor([s[1] for s in samples], device="cuda")
        t = torch.tensor(np.stack([s[2] for s in samples]), device="cuda").float()
        y = (lab[j] == pid[:, None]).float()
        logit = head(tok[j], pid, cat[j], Mn[j], cen[j], Xt[j], colp[j])
        lpt = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none")
        lro = rel_sq_error(point_readout(torch.softmax(logit, -1), Xt[j]), t).mean()
        lbce = split_bce(lpt, y).mean()
        loss = lro + BCE_WEIGHT * lbce
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(step / STEPS, 1)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or a.max_steps:
            print(f"step {step}/{STEPS} readout {float(lro):.6f} bce {float(lbce):.6f}", flush=True)
    out = Path(a.out) if a.out else config.ckpt(f"gate_{a.seed}")
    save_state(out, head, step=steps)
    print(f"gate {a.seed}: {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
