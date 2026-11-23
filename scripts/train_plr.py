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
from georead3d.common import load_state, save_state, seed_all
from georead3d.corpus import STREAM_SEED, Corpus
from georead3d.heads import Refiner, SegHead, group_readout, rel_sq_error

STEPS = 4000
LR = 1e-3


def out_of_fold(heads, tok, cats, folds):
    P0 = Fn.softmax(heads[0](tok, cats), -1)
    P1 = Fn.softmax(heads[1](tok, cats), -1)
    return torch.where(torch.as_tensor(folds == 0, device=tok.device)[:, None, None], P1, P0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = Corpus()
    enc = pointbert.build_encoder()
    heads = []
    for k in (0, 1):
        h = SegHead().cuda()
        h.load_state_dict(load_state(config.ckpt(f"segmenter_fold{k}")))
        h.eval()
        heads.append(h)
    seed_all(STREAM_SEED)
    ref = Refiner().cuda()
    opt = torch.optim.AdamW(ref.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    steps = a.max_steps or STEPS
    for step in range(steps):
        ix, xyz, samples = data.stream(step)
        seed_all(STREAM_SEED + step)
        with torch.no_grad():
            tok, idx, cidx = pointbert.encode(enc, xyz, data.R[ix])
            P = out_of_fold(heads, tok, torch.from_numpy(data.C[ix]).cuda(), data.folds[ix])
            Xt = torch.from_numpy(xyz).cuda().float()
            center, col = pointbert.group_inputs(Xt, torch.from_numpy(data.R[ix]).cuda(), idx, cidx)
        j = torch.tensor([s[0] for s in samples], device="cuda")
        pid = torch.tensor([s[1] for s in samples], device="cuda")
        t = torch.tensor(np.stack([s[2] for s in samples]), device="cuda").float()
        z = ref(tok[j, 1:].float(), center[j], col[j], P[j], pid)
        loss = rel_sq_error(group_readout(torch.softmax(z, -1), idx[j], Xt[j]), t).mean()
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(step / STEPS, 1)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or a.max_steps:
            print(f"step {step}/{STEPS} loss {float(loss):.6f} parts {len(samples)}", flush=True)
    out = Path(a.out) if a.out else config.ckpt("plr")
    save_state(out, ref, step=steps)
    print(f"plr: {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
