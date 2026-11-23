import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config
from georead3d.common import random_rotation
from georead3d.corpus import Corpus
from georead3d.utonia_features import FEATURE_DIM, UtoniaFeatures

POSE_SEED = 20261801
ENTRY_OFFSET = 10
BATCH = 32
N = 2048


def entries(data):
    out = []
    for r in data.pool:
        g = np.random.default_rng([POSE_SEED, int(r)])
        out.append({"row": int(r), "k": 0, "R": np.eye(3).tolist()})
        out += [{"row": int(r), "k": k, "R": random_rotation(g).tolist()} for k in (1, 2)]
    for i, e in enumerate(out):
        e["id"] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else config.CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Corpus()
    ent = entries(data)
    json.dump(ent, open(out_dir / "utonia_train_index.json", "w"))
    if a.limit:
        ent = ent[:a.limit]
    path = out_dir / "utonia_train.npy"
    n = len(ent)
    resume = path.exists() and np.load(path, mmap_mode="r").shape == (n, N, FEATURE_DIM)
    mm = np.lib.format.open_memmap(path, mode="r+" if resume else "w+", dtype=np.float16, shape=None if resume else (n, N, FEATURE_DIM))
    feats = UtoniaFeatures()
    t0 = time.time()
    for b0 in range(0, n, BATCH):
        eb = ent[b0:b0 + BATCH]
        if resume and all(np.abs(mm[e["id"], :4]).sum() > 0 for e in (eb[0], eb[-1])):
            continue
        clouds = [(data.X[e["row"]] @ np.array(e["R"])).astype(np.float32) for e in eb]
        F = feats(clouds, [data.R[e["row"]] for e in eb], [(e["id"] * 7919 + ENTRY_OFFSET) % 2 ** 31 for e in eb],
                  (20263000 + 100000 * ENTRY_OFFSET + b0 // BATCH) % 2 ** 31)
        mm[[e["id"] for e in eb]] = np.stack([f.half().cpu().numpy() for f in F])
        if (b0 // BATCH) % 200 == 0:
            print(f"{b0 + len(eb)}/{n} entries, {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    print(f"{n} entries -> {path}", flush=True)


if __name__ == "__main__":
    main()
