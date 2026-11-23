import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config, testsets
from georead3d.model import GeoRead3D

CHUNK = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=testsets.DATASETS, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out) if a.out else config.OUT_DIR / f"{a.dataset}_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = testsets.keys(a.dataset)
    if a.limit:
        keys = keys[:a.limit]
    model = GeoRead3D()
    t0 = time.time()
    chunk = []
    done = 0
    with open(out, "w") as fo:
        for item in testsets.items(a.dataset, keys):
            chunk.append(item)
            if len(chunk) == CHUNK:
                for r in model.run(chunk):
                    fo.write(json.dumps(r) + "\n")
                done += len(chunk)
                chunk = []
                if (done // CHUNK) % 50 == 0:
                    print(f"{done}/{len(keys)} shape-poses, {time.time() - t0:.0f}s", flush=True)
        if chunk:
            for r in model.run(chunk):
                fo.write(json.dumps(r) + "\n")
            done += len(chunk)
    print(f"{done} shape-poses in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
