import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config, testsets
from georead3d.solver import pointing_error, score_reorientation, true_direction

POINTING_DEG = 22.5
ANCHOR_DEG = 30.0
WHOLE_DEG = 30.0
SEVERE_DEG = 90.0
METRICS = ("pointing", "reorientation", "whole", "severe")


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["oid"], r["pid"], r["pose"])] = r
    return out


def rates(dataset, preds, route):
    B = testsets.build(dataset)
    SH = testsets.shapes(dataset)
    skip = testsets.excluded(dataset)
    tasks = {(t["oid"], t["A"], t["B"], t["pose"]): t for t in B["tasks"]}

    def vec(oid, pid, pose):
        if pid is None:
            return np.zeros(3)
        r = preds.get((oid, pid, pose))
        if r is None or route not in r:
            return None
        return np.array(r[route])

    pointing, severe, anchor, whole = [], [], [], []
    for u in B["pointing"]:
        if u["pose"] != "rot" or u["oid"] in skip:
            continue
        q = vec(u["oid"], u["sel"], u["pose"])
        if q is None:
            continue
        e = pointing_error(q, true_direction(SH[u["oid"]], u["pid"], u["pose"]))
        pointing.append(float(e <= POINTING_DEG))
        severe.append(float(e > SEVERE_DEG))
    for u in B["reorientation"]:
        if u["pose"] != "rot" or u["oid"] in skip:
            continue
        va, vb = vec(u["oid"], u["selA"], u["pose"]), vec(u["oid"], u["selB"], u["pose"])
        if va is None or vb is None:
            continue
        joint, geo = score_reorientation(tasks[(u["oid"], u["A"], u["B"], u["pose"])], SH[u["oid"]], va, vb)
        anchor.append(float(joint <= ANCHOR_DEG))
        whole.append(float(geo <= WHOLE_DEG))
    return {"pointing": 100 * float(np.mean(pointing)), "reorientation": 100 * float(np.mean(anchor)), "whole": 100 * float(np.mean(whole)),
            "severe": 100 * float(np.mean(severe)), "n_pointing": len(pointing), "n_reorientation": len(anchor)}


def summary(dataset, preds, seeds=(1, 2, 3)):
    per = [rates(dataset, preds, f"georead3d{k}") for k in seeds]
    out = {"n_pointing": per[0]["n_pointing"], "n_reorientation": per[0]["n_reorientation"]}
    for m in METRICS:
        v = [p[m] for p in per]
        out[m] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "seeds": v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=testsets.DATASETS, required=True)
    ap.add_argument("--predictions", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    path = Path(a.predictions) if a.predictions else config.OUT_DIR / f"{a.dataset}_predictions.jsonl"
    res = summary(a.dataset, load(path))
    print(f"{a.dataset}: {res['n_pointing']} pointing units, {res['n_reorientation']} reorientation units")
    for m in METRICS:
        r = res[m]
        print(f"  {m:14s} {r['mean']:5.1f} ± {r['sd']:.1f}   seeds " + ", ".join(f"{x:.3f}" for x in r["seeds"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
