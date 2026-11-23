import json
from functools import lru_cache

import numpy as np

from . import config, taxonomy
from .common import normalise_points
from .poses import pass_rotations

DATASETS = ("partnext", "artiverse")


@lru_cache(maxsize=2)
def build(dataset):
    return json.load(open(config.DATA / dataset / "build.json"))


def excluded(dataset):
    return set(json.load(open(config.DATA / dataset / "excluded.json"))["excluded"])


def shapes(dataset):
    return {s["oid"]: s for s in build(dataset)["shapes"]}


def point_file(dataset, source):
    if dataset == "partnext":
        return config.PARTNEXT_POINTS / f"{source[1]}.npz"
    return config.ARTIVERSE_POINTS / f"{source[0]}__{source[1]}__{source[2]}.npz"


def queried_parts(dataset):
    B = build(dataset)
    need = {}
    for s in B["shapes"]:
        for pose in ("canon", "rot"):
            need[(s["oid"], pose)] = set(p["pid"] for p in s["parts"])
    for u in B["reorientation"]:
        need.setdefault((u["oid"], u["pose"]), set()).update([u["A"], u["B"]] + [q for q in (u["selA"], u["selB"]) if q is not None])
    for u in B["pointing"]:
        need.setdefault((u["oid"], u["pose"]), set()).update([u["pid"]] + ([u["sel"]] if u["sel"] is not None else []))
    return {k: sorted(v) for k, v in need.items()}


def keys(dataset):
    return sorted(queried_parts(dataset))


def items(dataset, keys_=None):
    B = build(dataset)
    SH = shapes(dataset)
    need = queried_parts(dataset)
    frame = np.array(B["frame"])
    cache = {}
    for oid, pose in (keys_ if keys_ is not None else keys(dataset)):
        s = SH[oid]
        if oid not in cache:
            z = np.load(point_file(dataset, s["source"]))
            cache = {oid: (normalise_points(z["xyz"].astype(np.float64) @ frame), np.asarray(z["rgb"], np.float32))}
        x0, rgb = cache[oid]
        xyz = x0 if pose == "canon" else (x0 @ np.array(s["Q"])).astype(np.float32)
        yield {"oid": oid, "pose": pose, "row": int(s["row"]), "cat": s["cat"], "cat_id": taxonomy.category_id(s["cat"]),
               "xyz": xyz.astype(np.float32), "rgb": rgb, "pids": need[(oid, pose)], "fps_seed": s["fps_seed"][pose],
               "rotations": pass_rotations(s["row"], pose)}
