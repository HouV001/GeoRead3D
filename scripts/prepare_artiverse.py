import argparse
import io
import json
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config, taxonomy, testsets
from georead3d.common import normalise_points64
from georead3d.corpus import TAU

SAMPLE_SEED = 20263302
N_POINTS = 2048
MAPPING = json.load(open(config.DATA / "artiverse/mapping.json"))
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(f):
    b = open(f, "rb").read()
    assert b[:4] == b"glTF"
    clen = struct.unpack("<I", b[12:16])[0]
    J = json.loads(b[20:20 + clen])
    off = 20 + clen
    BIN = b""
    if off < len(b):
        blen = struct.unpack("<I", b[off:off + 4])[0]
        BIN = b[off + 8:off + 8 + blen]
    return J, BIN


def accessor(J, BIN, i):
    a = J["accessors"][i]
    bv = J["bufferViews"][a["bufferView"]]
    dt = np.dtype(CT[a["componentType"]])
    n = NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * n
    raw = np.frombuffer(BIN, dtype=np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * n, offset=start)
    out = np.lib.stride_tricks.as_strided(raw, shape=(a["count"], dt.itemsize * n), strides=(stride, 1)).copy().view(dt).reshape(a["count"], n)
    if a.get("normalized") and dt.kind in "iu":
        out = out.astype(np.float64) / np.iinfo(dt).max
    return out


def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    T = np.eye(4)
    t = n.get("translation", [0, 0, 0])
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    s = n.get("scale", [1, 1, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, :3] = R * np.array(s)[None, :]
    T[:3, 3] = t
    return T


def image(J, BIN, ti, cache):
    if ti in cache:
        return cache[ti]
    from PIL import Image
    src = J["textures"][ti].get("source")
    im = None
    if src is not None and "bufferView" in J["images"][src]:
        bv = J["bufferViews"][J["images"][src]["bufferView"]]
        o = bv.get("byteOffset", 0)
        im = np.asarray(Image.open(io.BytesIO(BIN[o:o + bv["byteLength"]])).convert("RGB"), dtype=np.float64) / 255.0
    cache[ti] = im
    return im


def submeshes(f):
    J, BIN = read_glb(f)
    nodes = J.get("nodes", [])
    sc = J["scenes"][J.get("scene", 0)]
    out = []

    def norm_label(x):
        return None if x is None else str(x).strip().lower().replace(" ", "_")

    def walk(i, M, pid):
        n = nodes[i]
        e = n.get("extras") or {}
        M = M @ node_matrix(n)
        if e.get("pid") is not None:
            pid = int(e["pid"])
        elif e.get("belongToPart") is not None:
            pid = int(e["belongToPart"])
        if "mesh" in n:
            for p in J["meshes"][n["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                V = accessor(J, BIN, p["attributes"]["POSITION"]).astype(np.float64)
                V = V @ M[:3, :3].T + M[:3, 3]
                F = accessor(J, BIN, p["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in p else np.arange(len(V)).reshape(-1, 3)
                mat = J["materials"][p["material"]] if "material" in p and J.get("materials") else {}
                pbr = mat.get("pbrMetallicRoughness", {})
                fac = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float64)
                tex = pbr.get("baseColorTexture")
                uv = None
                ti = None
                if tex is not None and f"TEXCOORD_{tex.get('texCoord', 0)}" in p["attributes"]:
                    uv = accessor(J, BIN, p["attributes"][f"TEXCOORD_{tex.get('texCoord', 0)}"]).astype(np.float64)
                    ti = tex["index"]
                vc = accessor(J, BIN, p["attributes"]["COLOR_0"]).astype(np.float64)[:, :3] if "COLOR_0" in p["attributes"] else None
                out.append((V, F, uv, (fac, ti, vc, "baseColorFactor" in pbr), pid))
        for c in n.get("children", []) or []:
            walk(c, M, pid)

    for r in sc["nodes"]:
        walk(r, np.eye(4), None)
    labels = {}
    for p in ((sc.get("extras") or {}).get("partsConnectivity") or {}).get("parts", []) or []:
        if p and p.get("pid") is not None:
            labels[int(p["pid"])] = norm_label(p.get("label") or p.get("name"))
    for n in nodes:
        e = n.get("extras") or {}
        if e.get("pid") is not None and int(e["pid"]) not in labels:
            labels[int(e["pid"])] = norm_label(e.get("label") or e.get("name"))
    fg = (J.get("extras") or {}).get("flattened_graph") or (sc.get("extras") or {}).get("flattened_graph") or {}
    parent = {int(k): int(v.get("parent", -1)) for k, v in fg.items()}
    return out, labels, parent, J, BIN


def path_of(pid, labels, parent):
    path, seen = [], set()
    while pid is not None and pid != -1 and pid not in seen:
        seen.add(pid)
        path.append(labels.get(pid))
        pid = parent.get(pid, -1)
    return [x for x in reversed(path) if x is not None]


def colours(sm, fi, bary, J, BIN, cache):
    V, F, uv, (fac, ti, vc, has_fac), pid = sm
    if uv is not None and ti is not None:
        im = image(J, BIN, ti, cache)
        if im is not None:
            u = (bary[:, :, None] * uv[F[fi]]).sum(1)
            h, w = im.shape[:2]
            x = np.floor((u[:, 0] % 1.0) * w).astype(int).clip(0, w - 1)
            y = np.floor((u[:, 1] % 1.0) * h).astype(int).clip(0, h - 1)
            return np.clip(im[y, x] * fac, 0, 1)
    if has_fac or (uv is None and vc is None):
        return np.tile(np.clip(fac, 0, 1) if has_fac else [0.5, 0.5, 0.5], (len(fi), 1))
    if vc is not None:
        return np.clip((bary[:, :, None] * vc[F[fi]]).sum(1), 0, 1)
    return np.full((len(fi), 3), 0.5)


def sample(sms, leaf_of, n, rng, J, BIN, cache):
    areas = np.concatenate([0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) for V, F, *_ in sms])
    mi = np.concatenate([np.full(len(sm[1]), k) for k, sm in enumerate(sms)])
    fi = np.concatenate([np.arange(len(sm[1])) for sm in sms])
    sel = rng.choice(len(areas), n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    xyz = np.zeros((n, 3))
    rgb = np.zeros((n, 3))
    lab = np.full(n, -1, dtype=np.int64)
    for k, sm in enumerate(sms):
        j = np.where(mi[sel] == k)[0]
        if not len(j):
            continue
        f = fi[sel][j]
        V, F = sm[0], sm[1]
        xyz[j] = (bary[j][:, :, None] * V[F[f]]).sum(1)
        rgb[j] = colours(sm, f, bary[j], J, BIN, cache)
        lab[j] = leaf_of[k]
    return xyz, rgb.astype(np.float32), lab, areas


def part_labels(cat, leaf, paths):
    rules = MAPPING["parts"].get(cat, {})
    cand = {nm: pid for pid, nm in taxonomy.candidates(MAPPING["categories"][cat])}
    lab = np.full(len(leaf), -1, dtype=np.int64)
    for li, path in enumerate(paths):
        nm = rules.get(path[-1]) if path else None
        if nm is not None:
            lab[leaf == li] = cand[nm]
    return lab


def one(source):
    cat, src, mid = source
    out = testsets.point_file("artiverse", source)
    if out.exists():
        return "cached", mid
    f = config.ARTIVERSE_ROOT / "data" / cat / src / mid / f"{mid}.segmented.glb"
    if not f.exists():
        return "missing", mid
    try:
        sms, labels, parent, J, BIN = submeshes(f)
        sms = [sm for sm in sms if len(sm[1])]
        if not sms:
            return "no triangles", mid
        pids = sorted({sm[4] for sm in sms if sm[4] is not None})
        li = {p: k for k, p in enumerate(pids)}
        leaf_of = [li[sm[4]] if sm[4] is not None else -1 for sm in sms]
        rng = np.random.default_rng([SAMPLE_SEED, zlib.crc32(f"{cat}__{src}__{mid}".encode())])
        xyz, rgb, leaf, areas = sample(sms, leaf_of, N_POINTS, rng, J, BIN, {})
        paths = [path_of(p, labels, parent) for p in pids]
    except Exception as e:
        return f"load error: {repr(e)[:60]}", mid
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, xyz=xyz.astype(np.float32), rgb=rgb, leaf=leaf, paths=json.dumps(paths), labels=part_labels(cat, leaf, paths))
    return "ok", mid


def check(limit):
    B = testsets.build("artiverse")
    frame = np.array(B["frame"])
    worst, n, bad = 0.0, 0, 0
    for s in B["shapes"][:limit or None]:
        z = np.load(testsets.point_file("artiverse", s["source"]))
        x = normalise_points64(z["xyz"].astype(np.float64) @ frame)
        lab = z["labels"]
        got = {}
        for pid in sorted(set(lab.tolist()) - {-1}):
            d = x[lab == pid].mean(0) - x.mean(0)
            if np.linalg.norm(d) >= TAU:
                got[pid] = d
        want = {p["pid"]: np.array(p["c"]) for p in s["parts"]}
        if set(got) != set(want):
            bad += 1
            continue
        worst = max([worst] + [float(np.abs(got[p] - want[p]).max()) for p in want])
        n += 1
    print(f"{n} shapes match the stored ground truth (max |diff| {worst:.2e}); {bad} differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.check:
        return check(a.limit)
    sources = [s["source"] for s in testsets.build("artiverse")["shapes"][:a.limit or None]]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, sources, chunksize=2))
    from collections import Counter
    print(dict(Counter(s for s, _ in res)))


if __name__ == "__main__":
    main()
