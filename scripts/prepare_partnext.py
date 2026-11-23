import argparse
import glob
import hashlib
import json
import string
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config, taxonomy, testsets
from georead3d.common import normalise_points64
from georead3d.corpus import TAU

SAMPLE_SEED = 20262602
N_POINTS = 2048
MAPPING = json.load(open(config.DATA / "partnext/mapping.json"))


def id_seed(mid, k):
    h = mid[:k] if all(ch in string.hexdigits for ch in mid[:k]) else hashlib.sha256(mid.encode()).hexdigest()[:k]
    return int(h, 16)


def scene_meshes(scene):
    import trimesh
    gn = scene.graph.geometry_nodes
    out = []
    for name, g in scene.geometry.items():
        if not isinstance(g, trimesh.Trimesh):
            raise ValueError("non-Trimesh geometry")
        node = gn[name]
        if len(node) != 1:
            raise ValueError("geometry with several nodes")
        T, _ = scene.graph[node[0]]
        g = g.copy()
        g.apply_transform(T)
        out.append(g)
    return out


def colours(m, fi, bary):
    import trimesh
    vis = m.visual
    F = m.faces[fi]
    try:
        if isinstance(vis, trimesh.visual.texture.TextureVisuals):
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            fac = np.asarray(getattr(mat, "baseColorFactor", None) if getattr(mat, "baseColorFactor", None) is not None else [255, 255, 255, 255], dtype=np.float64)[:3] / 255.
            if img is not None and vis.uv is not None and len(vis.uv) == len(m.vertices):
                uv = (bary[:, :, None] * vis.uv[F]).sum(1)
                c = trimesh.visual.color.uv_to_color(uv, img)[:, :3].astype(np.float64) / 255.
                return np.clip(c * fac, 0, 1)
            if getattr(mat, "baseColorFactor", None) is not None:
                return np.tile(fac, (len(fi), 1))
            if getattr(mat, "diffuse", None) is not None:
                return np.tile(np.asarray(mat.diffuse[:3], dtype=np.float64) / 255., (len(fi), 1))
        elif isinstance(vis, trimesh.visual.color.ColorVisuals) and vis.kind == "vertex":
            vc = vis.vertex_colors[:, :3].astype(np.float64) / 255.
            return (bary[:, :, None] * vc[F]).sum(1)
        elif isinstance(vis, trimesh.visual.color.ColorVisuals) and vis.kind == "face":
            return vis.face_colors[fi, :3].astype(np.float64) / 255.
    except Exception:
        pass
    return np.full((len(fi), 3), 0.5)


def sample(meshes, leaf_of, n, rng):
    areas = np.concatenate([m.area_faces for m in meshes])
    mi = np.concatenate([np.full(len(m.faces), k) for k, m in enumerate(meshes)])
    fi = np.concatenate([np.arange(len(m.faces)) for m in meshes])
    sel = rng.choice(len(areas), n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    xyz = np.zeros((n, 3))
    rgb = np.zeros((n, 3))
    lab = np.full(n, -1, dtype=np.int64)
    for k, m in enumerate(meshes):
        j = np.where(mi[sel] == k)[0]
        if not len(j):
            continue
        f = fi[sel][j]
        tri = m.vertices[m.faces[f]]
        xyz[j] = (bary[j][:, :, None] * tri).sum(1)
        rgb[j] = colours(m, f, bary[j])
        lab[j] = leaf_of[k][f]
    return xyz, rgb.astype(np.float32), lab


def leaf_names(cat, hierarchy):
    rules = {(k.split("@")[0], int(k.split("@")[1])): v for k, v in MAPPING["parts"][cat].items()}
    out = {}

    def walk(n, d, cur):
        hit = rules.get((n["name"], d))
        cur = hit if hit is not None else cur
        if "maskId" in n and n["maskId"] is not None:
            out[int(n["maskId"])] = cur
        for c in n.get("children", []) or []:
            walk(c, d + 1, cur)

    walk(json.loads(hierarchy)[0], 0, None)
    return out


def part_labels(cat, hierarchy, leaf):
    cand = {nm: pid for pid, nm in taxonomy.candidates(MAPPING["categories"][cat])}
    lab = np.full(len(leaf), -1, dtype=np.int64)
    for l, nm in leaf_names(cat, hierarchy).items():
        if nm is not None:
            lab[leaf == l] = cand[nm]
    return lab


def one(args):
    import trimesh
    cat, r, out = args
    if out.exists():
        return "cached", r["model_id"]
    p = config.PARTNEXT_ROOT / "mesh/glbs" / r["type_id"] / f"{r['model_id']}.glb"
    if not p.exists():
        return "missing", r["model_id"]
    try:
        ms = scene_meshes(trimesh.load(str(p), force="scene"))
    except Exception as e:
        return f"load error: {repr(e)[:60]}", r["model_id"]
    fn = json.loads(r["mesh_face_num"])
    if [len(m.faces) for m in ms] != [fn[str(i)] for i in range(len(fn))]:
        return "face-count mismatch", r["model_id"]
    leaf_of = [np.full(len(m.faces), -1, dtype=np.int64) for m in ms]
    for mid, d in json.loads(r["masks"]).items():
        for k, faces in d.items():
            leaf_of[int(k)][np.asarray(faces, dtype=np.int64)] = int(mid)
    rng = np.random.default_rng([SAMPLE_SEED, id_seed(r["model_id"], 12)])
    xyz, rgb, leaf = sample(ms, leaf_of, N_POINTS, rng)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, xyz=xyz, rgb=rgb, leaf=leaf, labels=part_labels(cat, r["hierarchyList"], leaf))
    return "ok", r["model_id"]


def annotations(want):
    import pyarrow as pa
    rows = {}
    for f in sorted(glob.glob(str(config.PARTNEXT_ROOT / "anno/data-*.arrow"))):
        with pa.memory_map(f) as src:
            t = pa.ipc.open_stream(src).read_all()
        mids = t.column("model_id").to_pylist()
        idx = [i for i, m in enumerate(mids) if m in want]
        if idx:
            for r in t.take(pa.array(idx)).to_pylist():
                rows[r["model_id"]] = r
    return rows


def download(rows):
    from huggingface_hub import hf_hub_download
    for r in rows.values():
        if not (config.PARTNEXT_ROOT / "mesh/glbs" / r["type_id"] / f"{r['model_id']}.glb").exists():
            hf_hub_download("AuWang/PartNeXt_mesh", f"glbs/{r['type_id']}/{r['model_id']}.glb", repo_type="dataset", local_dir=str(config.PARTNEXT_ROOT / "mesh"))


def check(limit):
    B = testsets.build("partnext")
    frame = np.array(B["frame"])
    worst, n, bad = 0.0, 0, 0
    for s in B["shapes"][:limit or None]:
        z = np.load(testsets.point_file("partnext", s["source"]))
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
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.check:
        return check(a.limit)
    shapes = testsets.build("partnext")["shapes"][:a.limit or None]
    want = {s["source"][1]: s["source"][0] for s in shapes}
    rows = annotations(want)
    if a.download:
        download(rows)
    jobs = [(want[m], rows[m], testsets.point_file("partnext", [want[m], m])) for m in want if m in rows]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=2))
    from collections import Counter
    print(dict(Counter(s for s, _ in res)), f"missing annotations: {len(want) - len(jobs)}")


if __name__ == "__main__":
    main()
