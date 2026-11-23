import numpy as np
import torch

from . import config, pointbert, taxonomy
from .common import lab_colour, load_state, seed_all
from .heads import GATE_ARCH, EvidenceHead, QueryGate, Refiner, SegHead, gated, membership, point_readout, spread
from .poses import pass_rotations, passes
from .solver import reorient

DIRECT_SALT = 109


def direct_seed(item, k):
    return int((item["row"] * 8 + (0 if item["pose"] == "canon" else 4) + k) * 7919 + DIRECT_SALT) % 2 ** 31


def back(v, R):
    Rt = torch.eye(3, device="cuda") if R is None else torch.tensor(R, device="cuda", dtype=torch.float32)
    return v @ Rt.T


def make_item(xyz, rgb, cat, pids, seed=0, fps_seed=None):
    xyz = np.asarray(xyz, np.float32)
    return {"oid": f"input:{seed}", "pose": "rot", "row": int(seed), "cat": cat, "cat_id": taxonomy.category_id(cat), "xyz": xyz,
            "rgb": np.asarray(rgb, np.float32), "pids": sorted(set(int(p) for p in pids)),
            "fps_seed": int(fps_seed if fps_seed is not None else (int(seed) * 7919 + 17) % 2 ** 31), "rotations": pass_rotations(seed, "rot")}


class GeoRead3D:
    def __init__(self, gate_seeds=config.GATE_SEEDS, direct_seeds=config.DIRECT_SEEDS):
        self.encoder = pointbert.build_encoder()
        self.segmenter = SegHead().cuda()
        self.segmenter.load_state_dict(load_state(config.ckpt("segmenter")))
        self.refiner = Refiner().cuda()
        self.refiner.load_state_dict(load_state(config.ckpt("plr")))
        self.gates = {}
        for s in gate_seeds:
            g = QueryGate(**GATE_ARCH).cuda()
            g.load_state_dict(load_state(config.ckpt(f"gate_{s}")))
            self.gates[s] = g
        self.direct = {}
        for s in direct_seeds:
            h = EvidenceHead().cuda()
            h.load_state_dict(load_state(config.ckpt(f"direct_{s}")))
            self.direct[s] = h
        for m in [self.segmenter, self.refiner, *self.gates.values(), *self.direct.values()]:
            m.eval()
        from .utonia_features import UtoniaFeatures
        self.features = UtoniaFeatures()
        self.pairs = list(zip(self.gates, self.direct))

    @torch.no_grad()
    def segmentation_route(self, item):
        pids = item["pids"]
        K = len(pids)
        pid = torch.tensor(pids, device="cuda")
        cat = torch.tensor([item["cat_id"]], device="cuda")
        per = {s: [] for s in self.gates}
        for k, xb, R in passes(item):
            seed_all(item["fps_seed"] + 1000 * k)
            tok, idx, cidx = pointbert.encode(self.encoder, xb[None], item["rgb"][None])
            X = torch.from_numpy(xb[None]).cuda().float()
            rg = torch.from_numpy(item["rgb"][None]).cuda()
            center, colg = pointbert.group_inputs(X, rg, idx, cidx)
            P = torch.softmax(self.segmenter(tok, cat), -1)
            colp = lab_colour(rg)
            Mn = membership(idx, X, cidx)
            e = lambda x: x[0][None].expand(K, *x.shape[1:])
            Pk, ik, Xk = e(P), e(idx), e(X)
            a = torch.softmax(self.refiner(e(tok)[:, 1:].float(), e(center), e(colg), Pk, pid), -1)
            sp = spread(a, ik, Xk.shape[1])
            for s, gate in self.gates.items():
                lg = gate(e(tok), pid, cat.expand(K), e(Mn), e(center), Xk, e(colp))
                per[s].append(back(point_readout(gated(sp, lg), Xk), R))
        return {s: torch.stack(v).mean(0) for s, v in per.items()}

    @torch.no_grad()
    def direct_route(self, chunk):
        clouds, rgbs, seeds, meta = [], [], [], []
        for i, item in enumerate(chunk):
            for k, xb, R in passes(item):
                clouds.append(xb)
                rgbs.append(item["rgb"])
                seeds.append(direct_seed(item, k))
                meta.append((i, R))
        feats = self.features(clouds, rgbs, seeds, seeds[0])
        per = [{s: [] for s in self.direct} for _ in chunk]
        for f, (i, R), xb in zip(feats, meta, clouds):
            item = chunk[i]
            K = len(item["pids"])
            pid = torch.tensor(item["pids"], device="cuda")
            cat = torch.tensor([item["cat_id"]], device="cuda")
            Xk = torch.from_numpy(xb).cuda()[None].float().expand(K, -1, -1)
            col = lab_colour(torch.from_numpy(item["rgb"]).cuda())[None].expand(K, -1, -1)
            for s, h in self.direct.items():
                w = torch.softmax(h(f[None].expand(K, -1, -1), pid, cat.expand(K), col), -1)
                per[i][s].append(back(point_readout(w, Xk), R))
        return [{s: torch.stack(v).mean(0) for s, v in d.items()} for d in per]

    def run(self, chunk):
        direct = self.direct_route(chunk)
        out = []
        for item, dv in zip(chunk, direct):
            sv = self.segmentation_route(item)
            for q, p in enumerate(item["pids"]):
                rec = {"oid": item["oid"], "pid": int(p), "pose": item["pose"]}
                for j, (gs, ds) in enumerate(self.pairs, 1):
                    a = [float(x) for x in sv[gs][q]]
                    b = [float(x) for x in dv[ds][q]]
                    rec[f"segmentation{j}"] = a
                    rec[f"direct{j}"] = b
                    rec[f"georead3d{j}"] = ((np.array(a) + np.array(b)) / 2).tolist()
                out.append(rec)
        return out

    def predict(self, xyz, rgb, cat, pids, seed=0, fps_seed=None):
        item = make_item(xyz, rgb, cat, pids, seed, fps_seed)
        recs = self.run([item])
        return {r["pid"]: np.array(r["georead3d1"]) for r in recs}

    @staticmethod
    def reorient(anchor_a, anchor_b, goal_a, goal_b):
        return reorient(anchor_a, anchor_b, goal_a, goal_b)
