import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .taxonomy import NCAT, NPART

EPS = 1e-12
GATE_ARCH = {"dtok": 384, "dpart": 32, "dcat": 16, "hidden": 256, "dz": 128, "dpt": 64}


class SegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = nn.Embedding(NCAT, 32)
        self.mlp = nn.Sequential(nn.Linear(384 + 384 + 32, 256), nn.GELU(), nn.Linear(256, NPART))

    def forward(self, tok, cat):
        g = tok[:, 1:].float()
        c = tok[:, :1].float().expand_as(g)
        e = self.cat(cat)[:, None].expand(-1, g.shape[1], -1)
        return self.mlp(torch.cat([g, c, e], -1))


def part_logp(P, pid):
    return P.gather(2, pid[:, None, None].expand(-1, P.shape[1], 1)).squeeze(-1).clamp_min(EPS).log()


def spread(a, knn, n):
    b, g, s = knn.shape
    return torch.zeros(b, n, device=a.device, dtype=a.dtype).scatter_add(1, knn.reshape(b, -1), (a / s).repeat_interleave(s, 1))


def group_readout(a, knn, X):
    w = spread(a, knn, X.shape[1])
    return (w[..., None] * X).sum(1) - X.mean(1)


def soft_weights(P, pid):
    return torch.softmax(part_logp(P, pid), -1)


def rel_sq_error(d, t):
    return ((d - t) ** 2).sum(-1) / (t ** 2).sum(-1).clamp_min(1e-12)


class Refiner(nn.Module):
    def __init__(self, dtok=384, dproj=64, dpart=16, hidden=64):
        super().__init__()
        self.proj = nn.Linear(dtok, dproj)
        self.emb = nn.Embedding(NPART, dpart)
        self.mlp = nn.Sequential(nn.Linear(dproj + 3 + 3 + 4 + dpart, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tok, center, colour, P, pid):
        b, g, _ = tok.shape
        lp = part_logp(P, pid)
        p = lp.exp()
        other = P.scatter(2, pid[:, None, None].expand(-1, g, 1), 0.0).max(-1).values
        ent = -(P.clamp_min(EPS) * P.clamp_min(EPS).log()).sum(-1)
        c0 = (torch.softmax(lp, -1)[..., None] * center).sum(1, keepdim=True)
        f = torch.cat([self.proj(tok), center - c0, colour, p[..., None], lp[..., None], other[..., None], ent[..., None],
                       self.emb(pid)[:, None].expand(-1, g, -1)], -1)
        return lp + self.mlp(f).squeeze(-1)


def centres(X, cidx):
    return X.gather(1, cidx[..., None].expand(-1, -1, 3))


def membership(knn, X, cidx):
    b, g, s = knn.shape
    n = X.shape[1]
    flat = (knn * g + torch.arange(g, device=knn.device)[None, :, None]).reshape(b, -1)
    M = torch.zeros(b, n * g, device=X.device, dtype=X.dtype).scatter_add_(1, flat, torch.ones(flat.shape, device=X.device, dtype=X.dtype)).view(b, n, g)
    miss = M.sum(2) == 0
    if miss.any():
        near = torch.cdist(X, centres(X, cidx)).argmin(2)
        M = torch.where(miss[..., None], Fn.one_hot(near, g).to(X.dtype), M)
    return M / M.sum(2, keepdim=True)


def point_readout(w, X):
    return (w[..., None] * X).sum(-2) - X.mean(-2)


def gated(w, logit):
    v = w * torch.sigmoid(logit)
    return v / v.sum(-1, keepdim=True).clamp_min(EPS)


def balanced_bce(logit, y):
    l = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return 0.5 * ((l * y).sum(-1) / y.sum(-1).clamp_min(1) + (l * (1 - y)).sum(-1) / (1 - y).sum(-1).clamp_min(1))


def split_bce(l, y):
    neg = 1 - y
    pos_m = (l * y).sum(-1) / y.sum(-1).clamp_min(1)
    return 0.5 * pos_m + 0.5 * (l * neg).sum(-1) / neg.sum(-1).clamp_min(1)


class GroupAttention(nn.Module):
    def __init__(self, d=128, heads=4, ffn=2):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, heads))
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Linear(ffn * d, d))
        for m in (self.o, self.ff[2]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        k, g, d = z.shape
        q, kk, v = self.qkv(self.ln1(z)).view(k, g, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        lg = q @ kk.transpose(-1, -2) / math.sqrt(self.dh)
        a = torch.softmax(lg, -1) @ v
        z = z + self.o(a.transpose(1, 2).reshape(k, g, d))
        return z + self.ff(self.ln2(z))


class QueryGate(nn.Module):
    def __init__(self, dtok=384, dpart=32, dcat=16, hidden=256, dz=128, dpt=64):
        super().__init__()
        self.part = nn.Embedding(NPART, dpart)
        self.cat = nn.Embedding(NCAT, dcat)
        self.inp = nn.Linear(2 * dtok, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.film = nn.Linear(dpart + dcat, 2 * hidden)
        self.mid = nn.Linear(hidden, dz)
        self.pt = nn.Sequential(nn.Linear(dz + 6, dpt), nn.GELU(), nn.Linear(dpt, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.rel = GroupAttention(dz)

    def groups(self, tok, pid, cat):
        g = tok[:, 1:].float()
        c = tok[:, :1].float().expand_as(g)
        h = self.ln(self.inp(torch.cat([g, c], -1)))
        gam, bet = self.film(torch.cat([self.part(pid), self.cat(cat)], -1)).chunk(2, -1)
        return Fn.gelu(self.mid(Fn.gelu(h * (1 + gam[:, None]) + bet[:, None])))

    def forward(self, tok, pid, cat, Mn, center, X, col):
        z = self.rel(self.groups(tok, pid, cat))
        u = torch.matmul(Mn, z)
        r = X - torch.matmul(Mn, center)
        return self.pt(torch.cat([u, 10 * r, col], -1)).squeeze(-1)


class EvidenceHead(nn.Module):
    def __init__(self, din=1386, dpart=32, dcat=16, hidden=256, dz=128, dpt=64):
        super().__init__()
        self.part = nn.Embedding(NPART, dpart)
        self.cat = nn.Embedding(NCAT, dcat)
        self.inp = nn.Linear(2 * din, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.film = nn.Linear(dpart + dcat, 2 * hidden)
        self.mid = nn.Linear(hidden, dz)
        self.pt = nn.Sequential(nn.Linear(dz + 3, dpt), nn.GELU(), nn.Linear(dpt, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, f, pid, cat, col):
        h = self.ln(self.inp(torch.cat([f, f.mean(1, keepdim=True).expand_as(f)], -1)))
        gam, bet = self.film(torch.cat([self.part(pid), self.cat(cat)], -1)).chunk(2, -1)
        z = Fn.gelu(self.mid(Fn.gelu(h * (1 + gam[:, None]) + bet[:, None])))
        return self.pt(torch.cat([z, col], -1)).squeeze(-1)
