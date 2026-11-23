import math

import torch


def unit(v, eps=1e-8):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def normalise(Xt):
    Xc = Xt - Xt.mean(1, keepdim=True)
    rms = Xc.pow(2).sum(-1).mean(1).sqrt()
    return Xc / rms[:, None, None], rms


def kernels(Xn, ells):
    D2 = torch.cdist(Xn, Xn).pow(2)
    return {l: torch.exp(-D2 / (2 * l * l)) for l in ells}


def slots(j):
    s = torch.zeros_like(j)
    for k in range(1, len(j)):
        s[k] = s[k - 1] + 1 if j[k] == j[k - 1] else 0
    return s


def kernel_quad(a, Kb, j, sl, kmax):
    B, N, _ = Kb.shape
    A = a.new_zeros(B, kmax, N)
    A[j, sl] = a
    KA = torch.bmm(A, Kb)
    return (A * KA).sum(-1)[j, sl]


def pair_index(j, t, lo=30.0, hi=150.0):
    out = []
    jj = j.tolist()
    tn = unit(t)
    for a in range(len(jj)):
        for b in range(a + 1, len(jj)):
            if jj[a] != jj[b]:
                continue
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(tn[a] @ tn[b])))))
            if lo <= ang <= hi:
                out.append((a, b))
    return torch.tensor(out, dtype=torch.long, device=t.device).reshape(-1, 2)


def direction_loss(r, t, beta_deg=30.0):
    return (unit(r) - unit(t)).pow(2).sum(-1) / (4 * math.sin(math.radians(beta_deg) / 2) ** 2)
