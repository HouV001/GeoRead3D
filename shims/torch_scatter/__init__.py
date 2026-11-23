import torch

__version__ = "0.0"


def segment_csr(src, indptr, reduce="sum", out=None):
    counts = indptr[1:] - indptr[:-1]
    m = counts.numel()
    seg = torch.repeat_interleave(torch.arange(m, device=src.device), counts)
    idx = seg.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    red = {"sum": "sum", "add": "sum", "mean": "mean", "max": "amax", "min": "amin"}[reduce]
    res = torch.zeros((m,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    return res.scatter_reduce(0, idx, src, reduce=red, include_self=False)
