import sys

import numpy as np
import torch

from . import config
from .common import estimate_normals

for p in (str(config.SHIMS), str(config.UTONIA_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

FEATURE_DIM = 1386
SETTINGS = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)


class UtoniaFeatures:
    def __init__(self):
        import utonia
        self.utonia = utonia
        if config.UTONIA_CKPT:
            self.model = utonia.load(config.UTONIA_CKPT, custom_config=SETTINGS)
        else:
            self.model = utonia.load("utonia", repo_id="Pointcept/Utonia", custom_config=SETTINGS)
        self.model = self.model.cuda().eval()
        self.transform = utonia.transform.default(normalize_coord=True)

    @torch.no_grad()
    def __call__(self, clouds, rgbs, cloud_seeds, batch_seed):
        pts = []
        for xyz, rgb, sd in zip(clouds, rgbs, cloud_seeds):
            np.random.seed(sd)
            pts.append(self.transform({"coord": xyz.copy(), "color": rgb.astype(np.float32) * 255.0, "normal": estimate_normals(xyz)}))
        b = self.utonia.data.collate_fn(pts)
        for k in b:
            if isinstance(b[k], torch.Tensor):
                b[k] = b[k].cuda()
        torch.manual_seed(batch_seed)
        with torch.inference_mode():
            p = self.model(b)
            for _ in range(4):
                parent = p.pop("pooling_parent")
                inv = p.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, p.feat[inv]], dim=-1)
                p = parent
            off = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), b["offset"].long()])
            n = clouds[0].shape[0]
            return [p.feat[off[j] + b["inverse"][j * n:(j + 1) * n].long()].half().float().clone() for j in range(len(clouds))]
