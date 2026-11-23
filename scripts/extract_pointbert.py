import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georead3d import config

PREFIX = "model.point_backbone."


def main():
    index = json.load(open(config.POINTLLM_CKPT / "pytorch_model.bin.index.json"))["weight_map"]
    shards = sorted({v for k, v in index.items() if k.startswith(PREFIX)})
    state = {}
    for shard in shards:
        full = torch.load(config.POINTLLM_CKPT / shard, map_location="cpu", weights_only=True)
        state.update({k[len(PREFIX):]: v.clone() for k, v in full.items() if k.startswith(PREFIX)})
        del full
    out = config.ckpt("pointbert_encoder")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, out)
    print(f"{len(state)} tensors -> {out}")


if __name__ == "__main__":
    main()
