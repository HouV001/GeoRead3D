import json
from functools import lru_cache

from . import config

NPART = 118
NCAT = 200


@lru_cache(maxsize=1)
def _table():
    table = json.load(open(config.DATA / "taxonomy.json"))["categories"]
    meta = config.COMPAT_ROOT / "metadata"
    if (meta / "classes.json").exists():
        parts = [p.replace("_", " ") for p in json.load(open(meta / "parts_coarse.json"))]
        classes = json.load(open(meta / "classes.json"))
        seg = json.load(open(meta / "coarse_seg_classes.json"))["seg_classes"]
        for k, c in enumerate(classes):
            table.setdefault(c, {"id": k, "parts": [[int(i), parts[int(i)]] for i in seg[str(k)]]})
    return table


def category_id(cat):
    return int(_table()[cat]["id"])


def candidates(cat):
    return [(int(p), n) for p, n in _table()[cat]["parts"]]


def display(cat):
    return cat.replace("_", " ")


def with_article(cat):
    d = display(cat)
    return ("an " if d[0] in "aeiou" else "a ") + d
