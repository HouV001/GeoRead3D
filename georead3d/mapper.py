import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache

from . import config, taxonomy

MODEL = os.environ.get("GEOREAD3D_MAPPER_MODEL", "claude-haiku-4-5")
SYSTEM = ("You are a careful annotator. Reply with the requested output format and nothing else: no preamble, "
          "no restatement, no markdown or code fences.")


def prompt(cat, expression):
    listed = ", ".join(n for _, n in taxonomy.candidates(cat))
    return (f"{taxonomy.with_article(cat).capitalize()} has these parts: {listed}.\n"
            f"Which ONE of them does the phrase \"{expression}\" refer to?\n"
            f"Answer with the part name exactly as listed.")


def parse_choice(text, cands):
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    hit = set()
    for pid, name in sorted(cands, key=lambda c: -len(c[1])):
        pat = " " + name.lower() + " "
        if pat in t:
            hit.add(pid)
            t = t.replace(pat, " " * len(pat))
    return hit.pop() if len(hit) == 1 else None


def ask(cat, expression, timeout=300):
    cmd = ["claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--safe-mode", "--tools", "",
           "--no-session-persistence", "--effort", "low", "--output-format", "json"]
    with tempfile.TemporaryDirectory() as cwd:
        p = subprocess.run(cmd, input=prompt(cat, expression), capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        return None, None
    pay = json.loads(p.stdout)
    if pay.get("is_error") or pay.get("subtype") != "success":
        return None, None
    raw = pay.get("result", "")
    return parse_choice(raw, taxonomy.candidates(cat)), raw


@lru_cache(maxsize=1)
def stored():
    out = {}
    for e in json.load(open(config.DATA / "expressions.json")):
        out.setdefault((e["cat"], e["expr"]), e)
    return out


def resolve(cat, expression, allow_call=False):
    e = stored().get((cat, expression))
    if e is not None:
        return e["choice"]
    if allow_call:
        return ask(cat, expression)[0]
    return None


def part_id(cat, name):
    return dict((n, p) for p, n in taxonomy.candidates(cat)).get(name)
