set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/prepare_partnext.py
$PY scripts/prepare_artiverse.py
for d in partnext artiverse; do
  $PY scripts/evaluate.py --dataset "$d"
  $PY scripts/score.py --dataset "$d" --out "${GEOREAD3D_OUT_DIR:-outputs}/${d}_scores.json"
done
