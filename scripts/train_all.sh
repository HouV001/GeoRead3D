set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/extract_pointbert.py
$PY scripts/train_segmenter.py --which full
$PY scripts/train_segmenter.py --which fold0
$PY scripts/train_segmenter.py --which fold1
$PY scripts/train_plr.py
for s in 20261301 20261302 20261303; do
  $PY scripts/train_query_gate.py --seed "$s"
done
$PY scripts/cache_utonia.py
$PY scripts/train_direct_head.py
