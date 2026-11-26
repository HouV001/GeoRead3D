# GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation

*Maintained by Hou; archived by Claude.*

> Official implementation of **GeoRead3D** for the paper *GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation* ([arXiv:2611.30519](https://arxiv.org/abs/2611.30519)).

Language-guided 3D part grounding for pointing to object parts and reorienting objects from part descriptions, built on two core designs:

1. **Geometric Readout Supervision (GRS).** Point evidence is a distribution over the observed points whose weighted centroid gives the part's location. GRS trains this evidence through the centroid's position relative to the object center and adds direction supervision on the parts used in reorientation commands, instead of supervising mask overlap alone.
2. **Dual-Route Part Evidence.** A refinement route corrects a part segmenter's evidence with a Part Localization Refiner (PLR) and a query gate on frozen Point-BERT features; a direct route predicts point evidence from frozen Utonia features. Evidence from both routes and four rotated copies of the input is averaged per point and read out once.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 and PyTorch with CUDA are required. Two frozen encoders come from their own repositories:

- [PointLLM](https://github.com/OpenRobotLab/PointLLM) and its PointLLM-7B v1.2 checkpoint, from which `scripts/extract_pointbert.py` extracts the Point-BERT encoder.
- [Utonia](https://github.com/Pointcept/Utonia); its weights are loaded from the Hugging Face Hub (`Pointcept/Utonia`) unless `GEOREAD3D_UTONIA_CKPT` points to a local checkpoint.

Evaluation with four pose passes peaks at about 75 GB of GPU memory.

## Datasets

- **3DCoMPaT** — training data: coarse taxonomy, `hdf5/train_coarse_2048_10.hdf5` (14,350 training shapes).
- **PartNeXt** — the test split; `scripts/prepare_partnext.py --download` fetches the required shapes.
- **Artiverse** — the test sample; a gated dataset that must be obtained separately.

`data/` holds the test-set definitions for the PartNeXt test split and the Artiverse sample, the taxonomy mappings, near-duplicate exclusions, part descriptions and the stored mapper answers. `scripts/prepare_partnext.py` and `scripts/prepare_artiverse.py` convert the listed shapes into 2,048 colored points with part labels (`--check` compares the prepared ground truth with the stored test-set definitions).

Hard paths are not baked in. All locations are set with environment variables; defaults are relative to the repository root:

| Variable | Used by | Purpose |
|---|---|---|
| `GEOREAD3D_COMPAT_ROOT` | training | 3DCoMPaT root (default `datasets/3DCoMPaT200`) |
| `GEOREAD3D_PARTNEXT_ROOT` | `prepare_partnext.py` | PartNeXt root (default `datasets/PartNeXt`) |
| `GEOREAD3D_ARTIVERSE_ROOT` | `prepare_artiverse.py` | Artiverse root (default `datasets/Artiverse`) |
| `GEOREAD3D_PARTNEXT_POINTS` | data preparation, evaluation | prepared PartNeXt points (default `prepared/partnext`) |
| `GEOREAD3D_ARTIVERSE_POINTS` | data preparation, evaluation | prepared Artiverse points (default `prepared/artiverse`) |
| `GEOREAD3D_POINTLLM_REPO` | Point-BERT encoder | PointLLM repository (default `third_party/PointLLM`) |
| `GEOREAD3D_POINTLLM_CKPT` | `extract_pointbert.py` | PointLLM-7B v1.2 checkpoint (default `third_party/PointLLM_7B_v1.2`) |
| `GEOREAD3D_UTONIA_REPO` | Utonia features | Utonia repository (default `third_party/Utonia`) |
| `GEOREAD3D_UTONIA_CKPT` | Utonia features | optional local Utonia checkpoint (default: Hugging Face Hub) |
| `GEOREAD3D_CKPT_DIR` | training, evaluation | checkpoints (default `checkpoints`) |
| `GEOREAD3D_CACHE_DIR` | `cache_utonia.py`, `train_direct_head.py` | cached Utonia features (default `cache`) |
| `GEOREAD3D_OUT_DIR` | `evaluate.py`, `score.py` | predictions and scores (default `outputs`) |

## Train

```bash
python scripts/extract_pointbert.py
python scripts/train_segmenter.py --which full
python scripts/train_segmenter.py --which fold0
python scripts/train_segmenter.py --which fold1
python scripts/train_plr.py
for s in 20261301 20261302 20261303; do python scripts/train_query_gate.py --seed $s; done
python scripts/cache_utonia.py
python scripts/train_direct_head.py
```

or all steps at once with `bash scripts/train_all.sh`. The steps write `pointbert_encoder.pt`; the part segmenter `segmenter.pt` and its two cross-fitting fold heads `segmenter_fold0.pt`, `segmenter_fold1.pt`; the Part Localization Refiner `plr.pt`, trained on out-of-fold probabilities; the query gates `gate_<seed>.pt`; the cached Utonia features of the training shapes; and the direct-route evidence heads trained with GRS, `direct_<seed>.pt`. Checkpoints are written to `GEOREAD3D_CKPT_DIR`; every training script accepts `--max-steps` for a short run.

## Pretrained weights

A single archive on Google Drive bundles all released checkpoints:

[Download from Google Drive](https://drive.google.com/file/d/1qa1GuWiE9kePJItdY_3BSYPTHgzv9zw0/view?usp=drive_link)

```bash
pip install gdown
gdown 1qa1GuWiE9kePJItdY_3BSYPTHgzv9zw0
# unrar x / 7z x the downloaded archive — the checkpoints end up in GeoRead3D_checkpoints/
mv GeoRead3D_checkpoints checkpoints     # or: export GEOREAD3D_CKPT_DIR=$PWD/GeoRead3D_checkpoints
```

| Checkpoint | Module | Trained on | Used for |
|---|---|---|---|
| `pointbert_encoder.pt` | frozen Point-BERT encoder | PointLLM-7B v1.2 | training, inference |
| `segmenter.pt` | part segmenter | 3DCoMPaT train | inference |
| `plr.pt` | Part Localization Refiner | 3DCoMPaT train | inference |
| `gate_20261301.pt`, `gate_20261302.pt`, `gate_20261303.pt` | query gate, three seeds | 3DCoMPaT train | inference |
| `direct_20261901.pt`, `direct_20261911.pt`, `direct_20261921.pt` | direct-route evidence heads trained with GRS, three seeds | 3DCoMPaT train | inference |
| `segmenter_fold0.pt`, `segmenter_fold1.pt` | cross-fitting fold heads of the segmenter | 3DCoMPaT train | training PLR |

Seed k of the model pairs `gate_2026130k.pt` with the k-th direct head; `evaluate.py` runs all three seeds and `score.py` reports the mean and standard deviation over them.

## Evaluate

```bash
python scripts/prepare_partnext.py --download
python scripts/evaluate.py --dataset partnext
python scripts/score.py --dataset partnext --out outputs/partnext_scores.json
```

For the Artiverse sample:

```bash
python scripts/prepare_artiverse.py
python scripts/evaluate.py --dataset artiverse
python scripts/score.py --dataset artiverse --out outputs/artiverse_scores.json
```

or both datasets with `bash scripts/evaluate_all.sh`. `evaluate.py` runs both routes on four pose passes for the three seeds, and `score.py` reports pointing, reorientation, whole-object rotation and severe error rates. Part descriptions are mapped to part names with the stored mapper answers in `data/expressions.json`.

## Results

**Table 1. Comparison with prior methods on the PartNeXt test split and Artiverse.** Success and severe error rates in percent on rotated input with the shared mapper and four poses, or one pose for COPS. Point.: pointing within 22.5°; Reorient.: reorientation within 30°; Whole: whole-object rotation within 30°; Severe: pointing errors above 90°. Trained rows give the mean ± standard deviation over three seeds, and bold marks the best mean. Params counts the networks run at inference, excluding cached text embeddings and the shared mapper. Time is the median latency per object on one GPU.

| Method | Variant | PartNeXt Point. | PartNeXt Reorient. | PartNeXt Whole | PartNeXt Severe | Artiverse Point. | Artiverse Reorient. | Artiverse Whole | Artiverse Severe | Params | Time (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COPS | Released | 18.5 | 6.4 | 5.7 | 50.8 | 14.8 | 3.7 | 3.2 | 54.7 | 236.2M | 712 |
| Find3D | Released | 38.4 | 26.0 | 23.0 | 23.0 | 26.5 | 13.3 | 11.1 | 33.7 | 46.2M | 282 |
| CoSMo3D | Released | 41.8 | 27.7 | 23.6 | 19.6 | 37.2 | 16.6 | 14.1 | 26.3 | 57.5M | 251 |
| PatchAlign3D | Released | 45.5 | 33.4 | 29.7 | 19.3 | 29.7 | 18.3 | 16.4 | 37.6 | 22.4M | 57 |
| Point-BERT | Frozen + mask head | 64.7 ± 1.1 | 53.8 ± 1.8 | 48.2 ± 1.8 | 5.8 ± 0.5 | 52.5 ± 0.9 | 27.1 ± 0.6 | 24.9 ± 0.4 | 16.5 ± 0.1 | 22.1M | 140 |
| Find3D | Frozen + mask head | 39.2 ± 0.8 | 26.8 ± 0.8 | 23.8 ± 0.7 | 23.5 ± 1.0 | 33.2 ± 0.3 | 7.6 ± 1.0 | 6.5 ± 1.3 | 30.9 ± 0.4 | 46.7M | 252 |
| PatchAlign3D | Frozen + mask head | 51.5 ± 0.9 | 37.1 ± 1.4 | 31.7 ± 1.8 | 9.6 ± 0.3 | 37.9 ± 0.4 | 17.8 ± 1.7 | 16.6 ± 1.4 | 30.9 ± 0.3 | 23.1M | 46 |
| Utonia | Frozen + mask head | 68.4 ± 1.0 | 60.8 ± 1.6 | 54.0 ± 1.3 | 4.2 ± 0.2 | 51.0 ± 0.8 | 25.0 ± 1.4 | 23.2 ± 1.1 | 18.4 ± 0.6 | 138.0M | 147 |
| Point-BERT | Fine-tuned | 67.8 ± 0.3 | 59.5 ± 1.0 | 54.2 ± 0.9 | 6.7 ± 0.2 | 52.7 ± 0.4 | 24.4 ± 0.8 | 22.8 ± 0.5 | 17.9 ± 0.7 | 22.1M | 41 |
| PointNeXt-S | From scratch | 43.3 ± 1.0 | 30.8 ± 0.8 | 27.4 ± 0.9 | 16.8 ± 0.6 | 37.1 ± 1.0 | 16.3 ± 0.7 | 15.3 ± 1.2 | 30.7 ± 0.9 | 22.6M | 52 |
| PointLLM-7B | Fine-tuned | 53.6 ± 1.2 | 40.8 ± 1.5 | 36.6 ± 1.5 | 10.2 ± 0.8 | 49.1 ± 1.0 | 20.7 ± 1.4 | 18.1 ± 0.7 | 18.0 ± 1.3 | 6.78B | 483 |
| Utonia | Fine-tuned | 63.6 ± 1.7 | 55.8 ± 2.6 | 50.6 ± 2.4 | 7.6 ± 0.6 | 44.1 ± 1.4 | 21.3 ± 1.7 | 19.8 ± 1.8 | 23.8 ± 0.8 | 137.6M | 159 |
| **GeoRead3D** (ours) | — | **74.1 ± 0.4** | **71.2 ± 0.3** | **65.3 ± 0.4** | **3.0 ± 0.1** | **60.0 ± 0.8** | **34.9 ± 2.7** | **32.2 ± 2.3** | **12.7 ± 0.8** | 160.6M | 307 |

## Repository layout

- `georead3d/` — model (`model.py`), modules and readout (`heads.py`), losses of the direct route (`losses.py`), frozen encoders (`pointbert.py`, `utonia_features.py`), pose passes (`poses.py`), two-anchor solver (`solver.py`), mapper (`mapper.py`), taxonomy and data loading (`taxonomy.py`, `testsets.py`, `corpus.py`), configuration (`config.py`)
- `scripts/` — CLI scripts for training, data preparation, evaluation and scoring
- `data/` — test-set definitions, taxonomy mappings, near-duplicate exclusions, part descriptions and stored mapper answers
- `shims/` — a PyTorch implementation of `torch_scatter.segment_csr`, used by Utonia

## Citation

```bibtex
@article{hou2026georead3d,
  title   = {GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation},
  author  = {Hou, Zhangcheng and Tonkich, Kevin and Ohtsuki, Tomoaki},
  journal = {arXiv preprint arXiv:2611.30519},
  year    = {2026}
}
```
