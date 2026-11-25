# GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation

Zhangcheng Hou, Kevin Tonkich, Tomoaki Ohtsuki

[[arXiv:2611.30519]](https://arxiv.org/abs/2611.30519)

Language often refers to an object by its parts: point to the seat of a chair, or turn the chair so that its seat and base lie in given directions. Both tasks need the direction from the object's center to the part's center. GeoRead3D learns point evidence whose weighted centroid gives the part's location, and trains it with **Geometric Readout Supervision (GRS)**, which supervises the centroid's position and adds direction supervision on the parts used in reorientation commands. **Dual-Route Part Evidence** combines refined segmentation evidence with directly predicted point evidence, averaged over four rotated copies of the input.

This repository contains the final GeoRead3D model, the scripts that train each of its modules, and the evaluation on the PartNeXt test split and the Artiverse sample.

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

## Repository structure

```
georead3d/   model, modules, losses, encoders, pose passes, two-anchor solver, mapper, data loading, configuration
scripts/     training, data preparation, evaluation and scoring
data/        test-set definitions for the PartNeXt test split and the Artiverse sample, taxonomy mappings,
             near-duplicate exclusions, part descriptions and the stored mapper answers used in the paper
shims/       a PyTorch implementation of torch_scatter.segment_csr, used by Utonia
```

## Requirements

- Python 3.10 and PyTorch with CUDA; the remaining packages are listed in `requirements.txt`.
- [PointLLM](https://github.com/OpenRobotLab/PointLLM) (repository) and its PointLLM-7B v1.2 checkpoint, from which `scripts/extract_pointbert.py` extracts the frozen Point-BERT encoder.
- [Utonia](https://github.com/Pointcept/Utonia) (repository). Its weights are loaded from the Hugging Face Hub (`Pointcept/Utonia`) unless `GEOREAD3D_UTONIA_CKPT` points to a local checkpoint.
- A GPU with large memory: evaluation with four pose passes peaks at about 75 GB.

```bash
pip install -r requirements.txt
```

## Data

- **Training:** 3DCoMPaT (coarse taxonomy, `hdf5/train_coarse_2048_10.hdf5`, 14,350 training shapes).
- **Evaluation:** PartNeXt and Artiverse. Artiverse is a gated dataset and must be obtained separately.

`scripts/prepare_partnext.py` and `scripts/prepare_artiverse.py` convert the test shapes listed in `data/` into 2,048 colored points with part labels (`--download` fetches the required PartNeXt shapes; `--check` compares the prepared ground truth with the stored test-set definitions).

All locations are set with environment variables; the defaults are relative to the repository root.

| Variable | Default | Content |
|---|---|---|
| `GEOREAD3D_COMPAT_ROOT` | `datasets/3DCoMPaT200` | 3DCoMPaT |
| `GEOREAD3D_PARTNEXT_ROOT` | `datasets/PartNeXt` | PartNeXt |
| `GEOREAD3D_ARTIVERSE_ROOT` | `datasets/Artiverse` | Artiverse |
| `GEOREAD3D_PARTNEXT_POINTS` | `prepared/partnext` | prepared PartNeXt points |
| `GEOREAD3D_ARTIVERSE_POINTS` | `prepared/artiverse` | prepared Artiverse points |
| `GEOREAD3D_POINTLLM_REPO` | `third_party/PointLLM` | PointLLM repository |
| `GEOREAD3D_POINTLLM_CKPT` | `third_party/PointLLM_7B_v1.2` | PointLLM-7B v1.2 checkpoint |
| `GEOREAD3D_UTONIA_REPO` | `third_party/Utonia` | Utonia repository |
| `GEOREAD3D_UTONIA_CKPT` | (Hugging Face Hub) | optional local Utonia checkpoint |
| `GEOREAD3D_CKPT_DIR` | `checkpoints` | trained modules |
| `GEOREAD3D_CACHE_DIR` | `cache` | cached Utonia features |
| `GEOREAD3D_OUT_DIR` | `outputs` | predictions and scores |

## Training

```bash
bash scripts/train_all.sh
```

This runs, in order:

1. `extract_pointbert.py`: Point-BERT encoder → `pointbert_encoder.pt`
2. `train_segmenter.py --which full|fold0|fold1`: part segmenter and its two cross-fitting fold heads → `segmenter.pt`, `segmenter_fold0.pt`, `segmenter_fold1.pt`
3. `train_plr.py`: Part Localization Refiner on out-of-fold probabilities → `plr.pt`
4. `train_query_gate.py --seed S` for the three seeds → `gate_<seed>.pt`
5. `cache_utonia.py`: frozen Utonia features of the training shapes
6. `train_direct_head.py`: evidence heads of the direct route, trained with GRS, for the three seeds → `direct_<seed>.pt`

Checkpoints are written to `GEOREAD3D_CKPT_DIR`. Every training script accepts `--max-steps` for a short run.

## Pretrained weights

The checkpoints of the final model are available as a single archive on [Google Drive](https://drive.google.com/file/d/1qa1GuWiE9kePJItdY_3BSYPTHgzv9zw0/view?usp=drive_link) (93 MB):

```bash
pip install gdown
gdown 1qa1GuWiE9kePJItdY_3BSYPTHgzv9zw0
unrar x GeoRead3D_checkpoints.rar        # or: 7z x GeoRead3D_checkpoints.rar
mv GeoRead3D_checkpoints checkpoints     # or: export GEOREAD3D_CKPT_DIR=$PWD/GeoRead3D_checkpoints
```

| Checkpoint | Module | Used for |
|---|---|---|
| `pointbert_encoder.pt` | frozen Point-BERT encoder (from PointLLM-7B v1.2) | inference, training |
| `segmenter.pt` | part segmenter | inference |
| `plr.pt` | Part Localization Refiner | inference |
| `gate_20261301.pt`, `gate_20261302.pt`, `gate_20261303.pt` | query gate, three seeds | inference |
| `direct_20261901.pt`, `direct_20261911.pt`, `direct_20261921.pt` | direct-route evidence heads trained with GRS, three seeds | inference |
| `segmenter_fold0.pt`, `segmenter_fold1.pt` | cross-fitting fold heads of the segmenter | training PLR |

Seed k of the model pairs `gate_2026130k.pt` with the k-th direct head; `evaluate.py` runs all three.

## Evaluation

```bash
bash scripts/evaluate_all.sh
```

or for one dataset (`partnext` or `artiverse`):

```bash
python scripts/evaluate.py --dataset partnext
python scripts/score.py --dataset partnext --out outputs/partnext_scores.json
```

`evaluate.py` runs both routes on four pose passes for the three recipe seeds, and `score.py` reports pointing, reorientation, whole-object rotation and severe error rates as mean and standard deviation over the seeds. Part descriptions are mapped to part names with the stored mapper answers in `data/expressions.json`.

## Citation

```bibtex
@article{hou2026georead3d,
  title   = {GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation},
  author  = {Hou, Zhangcheng and Tonkich, Kevin and Ohtsuki, Tomoaki},
  journal = {arXiv preprint arXiv:2611.30519},
  year    = {2026}
}
```
