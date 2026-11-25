# GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation

Zhangcheng Hou, Kevin Tonkich, Tomoaki Ohtsuki

[[arXiv:2611.30519]](https://arxiv.org/abs/2611.30519)

Language often refers to an object by its parts: point to the seat of a chair, or turn the chair so that its seat and base lie in given directions. Both tasks need the direction from the object's center to the part's center. GeoRead3D learns point evidence whose weighted centroid gives the part's location, and trains it with **Geometric Readout Supervision (GRS)**, which supervises the centroid's position and adds direction supervision on the parts used in reorientation commands. **Dual-Route Part Evidence** combines refined segmentation evidence with directly predicted point evidence, averaged over four rotated copies of the input.

This repository contains the final GeoRead3D model, the scripts that train each of its modules, and the evaluation on the PartNeXt test split and the Artiverse sample.

## Results

Success and severe error rates in percent on rotated input, mean ± standard deviation over three seeds (Table 1 of the paper).

| Dataset | Pointing (≤ 22.5°) | Reorientation (≤ 30°) | Whole-object rotation (≤ 30°) | Severe errors (> 90°) |
|---|---|---|---|---|
| PartNeXt test split | 74.1 ± 0.4 | 71.2 ± 0.3 | 65.3 ± 0.4 | 3.0 ± 0.1 |
| Artiverse | 60.0 ± 0.8 | 34.9 ± 2.7 | 32.2 ± 2.3 | 12.7 ± 0.8 |

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

Model weights are not included in this repository.

## Citation

```bibtex
@article{hou2026georead3d,
  title   = {GeoRead3D: Grounding Part Descriptions for 3D Pointing and Reorientation},
  author  = {Hou, Zhangcheng and Tonkich, Kevin and Ohtsuki, Tomoaki},
  journal = {arXiv preprint arXiv:2611.30519},
  year    = {2026}
}
```
