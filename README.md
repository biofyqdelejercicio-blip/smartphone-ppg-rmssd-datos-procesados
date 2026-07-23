# PPG interval reconstruction and RMSSD estimation at 30 and 60 Hz

Code, frozen models, derived results, and documentation associated with the manuscript **“Development and external validation under controlled rest of an algorithm to reconstruct pulse-to-pulse intervals and estimate heart rate and RMSSD from finger photoplethysmography.”**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21501595.svg)](https://doi.org/10.5281/zenodo.21501595)

## Scope

The study evaluated contact finger photoplethysmography acquired with instrumental sensors and resampled to 30 and 60 Hz. These sampling rates were selected for their compatibility with common smartphone frame rates. No RGB signal was acquired from a smartphone camera. The study therefore does not establish the validity of smartphone-camera PPG.

The frozen algorithm reconstructs pulse-to-pulse intervals from low-resolution PPG, applies an interval-quality classifier and correction model, and estimates heart rate and RMSSD. The intended domain is a seated, awake, still, 60-second recording. The UTSA office analysis is reported separately as an out-of-domain stress test.

## Main results

| Evaluation stage | Dataset | Accepted | RMSSD MAE (ms) | CCC |
|---|---|---:|---:|---:|
| Primary external validation | PTT-PPG, pleth_1 | 15/22 | 4.46 | 0.829 |
| Primary external validation | PTT-PPG, pleth_2 | 20/22 | 5.05 | 0.813 |
| Primary external validation | PTT-PPG, pleth_3 | 20/22 | 4.10 | 0.855 |
| Secondary external replication | Vollmer resting subset | 12/13 | 3.11 | 0.943 |
| Out-of-domain stress test | UTSA office, infrared | 14/60 | 28.68 | 0.201 |
| Out-of-domain stress test | UTSA office, red | 9/60 | 42.45 | 0.195 |

All values correspond to the primary SQI threshold of 0.65 fixed before external evaluation. Record-level outputs, participant-level bootstrap intervals, the SQI 0.75 sensitivity analysis, and the post hoc NN-index audit are available in [`results/`](results/).

## Repository contents

- `src/ppg_rmssd/`: reusable signal-processing and inference functions.
- `scripts/`: external evaluators, deterministic figure generation, and repository checks.
- `models/`: frozen 30 and 60 Hz Extra Trees models and metadata.
- `results/`: derived CSV outputs reported in the manuscript.
- `figures/`: figures generated from the archived derived results.
- `docs/METHODS.md`: frozen decision rules, dataset roles, and reproducibility notes.
- `data/README.md`: source-dataset access instructions. Third-party raw data are not redistributed.

## Computational environment

The analysis environment used Python 3.10.11. Compatibility is checked with Python 3.10 and 3.11. Package versions are defined by `pyproject.toml` and `requirements.txt`.

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e ".[test,figures]"
pytest
python scripts/verify_repository.py
```

The manuscript figures can be regenerated from the archived result tables:

```bash
python scripts/generate_figures.py
```

## External evaluation

The source datasets and their repositories are documented in [`data/README.md`](data/README.md). After local retrieval, the external evaluations are executed as follows:

```bash
python scripts/evaluate_ptt_ppg.py PATH_TO_PTT_PPG --output-dir outputs/ptt_ppg
python scripts/evaluate_vollmer.py PATH_TO_VOLLMER --output-dir outputs/vollmer
python scripts/evaluate_utsa_office.py PATH_TO_UTSA --output-dir outputs/utsa_office
```

The scripts use the frozen model in `models/frozen_candidate_v1_models.joblib.xz` unless another path is supplied. No external-validation record is used to refit the model or search for a threshold.

## Data availability and archiving

The repository contains only derived, non-identifying results. Raw physiological signals remain in their original repositories and are governed by the corresponding access and licensing terms. The version history is permanently archived in Zenodo under concept DOI [10.5281/zenodo.21501595](https://doi.org/10.5281/zenodo.21501595).

## Citation

Hernández-García F, Meléndez-Gallardo J, Camejo-Alvarez MÁ. *PPG interval reconstruction and RMSSD estimation at 30 and 60 Hz*. Version 1.0.2. Zenodo; 2026. [https://doi.org/10.5281/zenodo.21501595](https://doi.org/10.5281/zenodo.21501595).

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

The original code in this repository is distributed under the MIT License. The included derived results remain subject to citation of the original datasets. This license does not relicense third-party data.
