# Coastal Sea-Level Residual Forecasting

This repository contains the code, configurations, paper source, and selected result tables for 24-hour forecasting of non-tidal coastal sea-level residuals.

## Main models

- VARX-Ridge linear baseline
- Eta-only Graph WaveNet
- Multistate Graph WaveNet
- HS-DT-GWN fixed dual-expert combination
- C4 probabilistic residual-correction extension

The paper also reports matched physics-loss and ODE-prior controls. Their effects are conditional and mixed; they should not be interpreted as universal improvements.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The ordinary ensemble and fixed-scale controls can be rerun only after the saved prediction caches are placed in the paths described in `src/analysis/ensemble_scale_controls_20261009/run_controls.py`:

```powershell
python -u src/analysis/ensemble_scale_controls_20261009/run_controls.py
```

The script does not train models. It computes post-hoc controls from saved predictions and writes CSV files to its `results/` directory.

## Data

Raw data and large checkpoints are intentionally excluded. Download instructions, sources, station definitions, and preprocessing limits are in `docs/data/` and `docs/project_handoff/`.

## Paper

The LaTeX source is under `paper/`. Upload the contents of that directory to Overleaf and compile `main.tex`.

## Reproducibility limits

The main benchmark uses retrospective aligned forcing. Later temporal evaluations and external-network analyses have separate evidence status. Read `docs/project_handoff/00_START_HERE_CN.md` before interpreting results.
