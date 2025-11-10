# FW-DTOS-Fuzzy (METR-LA)

Turn-key federated training on METR-LA with:
- **FedAvg** (baseline)
- **FedProx** (baseline)
- **TAR-Fed-Fuzzy** (proposed): Mamdani FIS trust-weighted aggregation with spatial diffusion

## Data
Place these files into `data/` (already provided in your workspace):
- `metr_la.h5`
- `sensor_locations_la.csv`
- `distances_la.csv`

If you already have them at `/mnt/data/`, `runall.py` will copy them into `data/` automatically.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python runall.py
```

## Outputs (in `results/`)
- CSV logs: per-round MAE, RMSE, Acc, MacroF1, Recall
- Confusion matrices (CSV) for each method
- PNG plots: learning curves, confusion matrices, trust dynamics heatmap (for TAR-Fed-Fuzzy)
- Aggregated comparison CSV

## Reproducibility
- Fixed seeds across NumPy/TF for repeatability
- Non-IID partition via KMeans on sensor coordinates (K clients), windows of length 12 (predict t+1)
