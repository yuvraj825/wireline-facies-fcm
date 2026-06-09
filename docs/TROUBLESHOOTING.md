# Troubleshooting Guide

## Issue 1 — `python3` not recognised (Windows)
Use `python` instead. Always run inside the conda environment:
```powershell
conda activate wireline-fcm
python src/pipeline.py --data data/train.csv --out outputs/
```

## Issue 2 — FCM_Vsh = 1.0 everywhere
**Cause:** SHALE_CLUSTERS includes all 7 clusters.
**Fix:** In notebook 04 Cell 1, hardcode after the auto-detection block:
```python
SHALE_CLUSTERS = [0, 2, 3, 4, 5, 6]
SAND_CLUSTERS  = [1]
```

## Issue 3 — NTG_FCM ≈ 0.017–0.048 (too low)
**Cause:** Using `Vsh ≤ 0.5` cutoff on compressed FCM_Vsh scale.
**Fix:** Use Sand_prob directly:
```python
NTG_FCM = (master['Sand_prob'] >= 0.06).mean()
```

## Issue 4 — C=3 selected automatically
**Cause:** Shale-dominated data produces flat PC/FPE curves.
**Fix:** Edit `outputs/data/cluster_selection.json` → `"C_OPTIMAL": 7`
Then re-run notebook 03b.

## Issue 5 — `scaler.pkl` not found
Run notebooks in strict order: 01 → 02 → 03a → 03b → 04 → 05.
Notebook 02 creates `scaler.pkl`. Notebook 03b creates `fcm_centroids.npy`.

## Issue 6 — Memory error in notebook 03b
Subsample the training data at the top of Cell 1:
```python
master_sub = master.sample(500_000, random_state=42)
X_scaled   = scaler.transform(master_sub[FEAT_COLS].values)
```

## Issue 7 — `skfuzzy` import error
```bash
pip install scikit-fuzzy==0.4.2
```

## Issue 8 — cmeans_predict ValueError (shape mismatch)
`cntr` must NOT be transposed. Correct call:
```python
fuzz.cluster.cmeans_predict(
    X_scaled.T,   # (n_features, n_samples)
    cntr,         # (n_clusters, n_features) — do NOT write cntr.T
    m=2.0, error=0.005, maxiter=1000
)
```
