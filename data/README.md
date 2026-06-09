# Data Directory

## FORCE 2020 Dataset (required)

Place `train.csv` here before running any notebook.

**Download:**
```
https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
```

- Separator: `;` (semicolon — do NOT open and re-save in Excel)
- Size: ~200 MB | Rows: 1,170,511 | Wells: 98

**Columns used:** `WELL`, `DEPTH_MD`, `GR`, `RHOB`, `NPHI`, `DTC`, `RDEP`,
`FORCE_2020_LITHOFACIES_LITHOLOGY`

## Raniganj Pseudo-GR Data

No external file needed. Hardcoded from GPS-mapped field outcrops
(IIT Kharagpur, Raniganj Basin, 2024) in `notebooks/05_cross_validation.ipynb`.

| Section | Formation | NTG |
|---------|-----------|-----|
| Ramnagar Colliery | Barakar Fm | 0.500 |
| Shunuri Village | Panchet/Barakar | 0.571 |
| Duburdi Section | Barakar Fm | 0.190 |
