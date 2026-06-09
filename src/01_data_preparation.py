#!/usr/bin/env python3
"""
================================================================================
01_data_preparation.py — FORCE 2020 Log Preprocessing
================================================================================
Loads the FORCE 2020 wireline dataset, selects and clips the five primary logs,
applies per-well median imputation, engineers two derived features (NPHI/RHOB
and GR/RHOB ratios), applies per-well Z-score normalisation, and performs a
well-based train/test split. Saves normalised arrays and the raw test DataFrame
for downstream FCM training and Vsh derivation.

Author  : Kumar Yuvraj (23GG5PE02), IIT Kharagpur
================================================================================
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

LITHO_MAP = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    88000: "Chalk",
    86000: "Halite",
    99000: "Anhydrite",
    90000: "Tuff",
    93000: "Coal",
    10000: "Basement",
}

LOG_BOUNDS = {
    "GR":   (0,    300),
    "RHOB": (1.0,  3.5),
    "NPHI": (-0.15, 1.0),
    "DTC":  (40,   250),
    "RDEP": (0.2, 2000),
}

PRIMARY_LOGS = ["GR", "RHOB", "NPHI", "DTC", "RDEP"]

FEATURE_COLS = [
    "GR", "RHOB", "NPHI", "DTC", "RDEP",
    "NPHI_RHOB_ratio", "GR_RHOB_ratio",
]

# Best-performing wells from the electrofacies LightGBM project — held out
# for Vsh curve generation and cross-validation with Raniganj pseudo-GR.
HOLDOUT_WELLS = [
    "16/1-6 A",   # LightGBM accuracy 91.1%
    "31/3-2",     # 83.8%
    "34/5-1 A",   # 83.3%
    "34/7-20",    # 82.3%
    "16/7-5",     # 82.2%
    "31/2-9",     # 82.0%
    "25/6-2",     # 81.7%
    "16/10-1",    # 81.1%
    "15/9-13",    # 80.5%
    "25/8-7",     # 80.5%
]

# Lithology colour map (consistent with electrofacies project)
LITHO_COLOURS = {
    "Sandstone":       "#FFFF00",
    "Sandstone/Shale": "#FFD700",
    "Shale":           "#808080",
    "Marl":            "#7CFC00",
    "Dolomite":        "#1E90FF",
    "Limestone":       "#ADD8E6",
    "Chalk":           "#F5F5DC",
    "Halite":          "#FF69B4",
    "Anhydrite":       "#DDA0DD",
    "Tuff":            "#A0522D",
    "Coal":            "#000000",
    "Basement":        "#8B0000",
}


# ── Step 1: Load and map lithology labels ─────────────────────────────────────

def load_force2020(csv_path: str) -> pd.DataFrame:
    """Load FORCE 2020 train.csv (sep=';') and map numeric lithology codes."""
    print(f"  Loading {csv_path} …")
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Map numeric lithology codes to string labels
    lith_col = "FORCE_2020_LITHOFACIES_LITHOLOGY"
    if lith_col not in df.columns:
        raise KeyError(f"Column '{lith_col}' not found. Check CSV separator and column names.")

    df["LITHOLOGY"] = df[lith_col].map(LITHO_MAP)
    df = df.dropna(subset=["LITHOLOGY"])
    print(f"  After lithology drop: {df.shape}  |  "
          f"{df['LITHOLOGY'].nunique()} classes  |  "
          f"{df['WELL'].nunique()} wells")
    return df


# ── Step 2: Clip primary logs to physical bounds ──────────────────────────────

def clip_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical bound clipping to the five primary wireline logs."""
    for col, (lo, hi) in LOG_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


# ── Step 3: Per-well median imputation ───────────────────────────────────────

def impute_per_well(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values with the per-well median; fall back to global median."""
    for col in PRIMARY_LOGS:
        if col not in df.columns:
            continue
        well_med = df.groupby("WELL")[col].transform("median")
        global_med = df[col].median()
        df[col] = df[col].fillna(well_med).fillna(global_med)
    return df


# ── Step 4: Derived features ──────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the two SHAP-ranked most important engineered features:
      • NPHI/RHOB — neutron-density crossplot (gas sand vs brine sand vs shale)
      • GR/RHOB   — clay content relative to bulk density
    No rolling statistics: FCM is point-based, window features would create
    artificial spatial correlations between cluster assignments.
    """
    df["NPHI_RHOB_ratio"] = df["NPHI"] / (df["RHOB"] + 1e-6)
    df["GR_RHOB_ratio"]   = df["GR"]   / (df["RHOB"] + 1e-6)
    return df


# ── Step 5: Per-well Z-score normalisation ────────────────────────────────────

def normalise_per_well(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each feature column per well.
    FCM is distance-based — without normalisation, GR (0–300) dominates
    Euclidean distances and RHOB (1–3.5) is effectively ignored.
    """
    for col in FEATURE_COLS:
        mu    = df.groupby("WELL")[col].transform("mean")
        sigma = df.groupby("WELL")[col].transform("std").replace(0, 1)
        df[col + "_z"] = (df[col] - mu) / sigma
    return df


# ── Step 6: Well-based train/test split ───────────────────────────────────────

def split_wells(df: pd.DataFrame):
    """
    Hold out HOLDOUT_WELLS as the test set (well-based split prevents data
    leakage across depth intervals — identical logic to the electrofacies
    LightGBM pipeline).
    """
    z_cols = [c + "_z" for c in FEATURE_COLS]

    test_mask  = df["WELL"].isin(HOLDOUT_WELLS)
    train_mask = ~test_mask

    # Warn if any holdout well is missing
    missing = set(HOLDOUT_WELLS) - set(df["WELL"].unique())
    if missing:
        print(f"  WARNING: {len(missing)} holdout wells not found in dataset: {missing}")

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    X_train_z = df_train[z_cols].values
    X_test_z  = df_test[z_cols].values

    print(f"  Training set : {len(df_train):,} samples  |  "
          f"{df_train['WELL'].nunique()} wells")
    print(f"  Test set     : {len(df_test):,} samples  |  "
          f"{df_test['WELL'].nunique()} wells  "
          f"({[w for w in HOLDOUT_WELLS if w in df['WELL'].unique()]})")

    return X_train_z, X_test_z, df_train, df_test


# ── Output 1: Data summary JSON ───────────────────────────────────────────────

def save_data_summary(df_train, df_test, out_dir: str):
    z_cols = [c + "_z" for c in FEATURE_COLS]
    class_dist = df_train["LITHOLOGY"].value_counts().to_dict()
    summary = {
        "n_wells_total":    int(df_train["WELL"].nunique() + df_test["WELL"].nunique()),
        "n_wells_train":    int(df_train["WELL"].nunique()),
        "n_wells_test":     int(df_test["WELL"].nunique()),
        "n_samples_train":  int(len(df_train)),
        "n_samples_test":   int(len(df_test)),
        "feature_cols_z":   z_cols,
        "primary_logs":     PRIMARY_LOGS,
        "derived_features": ["NPHI_RHOB_ratio", "GR_RHOB_ratio"],
        "holdout_wells":    HOLDOUT_WELLS,
        "class_distribution_train": {k: int(v) for k, v in class_dist.items()},
        "log_bounds":       {k: list(v) for k, v in LOG_BOUNDS.items()},
    }
    path = os.path.join(out_dir, "metrics", "data_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {path}")
    return summary


# ── Output 2: Log distribution histograms ─────────────────────────────────────

def plot_log_distributions(df: pd.DataFrame, out_dir: str, n_lithos: int = 6):
    """
    Per-log histogram coloured by the N most common lithology classes.
    Shows why hard cutoffs fail: distributions of different litho classes
    overlap heavily in every log.
    """
    top_lithos = (df["LITHOLOGY"].value_counts().head(n_lithos).index.tolist())
    df_plot    = df[df["LITHOLOGY"].isin(top_lithos)].copy()

    fig, axes = plt.subplots(1, len(PRIMARY_LOGS), figsize=(18, 5))
    fig.suptitle(
        "FORCE 2020 — Wireline Log Distributions by Lithology\n"
        "(overlapping distributions justify fuzzy vs. hard classification)",
        fontsize=12, fontweight="bold"
    )

    for ax, col in zip(axes, PRIMARY_LOGS):
        for lith in top_lithos:
            vals = df_plot.loc[df_plot["LITHOLOGY"] == lith, col].dropna()
            if len(vals) < 10:
                continue
            ax.hist(
                vals, bins=60, alpha=0.45, density=True,
                color=LITHO_COLOURS.get(lith, "#888888"),
                label=lith, histtype="stepfilled",
            )
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density" if col == PRIMARY_LOGS[0] else "")
        ax.set_title(col)
        lo, hi = LOG_BOUNDS[col]
        ax.set_xlim(lo, hi)
        ax.grid(alpha=0.3)

    # Shared legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_lithos,
               fontsize=9, frameon=True, title="Lithology")
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    path = os.path.join(out_dir, "plots", "log_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(csv_path: str, out_dir: str) -> dict:
    print("\n" + "─" * 60)
    print("  MODULE 1 — Data Preparation")
    print("─" * 60)

    # 1. Load
    df = load_force2020(csv_path)

    # 2. Clip
    df = clip_logs(df)

    # 3. Impute
    df = impute_per_well(df)

    # 4. Derived features
    df = add_derived_features(df)

    # 5. Z-score normalise
    df = normalise_per_well(df)

    # 6. Split
    X_train_z, X_test_z, df_train, df_test = split_wells(df)

    # 7. Save arrays and DataFrames
    artifacts_dir = os.path.join(out_dir, "metrics")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    with open(os.path.join(artifacts_dir, "X_train_z.pkl"), "wb") as f:
        pickle.dump(X_train_z, f)
    with open(os.path.join(artifacts_dir, "X_test_z.pkl"), "wb") as f:
        pickle.dump(X_test_z, f)
    with open(os.path.join(artifacts_dir, "df_test.pkl"), "wb") as f:
        pickle.dump(df_test.reset_index(drop=True), f)
    with open(os.path.join(artifacts_dir, "df_train.pkl"), "wb") as f:
        pickle.dump(df_train.reset_index(drop=True), f)

    print(f"  Saved normalised arrays and DataFrames to {artifacts_dir}")

    # 8. Outputs
    summary = save_data_summary(df_train, df_test, out_dir)
    plot_log_distributions(df, out_dir)

    print("  MODULE 1 complete.\n")
    return {
        "X_train_z": X_train_z,
        "X_test_z":  X_test_z,
        "df_train":  df_train,
        "df_test":   df_test,
        "summary":   summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCM Lithofacies — Part 1: Data Preparation")
    parser.add_argument("--data", required=True, help="Path to FORCE 2020 train.csv")
    parser.add_argument("--out",  default="outputs", help="Output directory")
    args = parser.parse_args()
    run(args.data, args.out)
