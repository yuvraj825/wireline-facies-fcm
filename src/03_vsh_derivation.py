#!/usr/bin/env python3
"""
================================================================================
03_vsh_derivation.py — FCM Membership → Continuous Vsh + NTG
================================================================================
Converts the FCM membership matrix from Part 2 into a continuous shale volume
(Vsh) curve for each test well. Also computes GR-derived Vsh as a physics-based
baseline using the same GR cutoffs calibrated from Raniganj Basin outcrops.

The key insight:
  • Vsh_FCM[i] = Σ_j u_test[j, i]  for j in shale-type clusters
  • This is naturally bounded [0,1] because Σ_j u_test[j, i] = 1 for all i
  • A depth sample at the sand/shale boundary gets ~0.5 in sand and ~0.5 in
    shale clusters → Vsh_FCM ≈ 0.5  [physically correct, gradational]
  • A GR hard cutoff would call it sand (Vsh=0) or shale (Vsh=1) — wrong for
    transitional zones.

GR cutoffs (calibrated to Raniganj Barakar Formation, IIT Kharagpur 2024):
  GR_CLEAN = 30 API  (pure sandstone proxy)
  GR_SHALE = 120 API (pure shale proxy)
  Vsh_GR   = (GR − 30) / (120 − 30), clipped to [0, 1]
  NTG cutoff: Vsh > 0.5 → non-reservoir (GR > 75 API equivalent)

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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Constants ─────────────────────────────────────────────────────────────────

GR_CLEAN  = 30.0    # API — clean sandstone (Raniganj field calibration)
GR_SHALE  = 120.0   # API — pure shale
VSH_NTG_CUTOFF = 0.5   # Vsh ≤ 0.5 → reservoir (GR ≤ 75 API equivalent)

SHALE_LITHOS = {"Shale", "Sandstone/Shale", "Marl"}
SAND_LITHOS  = {"Sandstone"}
COAL_LITHOS  = {"Coal"}

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
    "Coal":            "#1a1a1a",
    "Basement":        "#8B0000",
}

HOLDOUT_WELLS = [
    "16/1-6 A", "31/3-2", "34/5-1 A", "34/7-20", "16/7-5",
    "31/2-9", "25/6-2", "16/10-1", "15/9-13", "25/8-7",
]


# ── Step 3.1: GR-derived Vsh baseline ────────────────────────────────────────

def compute_vsh_gr(gr: np.ndarray) -> np.ndarray:
    """Standard linear GR index, calibrated to Raniganj pseudo-GR cutoffs."""
    vsh = (gr - GR_CLEAN) / (GR_SHALE - GR_CLEAN)
    return np.clip(vsh, 0.0, 1.0)


# ── Core Vsh from FCM ─────────────────────────────────────────────────────────

def derive_vsh_fcm(u_test: np.ndarray,
                   cluster_geo_map: dict) -> np.ndarray:
    """
    Vsh_FCM[i] = total membership of sample i in shale-type clusters.
    Naturally bounded [0,1].  Captures gradational transitions that GR
    hard cutoffs misclassify.
    """
    shale_clusters = [j for j, name in cluster_geo_map.items()
                      if name in SHALE_LITHOS]
    if not shale_clusters:
        # Fallback: use all non-sand clusters
        sand_clusters  = [j for j, name in cluster_geo_map.items()
                          if name in SAND_LITHOS]
        shale_clusters = [j for j in cluster_geo_map
                          if j not in sand_clusters]

    vsh_fcm = np.zeros(u_test.shape[1], dtype=float)
    for j in shale_clusters:
        vsh_fcm += u_test[j]

    print(f"  Shale-type clusters: {shale_clusters}  "
          f"→  {[cluster_geo_map[j] for j in shale_clusters]}")
    return np.clip(vsh_fcm, 0.0, 1.0)


# ── Step 3.2: Per-well NTG from FCM Vsh ──────────────────────────────────────

def compute_ntg(vsh: np.ndarray) -> float:
    """NTG = fraction of samples with Vsh ≤ VSH_NTG_CUTOFF."""
    return float((vsh <= VSH_NTG_CUTOFF).sum() / len(vsh))


# ── Step 3.3: Build per-well DataFrame ───────────────────────────────────────

def build_per_well_vsh(df_test: pd.DataFrame,
                        vsh_fcm: np.ndarray,
                        vsh_gr:  np.ndarray,
                        hard_labels: np.ndarray) -> pd.DataFrame:
    """Assemble a single DataFrame with all Vsh columns for all test wells."""
    depth_col = "DEPTH_MD" if "DEPTH_MD" in df_test.columns else "DEPTH_MD"
    if depth_col not in df_test.columns:
        # Try common alternatives
        for alt in ["DEPTH", "MD", "depth_m"]:
            if alt in df_test.columns:
                depth_col = alt
                break
        else:
            df_test["DEPTH_MD"] = np.arange(len(df_test), dtype=float)
            depth_col = "DEPTH_MD"

    out = pd.DataFrame({
        "WELL":           df_test["WELL"].values,
        "DEPTH":          df_test[depth_col].values,
        "GR":             df_test["GR"].values,
        "RHOB":           df_test["RHOB"].values,
        "NPHI":           df_test["NPHI"].values,
        "DTC":            df_test["DTC"].values,
        "RDEP":           df_test["RDEP"].values,
        "Vsh_GR":         vsh_gr,
        "Vsh_FCM":        vsh_fcm,
        "hard_cluster":   hard_labels,
        "LITHOLOGY_TRUE": df_test["LITHOLOGY"].values,
    })
    return out


# ── Step 3.4: Single-well log display (best well) ────────────────────────────

def plot_single_well(df_vsh: pd.DataFrame, well_name: str, out_dir: str):
    """
    Dual-track log display for well_name:
      Track 1: GR log (colour fill — green=sand, grey=shale)
      Track 2: Vsh_GR (grey) vs Vsh_FCM (blue) overlaid + lithology strip
    """
    wdf = df_vsh[df_vsh["WELL"] == well_name].copy()
    if wdf.empty:
        print(f"  WARNING: Well {well_name} not found in test data — skipping plot.")
        return

    depth = wdf["DEPTH"].values
    gr    = wdf["GR"].values
    vsh_gr  = wdf["Vsh_GR"].values
    vsh_fcm = wdf["Vsh_FCM"].values
    litho   = wdf["LITHOLOGY_TRUE"].values

    fig = plt.figure(figsize=(10, 14))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[0.5, 2, 2], wspace=0.08)

    # ── Panel 0: Lithology strip ─────────────────────────────────────────
    ax_lith = fig.add_subplot(gs[0, 0])
    _depth_lith_strip(ax_lith, depth, litho)
    ax_lith.set_ylabel("Depth (m)", fontsize=10)
    ax_lith.set_title("Lith.", fontsize=9)
    ax_lith.yaxis.set_label_position("left")

    # ── Panel 1: GR log with colour fill ─────────────────────────────────
    ax_gr = fig.add_subplot(gs[0, 1], sharey=ax_lith)
    gr_sand_thresh = 75.0   # NTG GR cutoff
    # Green fill where GR < 75 (clean sand side)
    ax_gr.fill_betweenx(depth, 0, gr,
                        where=(gr <= gr_sand_thresh),
                        color="#90EE90", alpha=0.8, label="Sand (GR≤75)")
    ax_gr.fill_betweenx(depth, 0, gr,
                        where=(gr > gr_sand_thresh),
                        color="#C0C0C0", alpha=0.7, label="Shale (GR>75)")
    ax_gr.plot(gr, depth, "k-", lw=0.5)
    ax_gr.set_xlim(0, 200)
    ax_gr.set_xlabel("GR (API)")
    ax_gr.set_title("Gamma Ray")
    ax_gr.axvline(gr_sand_thresh, color="red", ls="--", lw=1, alpha=0.7,
                  label="GR=75 cutoff")
    ax_gr.legend(fontsize=7, loc="upper right")
    ax_gr.grid(axis="x", alpha=0.3)
    ax_gr.invert_yaxis()
    ax_gr.tick_params(labelleft=False)

    # ── Panel 2: Vsh_GR vs Vsh_FCM ───────────────────────────────────────
    ax_vsh = fig.add_subplot(gs[0, 2], sharey=ax_lith)
    ax_vsh.plot(vsh_gr,  depth, color="dimgray",   lw=1.2, alpha=0.7,
                label="Vsh_GR (linear index)")
    ax_vsh.plot(vsh_fcm, depth, color="steelblue", lw=1.8,
                label="Vsh_FCM (fuzzy)")
    ax_vsh.fill_betweenx(depth, vsh_gr, vsh_fcm,
                         where=(vsh_fcm < vsh_gr),
                         color="lightblue", alpha=0.35,
                         label="FCM < GR (optimistic sand)")
    ax_vsh.fill_betweenx(depth, vsh_gr, vsh_fcm,
                         where=(vsh_fcm > vsh_gr),
                         color="salmon", alpha=0.30,
                         label="FCM > GR (more shale)")
    ax_vsh.axvline(0.5, color="red", ls="--", lw=1, alpha=0.7,
                   label="NTG cutoff (Vsh=0.5)")
    ax_vsh.set_xlim(0, 1.05)
    ax_vsh.set_xlabel("Vsh")
    ax_vsh.set_title("Vsh: GR vs FCM")
    ax_vsh.legend(fontsize=7, loc="upper right")
    ax_vsh.grid(axis="x", alpha=0.3)
    ax_vsh.tick_params(labelleft=False)

    fig.suptitle(
        f"Well: {well_name}\nGR Log + Vsh Comparison (FCM vs GR linear index)",
        fontsize=12, fontweight="bold",
    )

    safe = well_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, "plots", f"vsh_single_well_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def _depth_lith_strip(ax, depth, litho):
    """Draw a colour-coded lithology strip."""
    for i in range(len(depth) - 1):
        colour = LITHO_COLOURS.get(litho[i], "#DDDDDD")
        ax.fill_betweenx([depth[i], depth[i + 1]], 0, 1, color=colour)
    # last interval
    if len(depth) > 0:
        colour = LITHO_COLOURS.get(litho[-1], "#DDDDDD")
        step = (depth[-1] - depth[-2]) if len(depth) > 1 else 1.0
        ax.fill_betweenx([depth[-1], depth[-1] + step], 0, 1, color=colour)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()


# ── Save NTG JSON ─────────────────────────────────────────────────────────────

def save_ntg_json(ntg_data: dict, out_dir: str):
    path = os.path.join(out_dir, "metrics", "ntg_fcm.json")
    with open(path, "w") as f:
        json.dump(ntg_data, f, indent=2)
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(out_dir: str, shared: dict = None) -> dict:
    print("\n" + "─" * 60)
    print("  MODULE 3 — Vsh Derivation")
    print("─" * 60)

    metrics_dir = os.path.join(out_dir, "metrics")

    # Load artifacts
    if shared and "u_test" in shared:
        u_test          = shared["u_test"]
        hard_labels     = shared["hard_labels"]
        cluster_geo_map = shared["cluster_geo_map"]
        df_test         = shared["df_test"]
    else:
        u_test = np.load(os.path.join(metrics_dir, "u_test.npy"))
        hard_labels = np.load(os.path.join(metrics_dir, "hard_labels.npy"))
        with open(os.path.join(metrics_dir, "cluster_geo_map.pkl"), "rb") as f:
            cluster_geo_map = pickle.load(f)
        with open(os.path.join(metrics_dir, "df_test.pkl"), "rb") as f:
            df_test = pickle.load(f)

    df_test = df_test.reset_index(drop=True)

    # Compute Vsh
    vsh_fcm = derive_vsh_fcm(u_test, cluster_geo_map)
    vsh_gr  = compute_vsh_gr(df_test["GR"].values)

    # Build full DataFrame
    df_vsh = build_per_well_vsh(df_test, vsh_fcm, vsh_gr, hard_labels)

    # Per-well NTG
    ntg_data = {}
    wells_present = [w for w in HOLDOUT_WELLS if w in df_vsh["WELL"].values]
    for well in wells_present:
        mask = df_vsh["WELL"] == well
        ntg_data[well] = {
            "ntg_fcm":  round(compute_ntg(vsh_fcm[mask.values]), 4),
            "ntg_gr":   round(compute_ntg(vsh_gr[mask.values]),  4),
            "n_samples": int(mask.sum()),
        }
    save_ntg_json(ntg_data, out_dir)

    # Save per-well CSVs
    for well in wells_present:
        wdf = df_vsh[df_vsh["WELL"] == well]
        safe = well.replace("/", "_").replace(" ", "_")
        wdf.to_csv(
            os.path.join(metrics_dir, f"vsh_per_well_{safe}.csv"),
            index=False
        )
    print(f"  Saved per-well Vsh CSVs for {len(wells_present)} wells")

    # Save combined df_vsh for downstream modules
    with open(os.path.join(metrics_dir, "df_vsh.pkl"), "wb") as f:
        pickle.dump(df_vsh, f)

    # Plot best well
    best_well = "16/1-6 A" if "16/1-6 A" in wells_present else (
        wells_present[0] if wells_present else None
    )
    if best_well:
        plot_single_well(df_vsh, best_well, out_dir)

    print(f"\n  NTG summary:")
    for well, vals in ntg_data.items():
        print(f"    {well:<20s}  NTG_FCM={vals['ntg_fcm']:.3f}  "
              f"NTG_GR={vals['ntg_gr']:.3f}")

    print("  MODULE 3 complete.\n")
    return {
        "vsh_fcm":    vsh_fcm,
        "vsh_gr":     vsh_gr,
        "df_vsh":     df_vsh,
        "ntg_data":   ntg_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCM Lithofacies — Part 3: Vsh Derivation")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()
    run(args.out)
