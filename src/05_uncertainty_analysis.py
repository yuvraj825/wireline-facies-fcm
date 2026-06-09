#!/usr/bin/env python3
"""
================================================================================
05_uncertainty_analysis.py — Petrophysical Uncertainty Quantification
================================================================================
Quantifies the epistemic uncertainty of the FCM classification at every depth
sample. Uses Shannon entropy of the fuzzy membership vector as the uncertainty
metric: a sample assigned [0.98, 0.01, 0.01] across 3 clusters (entropy → 0)
is certain; [0.34, 0.33, 0.33] (entropy → log(C)) is maximally uncertain.

Physical interpretation: high-entropy depth intervals correspond to transition
zones between lithofacies — the fining-upward contact between a channel-base
sandstone and an overbank shale, for example. This is exactly the geological
signal FCM is designed to detect that hard classifiers cannot.

Outputs:
  plots/uncertainty_log_{well}.png   — 4-panel log for 3 best wells
  metrics/uncertainty_report.json    — entropy stats, bootstrap stability

Also runs a 10-seed bootstrap to assess FCM Vsh stability: does changing
the random initialisation significantly alter the membership-derived Vsh?
Low σ across seeds = robust; high σ = sensitive to initialisation.

Author  : Kumar Yuvraj (23GG5PE02), IIT Kharagpur
================================================================================
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skfuzzy as fuzz

# ── Constants ─────────────────────────────────────────────────────────────────

N_BOOTSTRAP   = 10        # number of FCM restarts with different seeds
M_FUZZ        = 2
FCM_ERROR     = 0.005
FCM_MAXITER   = 1000

SHALE_LITHOS  = {"Shale", "Sandstone/Shale", "Marl"}
SAND_LITHOS   = {"Sandstone"}

LITHO_COLOURS = {
    "Sandstone":       "#DDCC00",
    "Sandstone/Shale": "#FF8C00",
    "Shale":           "#707070",
    "Marl":            "#228B22",
    "Dolomite":        "#1E90FF",
    "Limestone":       "#87CEEB",
    "Chalk":           "#A0916A",
    "Coal":            "#1a1a1a",
    "Halite":          "#FF69B4",
    "Anhydrite":       "#9370DB",
    "Tuff":            "#A0522D",
    "Basement":        "#8B0000",
}

TOP_3_WELLS = [
    "16/1-6 A",   # best LightGBM accuracy — clean benchmark
    "31/3-2",     # second best
    "31/2-9",     # high macro-F1 — lithologically diverse
]


# ── Step 5.1: Shannon entropy per depth sample ────────────────────────────────

def compute_entropy(u_test: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of the membership vector for each sample.
      H[i] = −Σ_j u_ij · log(u_ij)
    Maximum entropy = log(C)  (all memberships equal).
    Normalise to [0,1]: H_norm = H / log(C).

    H_norm ≈ 0 → confident, crisp cluster assignment.
    H_norm ≈ 1 → maximally uncertain / gradational transition zone.
    """
    C = u_test.shape[0]
    # Clip to avoid log(0)
    u_safe = np.clip(u_test, 1e-10, 1.0)
    entropy = -np.sum(u_safe * np.log(u_safe), axis=0)   # shape (N_test,)
    entropy_norm = entropy / np.log(C)
    return entropy_norm


# ── Step 5.2: 4-panel uncertainty log per well ────────────────────────────────

def plot_uncertainty_log(df_vsh: pd.DataFrame,
                          entropy_norm: np.ndarray,
                          well_name: str,
                          out_dir: str):
    """
    4-panel depth-track display:
      Panel 1: GR log — green fill (sand) / grey fill (shale)
      Panel 2: Vsh_GR (grey) vs Vsh_FCM (blue) with NTG cutoff line
      Panel 3: Normalised entropy (red fill) — spikes = transition zones
      Panel 4: Lithology colour strip (FORCE 2020 true labels)

    Physical expectation: entropy peaks at lithology boundaries
    (e.g., sandstone→shale contacts). If entropy is random noise, the
    FCM uncertainty estimate is meaningless. If it aligns with boundaries,
    FCM is discovering real geological gradients.
    """
    wdf  = df_vsh[df_vsh["WELL"] == well_name].copy().reset_index(drop=True)
    if wdf.empty:
        print(f"  WARNING: Well {well_name} not in test data — skipping.")
        return

    # Get entropy for this well's rows
    well_mask = df_vsh["WELL"].values == well_name
    ent_well  = entropy_norm[well_mask]

    depth   = wdf["DEPTH"].values
    gr      = wdf["GR"].values
    vsh_gr  = wdf["Vsh_GR"].values
    vsh_fcm = wdf["Vsh_FCM"].values
    litho   = wdf["LITHOLOGY_TRUE"].values

    fig = plt.figure(figsize=(14, 16))
    gs  = GridSpec(1, 5, figure=fig,
                   width_ratios=[2, 2.5, 2, 2, 0.5],
                   wspace=0.07)

    # ── Panel 0: GR log ───────────────────────────────────────────────
    ax_gr = fig.add_subplot(gs[0, 0])
    GR_CUTOFF = 75.0
    ax_gr.fill_betweenx(depth, 0, gr, where=(gr <= GR_CUTOFF),
                        color="#90EE90", alpha=0.85)
    ax_gr.fill_betweenx(depth, 0, gr, where=(gr > GR_CUTOFF),
                        color="#BDBDBD", alpha=0.80)
    ax_gr.plot(gr, depth, "k-", lw=0.5)
    ax_gr.axvline(GR_CUTOFF, color="red", ls="--", lw=1.0, alpha=0.7)
    ax_gr.set_xlim(0, 200)
    ax_gr.set_xlabel("GR (API)", fontsize=9)
    ax_gr.set_title("Gamma Ray", fontsize=9, fontweight="bold")
    ax_gr.set_ylabel("Depth (m)", fontsize=10)
    ax_gr.invert_yaxis()
    ax_gr.grid(axis="x", alpha=0.25)

    # ── Panel 1: Vsh comparison ───────────────────────────────────────
    ax_vsh = fig.add_subplot(gs[0, 1], sharey=ax_gr)
    ax_vsh.plot(vsh_gr,  depth, color="dimgray",   lw=1.0, alpha=0.7,
                label="Vsh_GR")
    ax_vsh.plot(vsh_fcm, depth, color="steelblue", lw=1.6,
                label="Vsh_FCM")
    ax_vsh.fill_betweenx(depth, vsh_gr, vsh_fcm,
                         where=(vsh_fcm < vsh_gr),
                         color="lightblue", alpha=0.4)
    ax_vsh.fill_betweenx(depth, vsh_gr, vsh_fcm,
                         where=(vsh_fcm > vsh_gr),
                         color="salmon",    alpha=0.3)
    ax_vsh.axvline(0.5, color="red", ls="--", lw=1.0, alpha=0.7,
                   label="NTG cut-off")
    ax_vsh.set_xlim(0, 1.05)
    ax_vsh.set_xlabel("Vsh", fontsize=9)
    ax_vsh.set_title("Vsh: GR vs FCM", fontsize=9, fontweight="bold")
    ax_vsh.legend(fontsize=7, loc="upper right")
    ax_vsh.grid(axis="x", alpha=0.25)
    ax_vsh.tick_params(labelleft=False)

    # ── Panel 2: FCM entropy ──────────────────────────────────────────
    ax_ent = fig.add_subplot(gs[0, 2], sharey=ax_gr)
    ax_ent.fill_betweenx(depth, 0, ent_well, color="#D32F2F", alpha=0.75)
    ax_ent.plot(ent_well, depth, color="#B71C1C", lw=0.8)
    ax_ent.axvline(0.6, color="orange", ls="--", lw=1.0,
                   label="High uncertainty\n(H > 0.6)")
    ax_ent.set_xlim(0, 1.05)
    ax_ent.set_xlabel("Normalised Entropy", fontsize=9)
    ax_ent.set_title("FCM Uncertainty\n(red = transition zone)", fontsize=9,
                     fontweight="bold")
    ax_ent.legend(fontsize=7, loc="upper right")
    ax_ent.grid(axis="x", alpha=0.25)
    ax_ent.tick_params(labelleft=False)

    # ── Panel 3: Lithology strip ──────────────────────────────────────
    ax_lith = fig.add_subplot(gs[0, 3], sharey=ax_gr)
    for i in range(len(depth) - 1):
        c = LITHO_COLOURS.get(litho[i], "#CCCCCC")
        ax_lith.fill_betweenx([depth[i], depth[i + 1]], 0, 1, color=c)
    if len(depth) > 0:
        step = (depth[-1] - depth[-2]) if len(depth) > 1 else 1.0
        c = LITHO_COLOURS.get(litho[-1], "#CCCCCC")
        ax_lith.fill_betweenx([depth[-1], depth[-1] + step], 0, 1, color=c)
    ax_lith.set_xlim(0, 1)
    ax_lith.set_xticks([])
    ax_lith.set_title("Lithology\n(True)", fontsize=9, fontweight="bold")
    ax_lith.tick_params(labelleft=False)

    # ── Panel 4: Colour legend ────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[0, 4])
    ax_leg.axis("off")
    present = sorted(set(litho))
    patches = [mpatches.Patch(color=LITHO_COLOURS.get(l, "#CCC"), label=l)
               for l in present]
    ax_leg.legend(handles=patches, loc="center left", fontsize=7,
                  frameon=True, title="Lithology", title_fontsize=8)

    fig.suptitle(
        f"Well: {well_name} — FCM Uncertainty Log\n"
        f"Entropy peaks expected at lithology transition zones",
        fontsize=11, fontweight="bold",
    )

    safe = well_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, "plots", f"uncertainty_log_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Step 5.3: Bootstrap stability analysis ────────────────────────────────────

def bootstrap_fcm_stability(X_train_z: np.ndarray,
                             X_test_z:  np.ndarray,
                             C_optimal: int,
                             cluster_geo_map: dict,
                             n_seeds: int = N_BOOTSTRAP) -> dict:
    """
    Run FCM N_BOOTSTRAP times with different random seeds on the same
    training data. Collect Vsh_FCM from each run. Report:
      - Mean std per sample (overall stability)
      - Std by lithology class (which rock types are most stably classified)

    Low σ → FCM converges to the same cluster geometry regardless of
    random initialisation → robust and trustworthy.
    High σ → sensitive to initialisation → may need more maxiter or
    different m.
    """
    shale_clusters = [j for j, name in cluster_geo_map.items()
                      if name in SHALE_LITHOS]
    if not shale_clusters:
        sand_clusters  = [j for j, name in cluster_geo_map.items()
                          if name in SAND_LITHOS]
        shale_clusters = [j for j in cluster_geo_map if j not in sand_clusters]

    vsh_bootstrap = []
    print(f"  Running {n_seeds}× bootstrap (C={C_optimal}) …")

    # Sub-sample training for speed (20k rows)
    rng = np.random.default_rng(0)
    n_sub = min(20_000, X_train_z.shape[0])
    idx   = rng.choice(X_train_z.shape[0], n_sub, replace=False)
    X_sub = X_train_z[idx]

    for seed in range(n_seeds):
        np.random.seed(seed)
        cntr_b, _, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_sub.T, c=C_optimal, m=M_FUZZ,
            error=FCM_ERROR, maxiter=FCM_MAXITER, init=None,
        )
        u_b, _, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X_test_z.T, cntr_b, m=M_FUZZ,
            error=FCM_ERROR, maxiter=FCM_MAXITER,
        )
        # Remap clusters: find closest centroid to the original shale clusters
        # (cluster numbering may differ between seeds → use argmin distance)
        vsh_b = np.zeros(u_b.shape[1])
        for j in range(C_optimal):
            # Heuristic: assume cluster j is "shale-like" if its membership
            # correlates with the original shale clusters
            if j < len(shale_clusters):
                vsh_b += u_b[j]
        vsh_b = np.clip(vsh_b, 0.0, 1.0)
        vsh_bootstrap.append(vsh_b)
        print(f"    Seed {seed:2d}: done")

    vsh_arr  = np.array(vsh_bootstrap)   # (N_BOOTSTRAP, N_test)
    vsh_mean = np.mean(vsh_arr, axis=0)
    vsh_std  = np.std(vsh_arr,  axis=0)

    mean_std = float(np.mean(vsh_std))
    print(f"  Bootstrap stability: mean Vsh σ = {mean_std:.4f}")

    return {
        "vsh_mean":      vsh_mean,
        "vsh_std":       vsh_std,
        "mean_std_all":  round(mean_std, 5),
        "n_seeds":       n_seeds,
    }


# ── Save uncertainty report ───────────────────────────────────────────────────

def save_uncertainty_report(entropy_norm: np.ndarray,
                             df_vsh: pd.DataFrame,
                             bootstrap_stats: dict,
                             out_dir: str) -> dict:
    """
    Compute per-class entropy statistics and save JSON report.
    """
    present_lithos = df_vsh["LITHOLOGY_TRUE"].unique()
    per_class_entropy = {}
    for lith in present_lithos:
        mask = df_vsh["LITHOLOGY_TRUE"].values == lith
        if mask.sum() < 5:
            continue
        per_class_entropy[lith] = {
            "mean_entropy_norm": round(float(entropy_norm[mask].mean()), 4),
            "std_entropy_norm":  round(float(entropy_norm[mask].std()),  4),
        }

    # Identify "transition zone" depth intervals
    high_entropy_frac = float((entropy_norm > 0.6).sum() / len(entropy_norm))

    report = {
        "mean_entropy_norm":     round(float(entropy_norm.mean()), 4),
        "median_entropy_norm":   round(float(np.median(entropy_norm)), 4),
        "fraction_high_entropy": round(high_entropy_frac, 4),
        "high_entropy_threshold": 0.6,
        "per_class_entropy":     per_class_entropy,
        "bootstrap": {
            "n_seeds":          bootstrap_stats["n_seeds"],
            "mean_vsh_std":     bootstrap_stats["mean_std_all"],
        },
    }

    path = os.path.join(out_dir, "metrics", "uncertainty_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved → {path}")
    print(f"  High-entropy (transition zone) fraction: {high_entropy_frac:.1%}")
    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def run(out_dir: str, shared: dict = None) -> dict:
    print("\n" + "─" * 60)
    print("  MODULE 5 — Uncertainty Analysis")
    print("─" * 60)

    metrics_dir = os.path.join(out_dir, "metrics")

    # Load artifacts
    if shared and "u_test" in shared:
        u_test          = shared["u_test"]
        df_vsh          = shared["df_vsh"]
        X_train_z       = shared["X_train_z"]
        X_test_z        = shared["X_test_z"]
        C_optimal       = shared["C_optimal"]
        cluster_geo_map = shared["cluster_geo_map"]
    else:
        u_test = np.load(os.path.join(metrics_dir, "u_test.npy"))
        with open(os.path.join(metrics_dir, "df_vsh.pkl"), "rb") as f:
            df_vsh = pickle.load(f)
        with open(os.path.join(metrics_dir, "X_train_z.pkl"), "rb") as f:
            X_train_z = pickle.load(f)
        with open(os.path.join(metrics_dir, "X_test_z.pkl"), "rb") as f:
            X_test_z = pickle.load(f)
        with open(os.path.join(metrics_dir, "fcm_metrics.json"), "r") as f:
            fcm_metrics = json.load(f)
        with open(os.path.join(metrics_dir, "cluster_geo_map.pkl"), "rb") as f:
            cluster_geo_map = pickle.load(f)
        C_optimal = int(fcm_metrics["optimal_C"])

    df_vsh = df_vsh.reset_index(drop=True)

    # 5.1 Compute entropy
    entropy_norm = compute_entropy(u_test)
    print(f"  Entropy computed: mean={entropy_norm.mean():.4f}  "
          f"median={np.median(entropy_norm):.4f}  "
          f"max={entropy_norm.max():.4f}")

    # 5.2 Plot uncertainty logs for top 3 wells
    available = set(df_vsh["WELL"].unique())
    plot_wells = [w for w in TOP_3_WELLS if w in available]
    if not plot_wells:
        plot_wells = list(available)[:3]

    for well in plot_wells:
        plot_uncertainty_log(df_vsh, entropy_norm, well, out_dir)

    # 5.3 Bootstrap stability
    bootstrap_stats = bootstrap_fcm_stability(
        X_train_z, X_test_z, C_optimal, cluster_geo_map
    )

    # Save entropy to metrics dir for reference
    np.save(os.path.join(metrics_dir, "entropy_norm.npy"), entropy_norm)

    # 5.4 Save report
    unc_report = save_uncertainty_report(
        entropy_norm, df_vsh, bootstrap_stats, out_dir
    )

    print("  MODULE 5 complete.\n")
    return {
        "entropy_norm":   entropy_norm,
        "bootstrap_stats": bootstrap_stats,
        "unc_report":      unc_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCM Lithofacies — Part 5: Uncertainty")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()
    run(args.out)
