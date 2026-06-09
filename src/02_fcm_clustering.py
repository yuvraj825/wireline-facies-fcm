#!/usr/bin/env python3
"""
================================================================================
02_fcm_clustering.py — Fuzzy C-Means Training & Membership Prediction
================================================================================
Trains FCM on the normalised FORCE 2020 wireline logs. Finds the optimal number
of clusters C by minimising Fuzzy Partition Entropy (PE) and maximising the
Partition Coefficient (PC). Applies the trained model to the 10 hold-out test
wells to produce a per-depth membership matrix U_test (shape C × N_test).
Maps each FCM cluster to a geological lithology label by dominant-class vote.

Scientific rationale for FCM over K-Means:
  K-Means: hard assignment — every sample is 100% one cluster.
  FCM:     soft assignment — sample at sand/shale boundary gets ~0.5 membership
           in both the sand and shale clusters. This is physically correct for
           gradational fining-upward sequences (e.g., Ramnagar Colliery section).

Objective function minimised:
  J = Σ_i Σ_j (u_ij)^m × ‖x_i − v_j‖²
  where m=2 (standard fuzziness), u_ij ∈ [0,1], Σ_j u_ij = 1 for all i.

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
import matplotlib.colors as mcolors
import skfuzzy as fuzz

from sklearn.metrics import confusion_matrix

# ── Constants ─────────────────────────────────────────────────────────────────

C_RANGE    = range(3, 13)   # test C = 3, 4, … 12
M_FUZZ     = 2              # fuzziness parameter (standard; do not change)
FCM_ERROR  = 0.005          # convergence threshold
FCM_MAXITER = 1000

LITHO_ORDER = [
    "Sandstone", "Sandstone/Shale", "Shale", "Marl",
    "Dolomite", "Limestone", "Chalk", "Halite",
    "Anhydrite", "Tuff", "Coal", "Basement",
]

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


# ── Step 2.1: Cluster selection via PC and PE curves ─────────────────────────

def select_optimal_c(X_train_z: np.ndarray, out_dir: str,
                     subsample: int = 80_000) -> int:
    """
    Run FCM for C = 3..12 on a subsample of training data.
    Choose optimal C at the knee of the PC/PE curves.

    Partition Coefficient (PC):  higher = crisper partition (good)
    Fuzzy Partition Entropy (PE): lower  = less uncertainty    (good)
    Returns the C value at the first joint elbow point.
    """
    rng = np.random.default_rng(42)
    if X_train_z.shape[0] > subsample:
        idx = rng.choice(X_train_z.shape[0], subsample, replace=False)
        X_sub = X_train_z[idx]
    else:
        X_sub = X_train_z

    X_input = X_sub.T    # skfuzzy expects (n_features, n_samples)

    pc_vals, pe_vals = [], []
    print(f"  Scanning C = {C_RANGE.start}–{C_RANGE.stop - 1} "
          f"(subsample {X_sub.shape[0]:,} rows) …")

    for c in C_RANGE:
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X_input, c=c, m=M_FUZZ, error=FCM_ERROR, maxiter=FCM_MAXITER,
            init=None,
        )
        pc = fpc  # skfuzzy returns FPC directly
        # Compute Fuzzy Partition Entropy manually
        pe = -float(np.mean(np.sum(u * np.log(u + 1e-10), axis=0)))
        pc_vals.append(pc)
        pe_vals.append(pe)
        print(f"    C={c:2d}   PC={pc:.4f}   PE={pe:.4f}")

    # ── Find elbow: largest second-difference in PC curve ─────────────────
    pc_arr   = np.array(pc_vals)
    diff2    = np.diff(np.diff(pc_arr))
    c_list   = list(C_RANGE)
    # elbow at index with largest negative second-derivative (PC flattens)
    elbow_idx = int(np.argmax(np.abs(diff2))) + 1  # offset for double diff
    C_optimal = c_list[elbow_idx]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("FCM Cluster Selection — Partition Coefficient & Entropy",
                 fontsize=12, fontweight="bold")

    ax1.plot(c_list, pc_vals, "o-", color="#1e90ff", lw=2, ms=7)
    ax1.axvline(C_optimal, color="crimson", ls="--", lw=1.5,
                label=f"Optimal C = {C_optimal}")
    ax1.set_xlabel("Number of clusters C")
    ax1.set_ylabel("Partition Coefficient (PC)")
    ax1.set_title("Partition Coefficient (↑ better)")
    ax1.legend()
    ax1.grid(alpha=0.35)

    ax2.plot(c_list, pe_vals, "o-", color="#ff7f0e", lw=2, ms=7)
    ax2.axvline(C_optimal, color="crimson", ls="--", lw=1.5,
                label=f"Optimal C = {C_optimal}")
    ax2.set_xlabel("Number of clusters C")
    ax2.set_ylabel("Fuzzy Partition Entropy (PE)")
    ax2.set_title("Fuzzy Partition Entropy (↓ better)")
    ax2.legend()
    ax2.grid(alpha=0.35)

    plt.tight_layout()
    path = os.path.join(out_dir, "plots", "fcm_cluster_selection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
    print(f"  → Optimal C = {C_optimal}  (PC elbow method)")

    return C_optimal, pc_vals, pe_vals


# ── Step 2.2: Train final FCM model ──────────────────────────────────────────

def train_fcm(X_train_z: np.ndarray, C_optimal: int):
    """
    Train FCM on the full training set with the chosen C.
    Returns cluster centres (cntr) and training membership matrix (u_train).
    """
    X_input = X_train_z.T    # (n_features, n_samples)
    print(f"  Training FCM: C={C_optimal}, m={M_FUZZ}, "
          f"N={X_train_z.shape[0]:,} samples …")

    cntr, u_train, _, _, jm, _, fpc = fuzz.cluster.cmeans(
        X_input, c=C_optimal, m=M_FUZZ, error=FCM_ERROR,
        maxiter=FCM_MAXITER, init=None,
    )
    print(f"  Training converged. Final FPC = {fpc:.4f}  "
          f"(1.0 = crisp, 1/C = fully fuzzy)")
    return cntr, u_train, fpc


# ── Step 2.3: Predict test memberships ────────────────────────────────────────

def predict_fcm(X_test_z: np.ndarray, cntr: np.ndarray):
    """
    Apply the trained cluster centres to the test set.
    Returns u_test (shape: C × N_test).
    """
    X_input = X_test_z.T
    print(f"  Predicting test memberships: N={X_test_z.shape[0]:,} samples …")
    u_test, _, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X_input, cntr, m=M_FUZZ, error=FCM_ERROR, maxiter=FCM_MAXITER,
    )
    return u_test   # shape (C, N_test)


# ── Step 2.4: Hard labels ─────────────────────────────────────────────────────

def hard_labels_from_memberships(u_test: np.ndarray) -> np.ndarray:
    """Argmax across clusters for each sample → hard cluster assignment."""
    return np.argmax(u_test, axis=0)   # shape (N_test,)


# ── Step 2.5: Map clusters to geological labels ───────────────────────────────

def map_clusters_to_geology(hard_labels: np.ndarray,
                             df_test: pd.DataFrame,
                             C_optimal: int) -> dict:
    """
    For each cluster id, find the most common LITHOLOGY label in that cluster.
    Returns cluster_geo_map: {cluster_id (int): geology_name (str)}.
    """
    cluster_geo_map = {}
    for cid in range(C_optimal):
        mask = hard_labels == cid
        n_in_cluster = mask.sum()
        if n_in_cluster == 0:
            cluster_geo_map[cid] = "Unknown"
            continue
        dominant = df_test.loc[mask, "LITHOLOGY"].mode()
        cluster_geo_map[cid] = dominant.iloc[0] if len(dominant) else "Unknown"

    print("  Cluster → Geology mapping:")
    for cid, geo in cluster_geo_map.items():
        n = int((hard_labels == cid).sum())
        pct = 100.0 * n / len(hard_labels)
        print(f"    Cluster {cid:2d} → {geo:<20s}  ({n:>7,} samples, {pct:.1f}%)")

    return cluster_geo_map


# ── Cluster purity ────────────────────────────────────────────────────────────

def cluster_purity(hard_labels: np.ndarray,
                   df_test: pd.DataFrame,
                   C_optimal: int) -> float:
    """
    Purity = fraction of samples in each cluster that match its dominant label,
    weighted by cluster size.
    """
    total = len(hard_labels)
    correct = 0
    for cid in range(C_optimal):
        mask = hard_labels == cid
        if mask.sum() == 0:
            continue
        dominant_count = df_test.loc[mask, "LITHOLOGY"].value_counts().iloc[0]
        correct += int(dominant_count)
    return correct / total


# ── Output: Confusion matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(hard_labels: np.ndarray,
                           cluster_geo_map: dict,
                           df_test: pd.DataFrame,
                           C_optimal: int,
                           out_dir: str):
    """
    Plot confusion matrix: FCM hard cluster labels (mapped to geology)
    vs. FORCE 2020 true lithology labels.
    """
    pred_geo = np.array([cluster_geo_map[c] for c in hard_labels])
    true_geo = df_test["LITHOLOGY"].values

    present_classes = sorted(set(true_geo) | set(pred_geo))
    cm = confusion_matrix(true_geo, pred_geo, labels=present_classes)

    # Normalise by true class size
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True),
                           0.0)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall (row-normalised)")

    ax.set_xticks(range(len(present_classes)))
    ax.set_yticks(range(len(present_classes)))
    ax.set_xticklabels(present_classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(present_classes, fontsize=9)
    ax.set_xlabel("FCM Predicted Label", fontsize=11)
    ax.set_ylabel("True Lithology (FORCE 2020)", fontsize=11)
    ax.set_title(
        f"FCM Confusion Matrix (C={C_optimal}) — Row-normalised recall\n"
        f"Hold-out: 10 wells",
        fontsize=12, fontweight="bold"
    )

    for i in range(len(present_classes)):
        for j in range(len(present_classes)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.5 else "black")

    plt.tight_layout()
    path = os.path.join(out_dir, "plots", "fcm_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Output: FCM metrics JSON ──────────────────────────────────────────────────

def save_fcm_metrics(C_optimal, fpc, cluster_geo_map, purity,
                     pc_vals, pe_vals, out_dir):
    metrics = {
        "optimal_C":           C_optimal,
        "fpc_final":           round(float(fpc), 4),
        "overall_cluster_purity": round(float(purity), 4),
        "cluster_geology_map": {
            str(k): v for k, v in cluster_geo_map.items()
        },
        "pc_by_c": {str(c): round(float(v), 4)
                    for c, v in zip(C_RANGE, pc_vals)},
        "pe_by_c": {str(c): round(float(v), 4)
                    for c, v in zip(C_RANGE, pe_vals)},
        "fcm_params": {
            "m_fuzziness": M_FUZZ,
            "error": FCM_ERROR,
            "maxiter": FCM_MAXITER,
        },
    }
    path = os.path.join(out_dir, "metrics", "fcm_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {path}")
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def run(out_dir: str, shared: dict = None) -> dict:
    print("\n" + "─" * 60)
    print("  MODULE 2 — FCM Clustering")
    print("─" * 60)

    # Load preprocessed data
    metrics_dir = os.path.join(out_dir, "metrics")

    if shared and "X_train_z" in shared:
        X_train_z = shared["X_train_z"]
        X_test_z  = shared["X_test_z"]
        df_test   = shared["df_test"]
    else:
        with open(os.path.join(metrics_dir, "X_train_z.pkl"), "rb") as f:
            X_train_z = pickle.load(f)
        with open(os.path.join(metrics_dir, "X_test_z.pkl"), "rb") as f:
            X_test_z = pickle.load(f)
        with open(os.path.join(metrics_dir, "df_test.pkl"), "rb") as f:
            df_test = pickle.load(f)

    df_test = df_test.reset_index(drop=True)

    # 2.1 Find optimal C
    C_optimal, pc_vals, pe_vals = select_optimal_c(X_train_z, out_dir)

    # 2.2 Train
    cntr, u_train, fpc = train_fcm(X_train_z, C_optimal)

    # 2.3 Predict
    u_test = predict_fcm(X_test_z, cntr)

    # 2.4 Hard labels
    hard_labels = hard_labels_from_memberships(u_test)

    # 2.5 Map clusters to geology
    cluster_geo_map = map_clusters_to_geology(hard_labels, df_test, C_optimal)

    # Purity
    purity = cluster_purity(hard_labels, df_test, C_optimal)
    print(f"  Overall cluster purity: {purity:.4f}")

    # Confusion matrix
    plot_confusion_matrix(hard_labels, cluster_geo_map, df_test,
                          C_optimal, out_dir)

    # Save metrics
    fcm_metrics = save_fcm_metrics(C_optimal, fpc, cluster_geo_map,
                                   purity, pc_vals, pe_vals, out_dir)

    # Persist model artifacts
    np.save(os.path.join(metrics_dir, "cntr.npy"), cntr)
    np.save(os.path.join(metrics_dir, "u_test.npy"), u_test)
    np.save(os.path.join(metrics_dir, "hard_labels.npy"), hard_labels)
    with open(os.path.join(metrics_dir, "cluster_geo_map.pkl"), "wb") as f:
        pickle.dump(cluster_geo_map, f)

    print("  MODULE 2 complete.\n")
    return {
        "C_optimal":       C_optimal,
        "fpc":             fpc,
        "cntr":            cntr,
        "u_test":          u_test,
        "hard_labels":     hard_labels,
        "cluster_geo_map": cluster_geo_map,
        "purity":          purity,
        "fcm_metrics":     fcm_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCM Lithofacies — Part 2: FCM Clustering")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()
    run(args.out)
