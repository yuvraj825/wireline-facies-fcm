#!/usr/bin/env python3
"""
================================================================================
pipeline.py — Master Orchestrator
Gradational Lithofacies Classification via Fuzzy C-Means
FORCE 2020 Well-log Dataset × Raniganj Basin Field Calibration
================================================================================

Author  : Kumar Yuvraj (23GG5PE02), IIT Kharagpur

Modules:
  01_data_preparation.py    — Load, clean, normalise FORCE 2020 logs
  02_fcm_clustering.py      — Train FCM, select optimal C, predict memberships
  03_vsh_derivation.py      — Convert memberships → continuous Vsh + NTG
  04_vsh_validation.py      — Cross-validate Vsh against pseudo-GR (Raniganj)
  05_uncertainty_analysis.py — Shannon entropy, bootstrap stability

Usage:
  python src/pipeline.py --data data/train.csv --out outputs/
  python src/pipeline.py --data data/train.csv --out outputs/ --modules 1 2 3
================================================================================
"""

import os
import sys
import json
import time
import argparse
import traceback

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradational Lithofacies Classification via Fuzzy C-Means                  ║
║  FORCE 2020 (1.17M samples) × Raniganj Basin Pseudo-GR Calibration         ║
║                                                                              ║
║  Author : Kumar Yuvraj (23GG5PE02) | IIT Kharagpur                         ║
║  Dataset: FORCE 2020 Machine Learning Competition (Norwegian Shelf)         ║
║  Ground truth calibration: Self-mapped GPS outcrops, Raniganj Basin 2024   ║
║                                                                              ║
║  M1 — Data Preparation  (FORCE 2020 preprocessing + well split)            ║
║  M2 — FCM Clustering    (optimal C selection + membership prediction)       ║
║  M3 — Vsh Derivation    (membership → continuous shale volume curve)        ║
║  M4 — Vsh Validation    (cross-validate vs Raniganj pseudo-GR ground truth) ║
║  M5 — Uncertainty       (Shannon entropy + bootstrap stability)             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _add_src_to_path():
    """Ensure src/ is importable regardless of working directory."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _status(msg: str):
    print(f"\n{'═' * 62}")
    print(f"  {msg}")
    print(f"{'═' * 62}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s // 60)}m {int(s % 60)}s"


# ── Module registry ───────────────────────────────────────────────────────────

def build_module_registry(data_path: str, out_dir: str):
    """
    Returns a list of (module_number, name, run_fn, kwargs) tuples.
    Each run() function accepts (out_dir, shared) and returns a dict that
    is merged into the shared state dict — so modules can access each
    other's outputs without re-loading from disk.
    """
    import importlib

    def _load(mod_file):
        spec = importlib.util.spec_from_file_location(
            mod_file,
            os.path.join(os.path.dirname(__file__), mod_file),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    registry = [
        (1, "Data Preparation",    lambda shared: _load("01_data_preparation.py").run(data_path, out_dir),    {}),
        (2, "FCM Clustering",      lambda shared: _load("02_fcm_clustering.py").run(out_dir, shared),          {}),
        (3, "Vsh Derivation",      lambda shared: _load("03_vsh_derivation.py").run(out_dir, shared),          {}),
        (4, "Vsh Validation",      lambda shared: _load("04_vsh_validation.py").run(out_dir, shared),          {}),
        (5, "Uncertainty Analysis",lambda shared: _load("05_uncertainty_analysis.py").run(out_dir, shared),    {}),
    ]
    return registry


# ── Run summary JSON ──────────────────────────────────────────────────────────

def write_run_summary(shared: dict, out_dir: str, elapsed: str):
    """
    Write a single run_summary.json that consolidates the key metrics
    so any reader gets an instant cross-project benchmark without needing
    to run either the electrofacies or the Raniganj projects.
    """
    fcm_m   = shared.get("fcm_metrics",        {})
    val_r   = shared.get("validation_report",  {})
    unc_r   = shared.get("unc_report",         {})
    bstrap  = shared.get("bootstrap_stats",    {})
    ntg_cmp = val_r.get("ntg_comparison",      {})

    def _get(d, *keys, default="N/A"):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    summary = {
        "pipeline_elapsed":              elapsed,
        "fcm_optimal_C":                 _get(fcm_m, "optimal_C"),
        "fpc":                           _get(fcm_m, "fpc_final"),
        "cluster_purity":                _get(fcm_m, "overall_cluster_purity"),
        "vsh_validation": {
            "r2_fcm_vs_gr":              _get(val_r, "vsh_fcm_vs_gr", "r2"),
            "rmse_fcm_vs_gr":            _get(val_r, "vsh_fcm_vs_gr", "rmse"),
            "per_class_rmse_sandstone":  _get(val_r, "vsh_fcm_vs_gr", "per_class_rmse", "Sandstone"),
            "per_class_rmse_shale":      _get(val_r, "vsh_fcm_vs_gr", "per_class_rmse", "Shale"),
            "per_class_rmse_trans":      _get(val_r, "vsh_fcm_vs_gr", "per_class_rmse", "Sandstone/Shale"),
            "ntg_ramnagar_pseudo_gr":    0.500,
            "ntg_ramnagar_fcm":         _get(ntg_cmp, "ramnagar", "fcm_vsh"),
            "ntg_ramnagar_gr":          _get(ntg_cmp, "ramnagar", "gr_vsh"),
            "ntg_shunuri_pseudo_gr":    0.571,
            "ntg_shunuri_fcm":          _get(ntg_cmp, "shunuri", "fcm_vsh"),
            "ntg_shunuri_gr":           _get(ntg_cmp, "shunuri", "gr_vsh"),
            "ntg_duburdi_pseudo_gr":    0.190,
            "ntg_duburdi_fcm":          _get(ntg_cmp, "duburdi", "fcm_vsh"),
            "ntg_duburdi_gr":           _get(ntg_cmp, "duburdi", "gr_vsh"),
        },
        "uncertainty": {
            "mean_entropy_norm":         _get(unc_r, "mean_entropy_norm"),
            "fraction_transition_zones": _get(unc_r, "fraction_high_entropy"),
            "bootstrap_vsh_std_mean":    _get(bstrap, "mean_std_all"),
            "n_bootstrap_seeds":         _get(unc_r, "bootstrap", "n_seeds"),
        },
        # Hardcoded from electrofacies-classification project — gives
        # any reader an instant benchmark without running the other repo.
        "lgbm_baseline_weighted_f1":    0.7041,
        "lgbm_baseline_accuracy":       0.7089,
        "lgbm_baseline_macro_f1":       0.3044,
    }

    path = os.path.join(out_dir, "run_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Run summary → {path}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _add_src_to_path()
    import importlib.util   # needed for dynamic module loading

    parser = argparse.ArgumentParser(
        description="FCM Lithofacies pipeline — runs all 5 modules in sequence"
    )
    parser.add_argument("--data",    required=True,
                        help="Path to FORCE 2020 train.csv")
    parser.add_argument("--out",     default="outputs",
                        help="Root output directory  [default: outputs]")
    parser.add_argument("--modules", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5],
                        metavar="N",
                        help="Module numbers to run (default: 1 2 3 4 5)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.data):
        sys.exit(f"ERROR: data file not found: {args.data}\n"
                 "Download train.csv from the FORCE 2020 GitHub repo "
                 "and place it in data/  (see data/README.md)")

    # Create output directories
    for sub in ["metrics", "plots", "data", "figures"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    print(BANNER)
    print(f"  Data  : {args.data}")
    print(f"  Output: {args.out}")
    print(f"  Modules to run: {args.modules}")

    t0     = time.time()
    shared = {}   # shared state dict — modules write here, downstream modules read
    registry = build_module_registry(args.data, args.out)

    for (mod_num, mod_name, run_fn, _) in registry:
        if mod_num not in args.modules:
            print(f"\n  ── Skipping M{mod_num}: {mod_name} ──")
            continue

        _status(f"M{mod_num} — {mod_name}")
        t_mod = time.time()

        try:
            result = run_fn(shared)
            if result:
                shared.update(result)
            print(f"  M{mod_num} finished in {_elapsed(t_mod)}")
        except Exception:
            print(f"\n  ✗ M{mod_num} — {mod_name} FAILED:")
            traceback.print_exc()
            sys.exit(1)

    # Write run summary
    if all(m in args.modules for m in [1, 2, 3, 4, 5]):
        summary = write_run_summary(shared, args.out, _elapsed(t0))

        print("\n" + "═" * 62)
        print("  PIPELINE COMPLETE")
        print(f"  Total elapsed : {_elapsed(t0)}")
        if "fcm_optimal_C" in summary:
            print(f"  FCM C         : {summary['fcm_optimal_C']}")
        vsh_v = summary.get("vsh_validation", {})
        if vsh_v.get("r2_fcm_vs_gr") != "N/A":
            print(f"  Vsh R²        : {vsh_v.get('r2_fcm_vs_gr')}")
            print(f"  Vsh RMSE      : {vsh_v.get('rmse_fcm_vs_gr')}")
        unc_v = summary.get("uncertainty", {})
        if unc_v.get("fraction_transition_zones") != "N/A":
            frac = unc_v['fraction_transition_zones']
            print(f"  Transition zones: {float(frac):.1%} of depth intervals")
        print("═" * 62 + "\n")
    else:
        print(f"\n  ✓ Selected modules complete in {_elapsed(t0)}")


if __name__ == "__main__":
    main()
