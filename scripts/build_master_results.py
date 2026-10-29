"""Combine every Phase 1-2 result table into one long-format master table."""
from pathlib import Path

import pandas as pd

TABLES = Path("results/tables")

# file -> (experiment, dataset, model, column that names the shift)
SOURCES = {
    "baselines_all_models.csv":          ("baseline",        None,       None,       None),
    "temporal_degradation_baf.csv":      ("temporal",        "baf",      "xgboost",  "month"),
    "temporal_degradation_baf_lgbm.csv": ("temporal",        "baf",      "lightgbm", "month"),
    "temporal_degradation_ieee.csv":     ("temporal",        "ieee_cis", "xgboost",  "window"),
    "crossvariant_baf.csv":              ("baf_variant",     "baf",      "xgboost",  "variant"),
    "subpop_shift_ieee.csv":             ("product_shift",   "ieee_cis", "xgboost",  "domain"),
    "subpop_shift_ieee_lgbm.csv":        ("product_shift",   "ieee_cis", "lightgbm", "domain"),
    "subpop_shift_ieee_mlp.csv":         ("product_shift",   "ieee_cis", "mlp",      "domain"),
    "synthetic_shift_ieee.csv":          ("synthetic_shift", "ieee_cis", "xgboost",  "scenario"),
    "synthetic_shift_ieee_lgbm.csv":     ("synthetic_shift", "ieee_cis", "lightgbm", "scenario"),
    "synthetic_shift_ieee_mlp.csv":      ("synthetic_shift", "ieee_cis", "mlp",      "scenario"),
    "synthetic_shift_baf_lgbm.csv":      ("synthetic_shift", "baf",      "lightgbm", "scenario"),
    "ensemble_baf_lgbm.csv":             ("ensemble",        "baf",      "lightgbm", "month"),
    "conformal_baf_lgbm.csv":            ("conformal",       "baf",      "lightgbm", "month"),
}

ID_COLS = ["experiment", "dataset", "model", "scenario", "seed", "method"]

frames = []
for fname, (experiment, dataset, model, shift_col) in SOURCES.items():
    df = pd.read_csv(TABLES / fname)
    df["experiment"] = experiment
    if dataset is not None:
        df["dataset"] = dataset
        df["model"] = model
    if shift_col is None:
        df["scenario"] = "test"
    else:
        df["scenario"] = df[shift_col].astype(str).str.replace(r"\.0$", "", regex=True)
        if shift_col != "scenario":
            df = df.drop(columns=shift_col)
    if "seed" not in df.columns:
        df["seed"] = pd.NA
    long = df.melt(id_vars=ID_COLS, var_name="metric", value_name="value")
    frames.append(long.dropna(subset=["value"]))

master = pd.concat(frames, ignore_index=True)
master.to_csv(TABLES / "master_results.csv", index=False)
print(master.groupby(["experiment", "dataset", "model"]).size().to_string())
print(f"\n{len(master)} rows written to {TABLES / 'master_results.csv'}")