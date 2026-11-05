"""Combine every Phase 1-3 result table into one long-format master table."""
from pathlib import Path

import pandas as pd

TABLES = Path("results/tables")

# file -> (experiment, dataset, model, column that names the shift, condition columns)
# dataset/model are written in only when the file doesn't already have them.
SOURCES = {
    "baselines_all_models.csv":          ("baseline",         None,       None,       None,       []),
    "temporal_degradation_baf.csv":      ("temporal",         "baf",      "xgboost",  "month",    []),
    "temporal_degradation_baf_lgbm.csv": ("temporal",         "baf",      "lightgbm", "month",    []),
    "temporal_degradation_ieee.csv":     ("temporal",         "ieee_cis", "xgboost",  "window",   []),
    "crossvariant_baf.csv":              ("baf_variant",      "baf",      "xgboost",  "variant",  []),
    "subpop_shift_ieee.csv":             ("product_shift",    "ieee_cis", "xgboost",  "domain",   []),
    "subpop_shift_ieee_lgbm.csv":        ("product_shift",    "ieee_cis", "lightgbm", "domain",   []),
    "subpop_shift_ieee_mlp.csv":         ("product_shift",    "ieee_cis", "mlp",      "domain",   []),
    "synthetic_shift_ieee.csv":          ("synthetic_shift",  "ieee_cis", "xgboost",  "scenario", []),
    "synthetic_shift_ieee_lgbm.csv":     ("synthetic_shift",  "ieee_cis", "lightgbm", "scenario", []),
    "synthetic_shift_ieee_mlp.csv":      ("synthetic_shift",  "ieee_cis", "mlp",      "scenario", []),
    "synthetic_shift_baf_lgbm.csv":      ("synthetic_shift",  "baf",      "lightgbm", "scenario", []),
    "ensemble_baf_lgbm.csv":             ("ensemble",         "baf",      "lightgbm", "month",    []),
    "conformal_baf_lgbm.csv":            ("conformal",        "baf",      "lightgbm", "month",    []),
    "policy_deployment.csv":             ("policy_flat_cost", None,       None,       "scenario", ["label_split", "cost_setting"]),
    "policy_amounts_ieee.csv":           ("policy_amounts",   None,       None,       None,       ["label_split", "cost_setting"]),
    "review_capacity_ieee.csv":          ("review_capacity",  "ieee_cis", None,       None,       ["cap"]),
    "cost_sensitivity_ieee.csv":         ("cost_sensitivity", "ieee_cis", None,       None,       ["setting"]),
}

ID_COLS = ["experiment", "dataset", "model", "scenario", "condition", "seed", "method"]

frames = []
for fname, (experiment, dataset, model, shift_col, cond_cols) in SOURCES.items():
    df = pd.read_csv(TABLES / fname)
    df["experiment"] = experiment
    if dataset is not None:
        df["dataset"] = dataset
    if model is not None:
        df["model"] = model
    if "policy" in df.columns:
        df = df.rename(columns={"policy": "method"})
    if shift_col is None:
        df["scenario"] = "test"
    else:
        df["scenario"] = df[shift_col].astype(str).str.replace(r"\.0$", "", regex=True)
        if shift_col != "scenario":
            df = df.drop(columns=shift_col)
    if cond_cols:
        df["condition"] = df[cond_cols].astype(str).agg("|".join, axis=1)
        df = df.drop(columns=cond_cols)
    else:
        df["condition"] = "-"
    if "seed" not in df.columns:
        df["seed"] = pd.NA
    long = df.melt(id_vars=ID_COLS, var_name="metric", value_name="value")
    frames.append(long.dropna(subset=["value"]))

master = pd.concat(frames, ignore_index=True)
master.to_csv(TABLES / "master_results.csv", index=False)
print(master.groupby(["experiment", "dataset", "model"]).size().to_string())
print(f"\n{len(master)} rows written to {TABLES / 'master_results.csv'}")