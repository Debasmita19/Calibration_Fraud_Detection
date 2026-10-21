"""Ensemble of saved BAF LightGBM seeds: mean probability and disagreement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils.prediction_store import load_predictions
from src.evaluation.calibration_metrics import compute_all_calibration_metrics

SEEDS = [42, 43, 44]
MONTHS = [5.0, 6.0, 7.0]

parts = []
for seed in SEEDS:
    df = load_predictions("baf", "lightgbm", seed)
    df = df[df["time_value"].isin(MONTHS)][
        ["sample_id", "time_value", "y_true", "y_prob"]
    ].rename(columns={"y_prob": f"p{seed}"})
    parts.append(df)

merged = parts[0]
for p in parts[1:]:
    merged = merged.merge(p, on=["sample_id", "time_value", "y_true"])

prob_cols = [f"p{s}" for s in SEEDS]
merged["p_mean"] = merged[prob_cols].mean(axis=1)
merged["disagreement"] = merged[prob_cols].var(axis=1)

rows = []
for month in MONTHS:
    chunk = merged[merged["time_value"] == month]
    y = chunk["y_true"].values
    for seed in SEEDS:
        m = compute_all_calibration_metrics(chunk[f"p{seed}"].values, y)
        rows.append({"month": month, "method": f"seed_{seed}", "disagreement": np.nan, **m})
    m = compute_all_calibration_metrics(chunk["p_mean"].values, y)
    rows.append({
        "month": month,
        "method": "ensemble_mean",
        "disagreement": chunk["disagreement"].mean(),
        **m,
    })

out = pd.DataFrame(rows)
out.to_csv("results/tables/ensemble_baf_lgbm.csv", index=False)
print(out.pivot(index="month", columns="method", values="ece").round(5))
print("\nMean disagreement by month:")
print(out[out["method"] == "ensemble_mean"][["month", "disagreement"]].round(6))