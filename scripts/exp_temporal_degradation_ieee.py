"""RQ1: calibration metrics per future month, per method, per seed (BAF)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.prediction_store import load_predictions
from src.calibration.post_hoc import CALIBRATORS, get_calibrator
from src.evaluation.calibration_metrics import compute_all_calibration_metrics

SEEDS = [42, 43, 44, 45, 46]
EVAL_MONTHS = [5.0, 6.0, 7.0]   # val month + the two test months

rows = []
for seed in SEEDS:
    df = load_predictions("ieee_cis", "xgboost", seed)
    cal = df[df["split"] == "calibration"]   # month 4 predictions

    fitted = {}
    for name in CALIBRATORS:
        c = get_calibrator(name)
        c.fit(cal["y_prob"].values, cal["y_true"].values)
        fitted[name] = c

    test = df[df["split"] == "test"].copy()
    test["window"] = pd.qcut(test["time_value"], q=5, labels=False) + 1
    for window, chunk in test.groupby("window"):
        m = compute_all_calibration_metrics(chunk["y_prob"].values,
                                            chunk["y_true"].values)
        rows.append({"seed": seed, "window": window,
                     "method": "uncalibrated", **m})
        for name, c in fitted.items():
            cal_probs = c.calibrate(chunk["y_prob"].values)
            mc = compute_all_calibration_metrics(cal_probs, chunk["y_true"].values)
            rows.append({"seed": seed, "window": window, "method": name, **mc})

out = pd.DataFrame(rows)
out.to_csv("results/tables/temporal_degradation_ieee.csv", index=False)
print(out.groupby(["window", "method"])["ece"].agg(["mean", "std"]).round(5))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = out.groupby(["window", "method"])["ece"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
for method, g in summary.groupby("method"):
    ax.errorbar(g["window"], g["mean"], yerr=g["std"],
                marker="o", capsize=3, label=method)
ax.set_yscale("log")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("Month (calibrators fitted on month 4)")
ax.set_ylabel("ECE, log scale (mean ± std over 5 seeds)")
ax.set_title("Calibration degradation over time — BAF, XGBoost")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/temporal_degradation_ieee.png", dpi=150)
print("Figure saved to results/figures/temporal_degradation_ieee.png")