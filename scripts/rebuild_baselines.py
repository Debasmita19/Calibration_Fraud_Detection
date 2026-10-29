"""Rebuild baseline + calibrator tables for every saved run from the prediction store."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.prediction_store import load_predictions
from src.calibration.post_hoc import CALIBRATORS, get_calibrator
from src.evaluation.calibration_metrics import compute_all_calibration_metrics

RUNS = {
    ("ieee_cis", "xgboost"): [42, 43, 44, 45, 46],
    ("ieee_cis", "lightgbm"): [42, 43, 44],
    ("ieee_cis", "mlp"): [42, 43, 44],
    ("baf", "xgboost"): [42, 43, 44, 45, 46],
    ("baf", "lightgbm"): [42, 43, 44],
}

rows = []
for (dataset, model), seeds in RUNS.items():
    for seed in seeds:
        df = load_predictions(dataset, model, seed)
        cal = df[df["split"] == "calibration"]
        test = df[df["split"] == "test"]
        y = test["y_true"].values
        p_raw = test["y_prob"].values

        methods = {"uncalibrated": p_raw}
        for name in CALIBRATORS:
            c = get_calibrator(name)
            c.fit(cal["y_prob"].values, cal["y_true"].values)
            methods[name] = c.calibrate(p_raw)

        for method, p in methods.items():
            rows.append({
                "dataset": dataset, "model": model, "seed": seed, "method": method,
                "auroc": roc_auc_score(y, p),
                "auprc": average_precision_score(y, p),
                **compute_all_calibration_metrics(p, y),
            })

out = pd.DataFrame(rows)
out.to_csv("results/tables/baselines_all_models.csv", index=False)
summary = out.groupby(["dataset", "model", "method"])[["auprc", "ece"]].agg(["mean", "std"])
print(summary.round(4).to_string())