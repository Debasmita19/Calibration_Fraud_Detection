"""Phase 3 diagnostic: why calibrated probabilities 'lost' the old cost comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils.helpers import load_config
from src.utils.prediction_store import load_predictions
from src.calibration.post_hoc import get_calibrator
from src.policies.decision import AdaptiveThresholdOptimizer
from src.evaluation.business_metrics import compute_business_metrics

costs = load_config("configs/config.yaml")["evaluation"]["costs"]
RUNS = [("ieee_cis", "xgboost"), ("ieee_cis", "lightgbm"), ("ieee_cis", "mlp"),
        ("baf", "xgboost"), ("baf", "lightgbm")]
SEED = 42


def total_cost(p, y, a, s, r):
    t = {"auto_approve_max": a, "step_up_auth_max": s, "manual_review_max": r}
    return compute_business_metrics(p, y, None, t, costs)["total_cost"]


rows = []
for dataset, model in RUNS:
    df = load_predictions(dataset, model, SEED)
    cal, test = df[df["split"] == "calibration"], df[df["split"] == "test"]
    y, raw = test["y_true"].values, test["y_prob"].values
    beta = get_calibrator("beta_calibration").fit(
        cal["y_prob"].values, cal["y_true"].values
    )
    p = beta.calibrate(raw)

    opt = AdaptiveThresholdOptimizer(costs)
    grid_raw = opt.optimize(raw, y)["best_total_cost"]
    grid_beta = opt.optimize(p, y)["best_total_cost"]

    q = np.quantile(raw, [0.80, 0.95, 0.99])   # same rank cutoffs for both
    rows.append({
        "dataset": dataset, "model": model,
        "grid_cost_raw": grid_raw, "grid_cost_beta": grid_beta,
        "grid_change_pct": 100 * (grid_raw - grid_beta) / grid_raw,
        "matched_cost_raw": total_cost(raw, y, *q),
        "matched_cost_beta": total_cost(p, y, *beta.calibrate(q)),
        "beta_share_below_0.01": (p < 0.01).mean(),
        "raw_share_below_0.01": (raw < 0.01).mean(),
    })

out = pd.DataFrame(rows)
out.to_csv("results/tables/policy_grid_diagnostic.csv", index=False)
print(out.round(3).to_string(index=False))