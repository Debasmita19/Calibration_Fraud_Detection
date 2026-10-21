"""RQ1: synthetic shift grid on IEEE-CIS — temporal train/cal, evaluate perturbed test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.models.baselines import get_model
from src.calibration.post_hoc import CALIBRATORS, get_calibrator
from src.evaluation.calibration_metrics import compute_all_calibration_metrics
from src.utils.helpers import load_config, set_seed
from src.data.loader import get_dataset
from src.shifts.shift_generator import create_shift_scenarios

config = load_config("configs/config.yaml")
SEEDS = [42, 43, 44]

X, y, splits, meta = get_dataset("baf", return_meta=True)
X_train = X.iloc[splits["train"]].values
y_train = y[splits["train"]]
X_val = X.iloc[splits["val"]].values
y_val = y[splits["val"]]
X_cal = X.iloc[splits["calibration"]].values
y_cal = y[splits["calibration"]]
X_test_df = X.iloc[splits["test"]]  # keep as DataFrame
y_test = y[splits["test"]]

rows = []
for seed in SEEDS:
    set_seed(seed)
    mc = dict(config["models"]["lightgbm"])
    mc["random_state"] = seed
    model = get_model("lightgbm", mc)
    model.fit(X_train, y_train, X_val, y_val)

    fitted = {}
    probs_cal = model.predict_proba(X_cal)
    for name in CALIBRATORS:
        c = get_calibrator(name)
        c.fit(probs_cal, y_cal)
        fitted[name] = c

    scenarios = create_shift_scenarios(
        X_test_df, y_test, config["shifts"]["synthetic"]
    )

    for scenario_name, (X_s, y_s) in scenarios.items():
        X_s_vals = X_s.values if hasattr(X_s, "values") else X_s
        probs = model.predict_proba(X_s_vals)

        m = compute_all_calibration_metrics(probs, y_s)
        rows.append({
            "seed": seed,
            "scenario": scenario_name,
            "fraud_rate": y_s.mean(),
            "method": "uncalibrated",
            **m,
        })
        for name, c in fitted.items():
            m2 = compute_all_calibration_metrics(c.calibrate(probs), y_s)
            rows.append({
                "seed": seed,
                "scenario": scenario_name,
                "fraud_rate": y_s.mean(),
                "method": name,
                **m2,
            })

out = pd.DataFrame(rows)
out.to_csv("results/tables/synthetic_shift_baf_lgbm.csv", index=False)
print(out.groupby(["scenario", "method"])["ece"].agg(["mean", "std"]).round(5))