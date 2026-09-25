"""RQ1: subpopulation shift — train/calibrate on ProductCD=W, test on C/R/H/S."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import load_ieee_cis, preprocess_ieee_cis
from src.models.baselines import get_model
from src.calibration.post_hoc import CALIBRATORS, get_calibrator
from src.evaluation.calibration_metrics import compute_all_calibration_metrics
from src.utils.helpers import load_config, set_seed
from src.data.loader import get_dataset
from src.shifts.shift_generator import create_shift_scenarios

config = load_config("configs/config.yaml")
SEEDS = [42, 43, 44]

X, y, splits, meta = get_dataset("ieee_cis", return_meta=True)
X_train = X.iloc[splits["train"]].values
y_train = y[splits["train"]]
X_val   = X.iloc[splits["val"]].values
y_val   = y[splits["val"]]
X_cal   = X.iloc[splits["calibration"]].values
y_cal   = y[splits["calibration"]]
X_test_df = X.iloc[splits["test"]]   # keep as DataFrame
y_test    = y[splits["test"]]


source_idx = np.where(df_raw["ProductCD"] == "W")[0]
target_idx = {t: np.where(df_raw["ProductCD"] == t)[0] for t in TARGETS}

rows = []
for seed in SEEDS:
    set_seed(seed)
    mc = dict(config["models"]["xgboost"])
    mc["random_state"] = seed
    model = get_model("xgboost", mc)
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

    

    mc = dict(config["models"]["xgboost"]); mc["random_state"] = seed
    model = get_model("xgboost", mc)
    model.fit(Xv[tr], y[tr], Xv[va], y[va])

    probs_cal = model.predict_proba(Xv[ca])
    fitted = {}
    for name in CALIBRATORS:
        c = get_calibrator(name)
        c.fit(probs_cal, y[ca])
        fitted[name] = c

    domains = {"W_source": te_src, **{t: idx for t, idx in target_idx.items()}}
    for domain, idx in domains.items():
        probs = model.predict_proba(Xv[idx])
        m = compute_all_calibration_metrics(probs, y[idx])
        rows.append({"seed": seed, "domain": domain, "fraud_rate": y[idx].mean(),
                     "method": "uncalibrated", **m})
        for name, c in fitted.items():
            m2 = compute_all_calibration_metrics(c.calibrate(probs), y[idx])
            rows.append({"seed": seed, "domain": domain, "fraud_rate": y[idx].mean(),
                         "method": name, **m2})

out = pd.DataFrame(rows)
out.to_csv("results/tables/synthetic_shift_ieee.csv", index=False)
print(out.groupby(["scenario", "method"])["ece"].agg(["mean", "std"]).round(5))