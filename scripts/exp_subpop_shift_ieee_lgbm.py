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

config = load_config("configs/config.yaml")
SEEDS = [42, 43, 44]          # 3 seeds: retraining is expensive
TARGETS = ["C", "R", "H", "S"]

df_raw = load_ieee_cis("data/raw/ieee-fraud-detection")
source_idx = np.where(df_raw["ProductCD"] == "W")[0]
target_idx = {t: np.where(df_raw["ProductCD"] == t)[0] for t in TARGETS}

rows = []
for seed in SEEDS:
    set_seed(seed)
    # Within-source random split: 60% train, 10% val, 15% cal, 15% source-test
    perm = np.random.default_rng(seed).permutation(source_idx)
    n = len(perm)
    tr = perm[:int(0.60 * n)]
    va = perm[int(0.60 * n):int(0.70 * n)]
    ca = perm[int(0.70 * n):int(0.85 * n)]
    te_src = perm[int(0.85 * n):]

    X, y, _, _ = preprocess_ieee_cis(df_raw.copy(), train_idx=tr)
    Xv = X.values

    mc = dict(config["models"]["lightgbm"]); mc["random_state"] = seed
    model = get_model("lightgbm", mc)
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
out.to_csv("results/tables/subpop_shift_ieee_lgbm.csv", index=False)
print(out.groupby(["domain", "method"])["ece"].agg(["mean", "std"]).round(5))