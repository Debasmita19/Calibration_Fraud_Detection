"""RQ1: BAF cross-variant shift — train/calibrate on Base, evaluate Variants I-V (months 6-7)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import load_baf, preprocess_baf, BAF_VARIANTS
from src.data.splits import temporal_split_baf
from src.models.baselines import get_model
from src.calibration.post_hoc import CALIBRATORS, get_calibrator
from src.evaluation.calibration_metrics import compute_all_calibration_metrics
from src.utils.helpers import load_config, set_seed

config = load_config("configs/config.yaml")
SEEDS = [42, 43, 44]

base = load_baf("Base")
splits = temporal_split_baf(base)
X_base, y_base, encoders = preprocess_baf(base, train_idx=splits["train"])
Xb = X_base.values

rows = []
for seed in SEEDS:
    set_seed(seed)
    mc = dict(config["models"]["xgboost"]); mc["random_state"] = seed
    model = get_model("xgboost", mc)
    model.fit(Xb[splits["train"]], y_base[splits["train"]],
              Xb[splits["val"]], y_base[splits["val"]])

    probs_cal = model.predict_proba(Xb[splits["calibration"]])
    fitted = {}
    for name in CALIBRATORS:
        c = get_calibrator(name)
        c.fit(probs_cal, y_base[splits["calibration"]])
        fitted[name] = c

    for variant in BAF_VARIANTS:          # includes "Base" = the control row
        dfv = load_baf(variant)
        keep = dfv["month"].isin([6, 7]).values
        Xv, yv, _ = preprocess_baf(dfv, encoders=encoders)
        Xv = Xv.reindex(columns=X_base.columns, fill_value=0)
        probs = model.predict_proba(Xv.values[keep])
        yk = yv[keep]
        m = compute_all_calibration_metrics(probs, yk)
        rows.append({"seed": seed, "variant": variant, "fraud_rate": yk.mean(),
                     "method": "uncalibrated", **m})
        for name, c in fitted.items():
            m2 = compute_all_calibration_metrics(c.calibrate(probs), yk)
            rows.append({"seed": seed, "variant": variant, "fraud_rate": yk.mean(),
                         "method": name, **m2})

out = pd.DataFrame(rows)
out.to_csv("results/tables/crossvariant_baf.csv", index=False)
print(out.groupby(["variant", "method"])["ece"].agg(["mean", "std"]).round(5))