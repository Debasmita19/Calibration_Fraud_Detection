"""Standard vs Mondrian conformal on BAF LightGBM, by future month."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils.prediction_store import load_predictions
from src.calibration.conformal import SplitConformalClassifier

ALPHA = 0.10
SEEDS = [42, 43, 44]
MONTHS = [5.0, 6.0, 7.0]


class MondrianConformal:
    """Separate threshold for each class, so fraud is not drowned by safe cases."""

    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.t0 = None
        self.t1 = None

    def fit(self, probs, y):
        def q(scores):
            n = len(scores)
            level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            return np.quantile(scores, level)

        self.t0 = q(probs[y == 0])          # score if the truth is "safe"
        self.t1 = q((1 - probs)[y == 1])    # score if the truth is "fraud"
        return self

    def evaluate(self, probs, y):
        sets = np.column_stack([
            probs <= self.t0,
            (1 - probs) <= self.t1,
        ])
        covered = sets[np.arange(len(y)), y.astype(int)]
        sizes = sets.sum(axis=1)
        return {
            "coverage_all": covered.mean(),
            "coverage_fraud": covered[y == 1].mean() if (y == 1).any() else np.nan,
            "coverage_safe": covered[y == 0].mean() if (y == 0).any() else np.nan,
            "frac_both": (sizes == 2).mean(),
            "frac_empty": (sizes == 0).mean(),
        }


rows = []
for seed in SEEDS:
    df = load_predictions("baf", "lightgbm", seed)
    cal = df[df["split"] == "calibration"]
    standard = SplitConformalClassifier(alpha=ALPHA).fit(
        cal["y_prob"].values, cal["y_true"].values
    )
    mondrian = MondrianConformal(alpha=ALPHA).fit(
        cal["y_prob"].values, cal["y_true"].values
    )

    for month in MONTHS:
        chunk = df[df["time_value"] == month]
        p, y = chunk["y_prob"].values, chunk["y_true"].values

        s = standard.evaluate_coverage(p, y)
        rows.append({
            "seed": seed, "month": month, "method": "standard",
            "coverage_all": s["empirical_coverage"],
            "coverage_fraud": s["empirical_coverage"],  # replaced below
            "frac_both": s["frac_both"],
            "frac_empty": s["frac_empty"],
        })
        # class-wise coverage for the standard method too
        sets = standard.predict_sets(p)
        covered = sets[np.arange(len(y)), y.astype(int)]
        rows[-1]["coverage_fraud"] = covered[y == 1].mean()
        rows[-1]["coverage_safe"] = covered[y == 0].mean()

        m = mondrian.evaluate(p, y)
        rows.append({"seed": seed, "month": month, "method": "mondrian", **m})

out = pd.DataFrame(rows)
out.to_csv("results/tables/conformal_baf_lgbm.csv", index=False)
summary = out.groupby(["month", "method"])[
    ["coverage_all", "coverage_fraud", "coverage_safe", "frac_both", "frac_empty"]
].mean().round(4)
print(summary)