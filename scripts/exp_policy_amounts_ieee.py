"""Phase 3 (RQ3): decisions when a missed fraud costs the transaction amount (IEEE-CIS).

Policies, all on frozen predictions:
  raw_tuned       cut-points on the raw score, tuned on an earlier labeled split
  raw_amt_tuned   cut-points on raw score x amount, tuned on that same split
  beta_bayes      Beta fitted on that split, then the cheapest action per transaction
                  from its probability and its amount
  raw_bayes       the same rule applied to raw scores, treated as probabilities
  best_cutpoints  cut-points on raw score x amount tuned on the scored data's own labels
                  (best single-score policy found; not deployable)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils.helpers import load_config
from src.utils.prediction_store import load_predictions
from src.calibration.post_hoc import get_calibrator
from src.data.loader import load_ieee_cis

BASE_COSTS = load_config("configs/config.yaml")["evaluation"]["costs"]
COST_SETTINGS = {
    "base": BASE_COSTS,
    "block_cost_8": dict(BASE_COSTS, false_positive_cost=8.0),
}
TIERS = ["approve", "step_up", "review", "block"]
RUNS = {"xgboost": [42, 43, 44, 45, 46], "lightgbm": [42, 43, 44], "mlp": [42, 43, 44]}

AMOUNTS = (
    load_ieee_cis()[["TransactionID", "TransactionAmt"]]
    .set_index("TransactionID")["TransactionAmt"]
)


def action_costs(y, amount, c):
    """(n, 4) cost of each action. With labels it is the realized cost;
    with probabilities in place of labels it is the expected cost."""
    y = np.asarray(y, dtype=float)
    loss = amount * c["fraud_loss_multiplier"] * y
    return np.column_stack([
        loss,
        c["step_up_auth_cost"] + (1 - c["step_up_catch_rate"]) * loss,
        c["manual_review_cost"] + (1 - c["review_catch_rate"]) * loss,
        c["false_positive_cost"] * (1 - y),
    ])


def tune_thresholds(scores, y, amount, c, n_cand=120):
    """Lowest-cost three cut-points over score quantiles, using prefix sums."""
    order = np.argsort(scores)
    s, n = scores[order], len(scores)
    P = np.vstack([np.zeros(4), np.cumsum(action_costs(y[order], amount[order], c), axis=0)])
    q = np.unique(np.concatenate([np.linspace(0, 1, n_cand), np.geomspace(1e-4, 1, n_cand)]))
    idx = np.unique((q * n).astype(int).clip(0, n))
    best_cost, best_cut = np.inf, None
    for i in idx:
        js = idx[idx >= i]
        for j in js:
            ks = js[js >= j]
            rest = (P[ks, 2] - P[j, 2]) + (P[n, 3] - P[ks, 3])
            total = P[i, 0] + (P[j, 1] - P[i, 1]) + rest.min()
            if total < best_cost:
                best_cost, best_cut = total, (i, j, ks[rest.argmin()])
    return np.array([s[m] if m < n else np.inf for m in best_cut])


def apply_cuts(cuts, scores):
    return np.searchsorted(cuts, scores, side="right")


def score(actions, y, amount, c):
    cost = action_costs(y, amount, c)[np.arange(len(y)), actions].sum()
    frac = np.bincount(actions, minlength=4) / len(y)
    return {"cost_per_1k": 1000 * cost / len(y),
            **{f"frac_{t}": f for t, f in zip(TIERS, frac)}}


rows = []
for model, seeds in RUNS.items():
    for seed in seeds:
        df = load_predictions("ieee_cis", model, seed)
        df["amount"] = df["sample_id"].map(AMOUNTS)
        test = df[df["split"] == "test"]
        raw, y, a = test["y_prob"].values, test["y_true"].values, test["amount"].values

        for label_split in ("calibration", "val"):
            lab = df[df["split"] == label_split]
            pl, yl, al = lab["y_prob"].values, lab["y_true"].values, lab["amount"].values
            p = get_calibrator("beta_calibration").fit(pl, yl).calibrate(raw)
            cuts_raw = tune_thresholds(pl, yl, al, BASE_COSTS)       # tuned once, base costs
            cuts_amt = tune_thresholds(pl * al, yl, al, BASE_COSTS)

            for cost_name, c in COST_SETTINGS.items():
                policies = {
                    "raw_tuned": apply_cuts(cuts_raw, raw),
                    "raw_amt_tuned": apply_cuts(cuts_amt, raw * a),
                    "beta_bayes": action_costs(p, a, c).argmin(axis=1),
                    "raw_bayes": action_costs(raw, a, c).argmin(axis=1),
                    "best_cutpoints": apply_cuts(tune_thresholds(raw * a, y, a, c), raw * a),
                }
                for policy, act in policies.items():
                    rows.append({
                        "dataset": "ieee_cis", "model": model, "seed": seed,
                        "label_split": label_split, "cost_setting": cost_name,
                        "policy": policy, **score(act, y, a, c),
                    })

out = pd.DataFrame(rows)
out.to_csv("results/tables/policy_amounts_ieee.csv", index=False)
table = out.pivot_table(
    index=["model", "label_split", "cost_setting"],
    columns="policy", values="cost_per_1k", aggfunc="mean",
)
cols = ["best_cutpoints", "raw_tuned", "raw_amt_tuned", "beta_bayes", "raw_bayes"]
print(table[cols].round(1).to_string())