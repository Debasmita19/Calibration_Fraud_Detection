"""Phase 3 (RQ3): can thresholds be set without labels from the month being scored?

Policies, all on frozen predictions:
  raw_tuned   cut-points tuned on an earlier labeled split (raw scores)
  beta_bayes  Beta fitted on that same split, then the cheapest action under the cost model
  raw_bayes   the same cost rule applied to raw scores, treated as probabilities
  oracle      cut-points tuned on the scored data's own labels (best achievable, not deployable)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils.helpers import load_config
from src.utils.prediction_store import load_predictions
from src.calibration.post_hoc import get_calibrator

BASE_COSTS = load_config("configs/config.yaml")["evaluation"]["costs"]
COST_SETTINGS = {
    "base": BASE_COSTS,
    "block_cost_8": dict(BASE_COSTS, false_positive_cost=8.0),
}
TIERS = ["approve", "step_up", "review", "block"]
RUNS = {
    ("baf", "xgboost"): [42, 43, 44, 45, 46],
    ("baf", "lightgbm"): [42, 43, 44],
    ("ieee_cis", "xgboost"): [42, 43, 44, 45, 46],
    ("ieee_cis", "lightgbm"): [42, 43, 44],
    ("ieee_cis", "mlp"): [42, 43, 44],
}


def action_costs(y, c):
    """(n, 4) cost of each action. With labels it is the realized cost;
    with probabilities in place of labels it is the expected cost."""
    y = np.asarray(y, dtype=float)
    loss = c["avg_transaction_amount"] * c["fraud_loss_multiplier"] * y
    return np.column_stack([
        loss,
        c["step_up_auth_cost"] + (1 - c["step_up_catch_rate"]) * loss,
        c["manual_review_cost"] + (1 - c["review_catch_rate"]) * loss,
        c["false_positive_cost"] * (1 - y),
    ])


def tune_thresholds(scores, y, c, n_cand=120):
    """Lowest-cost three cut-points over score quantiles, using prefix sums."""
    order = np.argsort(scores)
    s, n = scores[order], len(scores)
    P = np.vstack([np.zeros(4), np.cumsum(action_costs(y[order], c), axis=0)])
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


def score(actions, y, c):
    cost = action_costs(y, c)[np.arange(len(y)), actions].sum()
    frac = np.bincount(actions, minlength=4) / len(y)
    return {"cost_per_1k": 1000 * cost / len(y),
            **{f"frac_{t}": f for t, f in zip(TIERS, frac)}}


rows = []
for (dataset, model), seeds in RUNS.items():
    for seed in seeds:
        df = load_predictions(dataset, model, seed)
        if dataset == "baf":
            evals = {str(m): df[df["time_value"] == m] for m in (6, 7)}
        else:
            evals = {"test": df[df["split"] == "test"]}

        for label_split in ("calibration", "val"):
            lab = df[df["split"] == label_split]
            p_lab, y_lab = lab["y_prob"].values, lab["y_true"].values
            beta = get_calibrator("beta_calibration").fit(p_lab, y_lab)
            cuts_base = tune_thresholds(p_lab, y_lab, BASE_COSTS)  # tuned once, under base costs

            for scenario, chunk in evals.items():
                raw, y = chunk["y_prob"].values, chunk["y_true"].values
                p = beta.calibrate(raw)
                for cost_name, c in COST_SETTINGS.items():
                    policies = {
                        "raw_tuned": apply_cuts(cuts_base, raw),
                        "beta_bayes": action_costs(p, c).argmin(axis=1),
                        "raw_bayes": action_costs(raw, c).argmin(axis=1),
                        "oracle": apply_cuts(tune_thresholds(raw, y, c), raw),
                    }
                    for policy, act in policies.items():
                        rows.append({
                            "dataset": dataset, "model": model, "seed": seed,
                            "label_split": label_split, "scenario": scenario,
                            "cost_setting": cost_name, "policy": policy,
                            **score(act, y, c),
                        })

out = pd.DataFrame(rows)
out.to_csv("results/tables/policy_deployment.csv", index=False)
table = out.pivot_table(
    index=["dataset", "model", "label_split", "scenario", "cost_setting"],
    columns="policy", values="cost_per_1k", aggfunc="mean",
)
print(table[["oracle", "raw_tuned", "beta_bayes", "raw_bayes"]].round(1).to_string())