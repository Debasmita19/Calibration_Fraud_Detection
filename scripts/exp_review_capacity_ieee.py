"""Phase 3 (RQ3): decisions when analysts can review only a fixed share of transactions (IEEE-CIS).

A missed fraud costs its transaction amount. Policies learn from the validation split
and are scored on the test split.
  best_cutpoints  cut-points on raw score x amount, tuned on test labels under the cap
                  (best line-drawing found; not deployable)
  raw_amt_tuned   the same cut-points, tuned on validation labels under the cap
  beta_bayes      cheapest action per transaction from calibrated probability and amount;
                  if too many cases want review, keep the ones where review saves the most
Over-capacity cases in the cut-point policies drop to step-up, the next tier down.
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

COSTS = load_config("configs/config.yaml")["evaluation"]["costs"]
CAPS = {"none": None, "2%": 0.02, "1%": 0.01, "0.5%": 0.005}
RUNS = {"xgboost": [42, 43, 44, 45, 46], "lightgbm": [42, 43, 44], "mlp": [42, 43, 44]}
REVIEW = 2
CANDIDATES = np.unique(np.concatenate([
    np.linspace(0, 1, 120), np.geomspace(1e-4, 1, 120), 1 - np.geomspace(1e-4, 1, 120),
]))

AMOUNTS = (
    load_ieee_cis()[["TransactionID", "TransactionAmt"]]
    .set_index("TransactionID")["TransactionAmt"]
)


def action_costs(y, amount):
    """(n, 4) cost of approve, step-up, review, block. Labels give realized cost;
    probabilities in place of labels give expected cost."""
    y = np.asarray(y, dtype=float)
    loss = amount * COSTS["fraud_loss_multiplier"] * y
    return np.column_stack([
        loss,
        COSTS["step_up_auth_cost"] + (1 - COSTS["step_up_catch_rate"]) * loss,
        COSTS["manual_review_cost"] + (1 - COSTS["review_catch_rate"]) * loss,
        COSTS["false_positive_cost"] * (1 - y),
    ])


def tune_thresholds(scores, y, amount, cap):
    """Lowest-cost three cut-points whose review band holds at most cap * n rows."""
    order = np.argsort(scores)
    s, n = scores[order], len(scores)
    P = np.vstack([np.zeros(4), np.cumsum(action_costs(y[order], amount[order]), axis=0)])
    idx = np.unique((CANDIDATES * n).astype(int).clip(0, n))
    max_review = n if cap is None else int(cap * n)
    best_cost, best_cut = np.inf, None
    for i in idx:
        js = idx[idx >= i]
        for j in js:
            ks = js[(js >= j) & (js - j <= max_review)]
            rest = (P[ks, 2] - P[j, 2]) + (P[n, 3] - P[ks, 3])
            total = P[i, 0] + (P[j, 1] - P[i, 1]) + rest.min()
            if total < best_cost:
                best_cost, best_cut = total, (i, j, ks[rest.argmin()])
    return np.array([s[m] if m < n else np.inf for m in best_cut])


def cutpoint_actions(cuts, scores, cap):
    actions = np.searchsorted(cuts, scores, side="right")
    if cap is not None:
        review = np.where(actions == REVIEW)[0]
        max_review = int(cap * len(scores))
        if len(review) > max_review:
            overflow = review[np.argsort(-scores[review])[max_review:]]
            actions[overflow] = REVIEW - 1
    return actions


def bayes_actions(p, amount, cap):
    expected = action_costs(p, amount)
    actions = expected.argmin(axis=1)
    if cap is not None:
        others = expected[:, [0, 1, 3]]
        fallback = np.array([0, 1, 3])[others.argmin(axis=1)]
        saving = others.min(axis=1) - expected[:, REVIEW]
        review = np.where(actions == REVIEW)[0]
        max_review = int(cap * len(p))
        if len(review) > max_review:
            drop = review[np.argsort(-saving[review])[max_review:]]
            actions[drop] = fallback[drop]
    return actions


def summarize(actions, y, amount):
    reviewed = actions == REVIEW
    cost = action_costs(y, amount)[np.arange(len(y)), actions].sum()
    return {
        "cost_per_1k": 1000 * cost / len(y),
        "pct_reviewed": 100 * reviewed.mean(),
        "review_fraud_rate": y[reviewed].mean() if reviewed.any() else np.nan,
    }


rows = []
for model, seeds in RUNS.items():
    for seed in seeds:
        df = load_predictions("ieee_cis", model, seed)
        df["amount"] = df["sample_id"].map(AMOUNTS)
        lab, test = df[df["split"] == "val"], df[df["split"] == "test"]
        pl, yl, al = lab["y_prob"].values, lab["y_true"].values, lab["amount"].values
        raw, y, a = test["y_prob"].values, test["y_true"].values, test["amount"].values
        p = get_calibrator("beta_calibration").fit(pl, yl).calibrate(raw)

        for cap_name, cap in CAPS.items():
            policies = {
                "best_cutpoints": cutpoint_actions(tune_thresholds(raw * a, y, a, cap), raw * a, cap),
                "raw_amt_tuned": cutpoint_actions(tune_thresholds(pl * al, yl, al, cap), raw * a, cap),
                "beta_bayes": bayes_actions(p, a, cap),
            }
            for policy, act in policies.items():
                rows.append({"model": model, "seed": seed, "cap": cap_name,
                             "policy": policy, **summarize(act, y, a)})
        print(f"done: {model} seed {seed}")

out = pd.DataFrame(rows)
out["cap"] = pd.Categorical(out["cap"], categories=list(CAPS), ordered=True)
out.to_csv("results/tables/review_capacity_ieee.csv", index=False)
cols = ["best_cutpoints", "raw_amt_tuned", "beta_bayes"]
for metric in ("cost_per_1k", "review_fraud_rate"):
    print(f"\n{metric}")
    t = out.pivot_table(index=["model", "cap"], columns="policy", values=metric,
                        aggfunc="mean", observed=True)
    print(t[cols].round(3 if metric == "review_fraud_rate" else 1).to_string())