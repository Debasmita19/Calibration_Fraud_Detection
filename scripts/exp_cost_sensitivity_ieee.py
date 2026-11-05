"""Phase 3 (RQ3): do the decision results survive other values of the assumed costs?

A missed fraud costs its transaction amount (IEEE-CIS). Each assumed cost is set to a low
and a high value, one at a time, with the others at their base values. Policies learn from
the validation split and are scored on the test split; no review cap.
Also draws the Phase 3 figure from this run and results/tables/review_capacity_ieee.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.helpers import load_config
from src.utils.prediction_store import load_predictions
from src.calibration.post_hoc import get_calibrator
from src.data.loader import load_ieee_cis

BASE = load_config("configs/config.yaml")["evaluation"]["costs"]
VARY = {
    "step_up_catch_rate": (0.5, 0.9),
    "review_catch_rate": (0.85, 0.99),
    "step_up_auth_cost": (0.25, 1.0),
    "manual_review_cost": (2.0, 10.0),
    "false_positive_cost": (5.0, 50.0),
}
SETTINGS = {"base": BASE}
for param, values in VARY.items():
    for v in values:
        SETTINGS[f"{param}={v}"] = dict(BASE, **{param: v})

RUNS = {"xgboost": [42, 43, 44, 45, 46], "lightgbm": [42, 43, 44]}
CANDIDATES = np.unique(np.concatenate([
    np.linspace(0, 1, 120), np.geomspace(1e-4, 1, 120), 1 - np.geomspace(1e-4, 1, 120),
]))

AMOUNTS = (
    load_ieee_cis()[["TransactionID", "TransactionAmt"]]
    .set_index("TransactionID")["TransactionAmt"]
)


def action_costs(y, amount, c):
    """(n, 4) cost of approve, step-up, review, block. Labels give realized cost;
    probabilities in place of labels give expected cost."""
    y = np.asarray(y, dtype=float)
    loss = amount * c["fraud_loss_multiplier"] * y
    return np.column_stack([
        loss,
        c["step_up_auth_cost"] + (1 - c["step_up_catch_rate"]) * loss,
        c["manual_review_cost"] + (1 - c["review_catch_rate"]) * loss,
        c["false_positive_cost"] * (1 - y),
    ])


def tune_thresholds(scores, y, amount, c):
    """Lowest-cost three cut-points over score quantiles, using prefix sums."""
    order = np.argsort(scores)
    s, n = scores[order], len(scores)
    P = np.vstack([np.zeros(4), np.cumsum(action_costs(y[order], amount[order], c), axis=0)])
    idx = np.unique((CANDIDATES * n).astype(int).clip(0, n))
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


def cost_per_1k(actions, y, amount, c):
    return 1000 * action_costs(y, amount, c)[np.arange(len(y)), actions].sum() / len(y)


rows = []
for model, seeds in RUNS.items():
    for seed in seeds:
        df = load_predictions("ieee_cis", model, seed)
        df["amount"] = df["sample_id"].map(AMOUNTS)
        lab, test = df[df["split"] == "val"], df[df["split"] == "test"]
        pl, yl, al = lab["y_prob"].values, lab["y_true"].values, lab["amount"].values
        raw, y, a = test["y_prob"].values, test["y_true"].values, test["amount"].values
        p = get_calibrator("beta_calibration").fit(pl, yl).calibrate(raw)

        for setting, c in SETTINGS.items():
            policies = {
                "best_cutpoints": np.searchsorted(tune_thresholds(raw * a, y, a, c), raw * a, side="right"),
                "raw_amt_tuned": np.searchsorted(tune_thresholds(pl * al, yl, al, c), raw * a, side="right"),
                "beta_bayes": action_costs(p, a, c).argmin(axis=1),
            }
            for policy, act in policies.items():
                rows.append({"model": model, "seed": seed, "setting": setting,
                             "policy": policy, "cost_per_1k": cost_per_1k(act, y, a, c)})
        print(f"done: {model} seed {seed}")

out = pd.DataFrame(rows)
out.to_csv("results/tables/cost_sensitivity_ieee.csv", index=False)

wide = out.pivot_table(index=["model", "seed", "setting"], columns="policy",
                       values="cost_per_1k").reset_index()
wide["saving_vs_best_pct"] = 100 * (1 - wide["beta_bayes"] / wide["best_cutpoints"])
wide["saving_vs_tuned_pct"] = 100 * (1 - wide["beta_bayes"] / wide["raw_amt_tuned"])
wide["wins_vs_best"] = wide["beta_bayes"] < wide["best_cutpoints"]

summary = wide.groupby(["setting", "model"]).agg(
    best_cutpoints=("best_cutpoints", "mean"),
    raw_amt_tuned=("raw_amt_tuned", "mean"),
    beta_bayes=("beta_bayes", "mean"),
    saving_vs_best_pct=("saving_vs_best_pct", "mean"),
    saving_vs_tuned_pct=("saving_vs_tuned_pct", "mean"),
    wins_vs_best=("wins_vs_best", "sum"),
    n_seeds=("wins_vs_best", "size"),
)
summary = summary.reindex(pd.MultiIndex.from_product([list(SETTINGS), list(RUNS)]))
print(summary.round(1).to_string())

# Phase 3 figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

cap = pd.read_csv("results/tables/review_capacity_ieee.csv")
order = ["none", "2%", "1%", "0.5%"]
labels = {"best_cutpoints": "best line-drawing (sees test labels)",
          "raw_amt_tuned": "line-drawing tuned on past labels",
          "beta_bayes": "calibrated + expected-cost rule"}
for model, style in {"xgboost": "-", "lightgbm": "--"}.items():
    for policy, label in labels.items():
        m = (cap[(cap["model"] == model) & (cap["policy"] == policy)]
             .groupby("cap")["cost_per_1k"].mean().reindex(order))
        ax1.plot(order, m.values, style, marker="o", label=f"{model}: {label}")
ax1.set_xlabel("Analyst capacity (share of transactions reviewable)")
ax1.set_ylabel("Loss per 1,000 transactions ($)")
ax1.set_title("Cost as analyst capacity shrinks")
ax1.legend(fontsize=7)

stats = wide.groupby(["setting", "model"])["saving_vs_best_pct"].agg(["mean", "std"])
names = list(SETTINGS)
ypos = np.arange(len(names))
for k, model in enumerate(RUNS):
    m = stats.xs(model, level="model").reindex(names)
    ax2.barh(ypos + (k - 0.5) * 0.38, m["mean"], height=0.38, xerr=m["std"], label=model)
ax2.set_yticks(ypos)
ax2.set_yticklabels(names, fontsize=8)
ax2.invert_yaxis()
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Calibrated policy saving vs best line-drawing (%)")
ax2.set_title("Saving under each cost assumption")
ax2.legend()

fig.tight_layout()
fig.savefig("results/figures/phase3_decisions.png", dpi=200)
print("Figure saved to results/figures/phase3_decisions.png")