"""
Business-relevant metrics for fraud detection evaluation.

Translates model performance into monetary and operational terms.
"""

from typing import Dict

import numpy as np


def compute_business_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    amounts: np.ndarray = None,
    thresholds: Dict[str, float] = None,
    costs: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Compute business-relevant metrics based on decision policy thresholds.

    Decision zones (by default):
        - auto_approve:   prob < 0.10
        - step_up_auth:   0.10 <= prob < 0.50
        - manual_review:  0.50 <= prob < 0.85
        - auto_block:     prob >= 0.85

    Args:
        probs: Predicted fraud probabilities
        y_true: True binary labels (1=fraud)
        amounts: Transaction amounts (optional, uses avg if None)
        thresholds: Decision threshold dict
        costs: Cost configuration dict
    """
    if thresholds is None:
        thresholds = {
            "auto_approve_max": 0.10,
            "step_up_auth_max": 0.50,
            "manual_review_max": 0.85,
        }
    if costs is None:
        costs = {
            "avg_transaction_amount": 150.0,
            "fraud_loss_multiplier": 1.0,
            "step_up_auth_cost": 0.50,
            "manual_review_cost": 5.00,
            "false_positive_cost": 2.00,
        }

    if amounts is None:
        amounts = np.full(len(probs), costs["avg_transaction_amount"])

    # Assign decisions
    auto_approve = probs < thresholds["auto_approve_max"]
    step_up = (probs >= thresholds["auto_approve_max"]) & (
        probs < thresholds["step_up_auth_max"]
    )
    manual_review = (probs >= thresholds["step_up_auth_max"]) & (
        probs < thresholds["manual_review_max"]
    )
    auto_block = probs >= thresholds["manual_review_max"]

    # Fraud losses: fraud that was auto-approved
    # Fraud losses: approved fraud is lost in full; step-up and review stop
# only a fraction of the fraud routed to them.
    step_up_catch = costs.get("step_up_catch_rate", 1.0)
    review_catch = costs.get("review_catch_rate", 1.0)
    fraud_value = amounts * y_true
    fraud_approved = y_true[auto_approve].sum()
    fraud_loss = (
        fraud_value[auto_approve].sum()
        + (1 - step_up_catch) * fraud_value[step_up].sum()
        + (1 - review_catch) * fraud_value[manual_review].sum()
    )
    fraud_loss *= costs["fraud_loss_multiplier"]
    

    # Operational costs
    step_up_cost = step_up.sum() * costs["step_up_auth_cost"]
    review_cost = manual_review.sum() * costs["manual_review_cost"]

    # Customer friction: legit transactions blocked
    legit_blocked = ((1 - y_true) * auto_block).sum()
    friction_cost = legit_blocked * costs["false_positive_cost"]

    # Total cost
    total_cost = fraud_loss + step_up_cost + review_cost + friction_cost

    n = len(probs)
    return {
        # Decision distribution
        "frac_auto_approve": auto_approve.mean(),
        "frac_step_up": step_up.mean(),
        "frac_manual_review": manual_review.mean(),
        "frac_auto_block": auto_block.mean(),
        # Loss components
        "fraud_loss": fraud_loss,
        "n_fraud_approved": int(fraud_approved),
        "step_up_cost": step_up_cost,
        "review_cost": review_cost,
        "friction_cost": friction_cost,
        "total_cost": total_cost,
        "cost_per_transaction": total_cost / n if n > 0 else 0,
        # Rates
        "fraud_catch_rate": 1 - (fraud_approved / max(y_true.sum(), 1)),
        "legit_blocked_rate": legit_blocked / max((1 - y_true).sum(), 1),
        "review_load": (step_up.sum() + manual_review.sum()) / n if n > 0 else 0,
    }


def pareto_frontier(
    probs: np.ndarray,
    y_true: np.ndarray,
    amounts: np.ndarray = None,
    n_points: int = 50,
    costs: Dict[str, float] = None,
) -> Dict[str, np.ndarray]:
    """
    Sweep threshold configurations to trace the Pareto frontier
    of fraud_loss vs review_cost+friction.

    Returns arrays suitable for plotting the trade-off curve.
    """
    threshold_configs = []
    for approve_t in np.linspace(0.01, 0.40, n_points):
        for block_t in np.linspace(0.60, 0.99, 5):
            threshold_configs.append({
                "auto_approve_max": approve_t,
                "step_up_auth_max": (approve_t + block_t) / 2,
                "manual_review_max": block_t,
            })

    results = {
        "fraud_loss": [],
        "operational_cost": [],
        "fraud_catch_rate": [],
        "review_load": [],
        "thresholds": [],
    }

    for thresholds in threshold_configs:
        metrics = compute_business_metrics(
            probs, y_true, amounts, thresholds, costs
        )
        results["fraud_loss"].append(metrics["fraud_loss"])
        results["operational_cost"].append(
            metrics["step_up_cost"] + metrics["review_cost"] + metrics["friction_cost"]
        )
        results["fraud_catch_rate"].append(metrics["fraud_catch_rate"])
        results["review_load"].append(metrics["review_load"])
        results["thresholds"].append(thresholds)

    for key in ["fraud_loss", "operational_cost", "fraud_catch_rate", "review_load"]:
        results[key] = np.array(results[key])

    return results
