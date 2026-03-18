"""
Risk-aware decision policies for fraud detection.

Maps calibrated confidence scores to actionable decisions with
configurable cost trade-offs.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


class TieredDecisionPolicy:
    """
    Maps predicted fraud probabilities to tiered decisions.

    Zones:
        1. Auto-approve (low risk) — no friction, risk of fraud loss
        2. Step-up authentication (medium risk) — mild friction (OTP, KYC)
        3. Manual review (high risk) — expensive but thorough
        4. Auto-block (very high risk) — no fraud loss, risk of false positives
    """

    def __init__(
        self,
        approve_threshold: float = 0.10,
        stepup_threshold: float = 0.50,
        review_threshold: float = 0.85,
    ):
        self.approve_threshold = approve_threshold
        self.stepup_threshold = stepup_threshold
        self.review_threshold = review_threshold

    def decide(self, probs: np.ndarray) -> np.ndarray:
        """
        Map probabilities to decisions.

        Returns:
            Array of strings: 'auto_approve', 'step_up', 'manual_review', 'auto_block'
        """
        decisions = np.full(len(probs), "manual_review", dtype=object)
        decisions[probs < self.approve_threshold] = "auto_approve"
        decisions[
            (probs >= self.approve_threshold) & (probs < self.stepup_threshold)
        ] = "step_up"
        decisions[
            (probs >= self.stepup_threshold) & (probs < self.review_threshold)
        ] = "manual_review"
        decisions[probs >= self.review_threshold] = "auto_block"
        return decisions

    def compute_expected_cost(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        amounts: np.ndarray = None,
        costs: dict = None,
    ) -> float:
        """Compute the total expected cost of the current policy."""
        from src.evaluation.business_metrics import compute_business_metrics

        thresholds = {
            "auto_approve_max": self.approve_threshold,
            "step_up_auth_max": self.stepup_threshold,
            "manual_review_max": self.review_threshold,
        }
        metrics = compute_business_metrics(
            probs, y_true, amounts, thresholds, costs
        )
        return metrics["total_cost"]


class AdaptiveThresholdOptimizer:
    """
    Optimize decision thresholds to minimize expected total cost.

    Uses calibrated probabilities to find the threshold configuration
    that best balances fraud loss against operational costs.
    """

    def __init__(self, costs: dict = None):
        self.costs = costs or {
            "avg_transaction_amount": 150.0,
            "fraud_loss_multiplier": 1.0,
            "step_up_auth_cost": 0.50,
            "manual_review_cost": 5.00,
            "false_positive_cost": 2.00,
        }
        self.best_thresholds = None

    def optimize(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        amounts: np.ndarray = None,
        n_grid: int = 20,
    ) -> Dict[str, float]:
        """
        Grid search over threshold configurations to minimize total cost.
        """
        best_cost = float("inf")
        best_config = None

        for approve_t in np.linspace(0.01, 0.30, n_grid):
            for stepup_t in np.linspace(
                approve_t + 0.05, 0.70, max(n_grid // 2, 5)
            ):
                for review_t in np.linspace(
                    stepup_t + 0.05, 0.99, max(n_grid // 3, 5)
                ):
                    policy = TieredDecisionPolicy(approve_t, stepup_t, review_t)
                    cost = policy.compute_expected_cost(
                        probs, y_true, amounts, self.costs
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_config = {
                            "auto_approve_max": approve_t,
                            "step_up_auth_max": stepup_t,
                            "manual_review_max": review_t,
                        }

        self.best_thresholds = best_config
        return {
            "best_thresholds": best_config,
            "best_total_cost": best_cost,
        }


def compare_calibrated_vs_uncalibrated_policies(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    y_true: np.ndarray,
    amounts: np.ndarray = None,
    costs: dict = None,
) -> Dict[str, Dict]:
    """
    Head-to-head comparison of decision policies using
    uncalibrated vs calibrated probabilities.
    """
    optimizer = AdaptiveThresholdOptimizer(costs)

    uncal_result = optimizer.optimize(uncalibrated_probs, y_true, amounts)
    cal_result = optimizer.optimize(calibrated_probs, y_true, amounts)

    return {
        "uncalibrated": uncal_result,
        "calibrated": cal_result,
        "cost_reduction": uncal_result["best_total_cost"] - cal_result["best_total_cost"],
        "cost_reduction_pct": (
            (uncal_result["best_total_cost"] - cal_result["best_total_cost"])
            / max(uncal_result["best_total_cost"], 1e-10)
            * 100
        ),
    }
