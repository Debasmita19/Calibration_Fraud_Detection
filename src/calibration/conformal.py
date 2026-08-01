"""
Conformal prediction wrappers for fraud detection.

Provides distribution-free coverage guarantees on predictions,
which is especially valuable under distribution shift.
"""

from typing import Dict, Tuple

import numpy as np


class SplitConformalClassifier:
    """
    Split conformal prediction for binary classification.

    Given a calibration set, computes nonconformity scores and provides
    prediction sets with guaranteed marginal coverage at level 1-alpha.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Miscoverage rate. Prediction sets cover the true label
                   with probability >= 1 - alpha.
        """
        self.alpha = alpha
        self.threshold = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """
        Compute the conformal threshold from calibration data.

        Uses 1 - p(y_true) as the nonconformity score.
        """
        scores = np.where(y_true == 1, 1 - probs, probs)

        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)

        self.threshold = np.quantile(scores, quantile_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """
        Return prediction sets as a (n, 2) boolean array.
        Column 0 = class 0 included, column 1 = class 1 included.
        """
        sets = np.zeros((len(probs), 2), dtype=bool)

        # Class 0 included if score for class 0 <= threshold
        # Score for class 0 when true class is 0: probs (i.e., 1 - p(class=0))
        # But since we don't know true class, we check both:
        sets[:, 0] = probs <= self.threshold       # class 0 plausible
        sets[:, 1] = (1 - probs) <= self.threshold  # class 1 plausible

        return sets

    def get_set_sizes(self, probs: np.ndarray) -> np.ndarray:
        """Return the size of each prediction set (0, 1, or 2)."""
        sets = self.predict_sets(probs)
        return sets.sum(axis=1)

    def evaluate_coverage(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate empirical coverage and average set size."""
        sets = self.predict_sets(probs)
        covered = sets[np.arange(len(y_true)), y_true.astype(int)]
        sizes = sets.sum(axis=1)

        return {
            "empirical_coverage": covered.mean(),
            "target_coverage": 1 - self.alpha,
            "avg_set_size": sizes.mean(),
            "frac_empty": (sizes == 0).mean(),
            "frac_singleton": (sizes == 1).mean(),
            "frac_both": (sizes == 2).mean(),
        }


class CostAwareConformal(SplitConformalClassifier):
    """
    Cost-aware conformal wrapper that uses asymmetric nonconformity scores
    to reflect the different costs of missing fraud vs. blocking legit.

    This is a novel contribution direction: conformal prediction with
    fraud-specific cost asymmetry.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        fn_cost_weight: float = 10.0,
        fp_cost_weight: float = 1.0,
    ):
        """
        Args:
            alpha: Miscoverage rate
            fn_cost_weight: Relative cost of missing a fraud (false negative)
            fp_cost_weight: Relative cost of blocking a legit (false positive)
        """
        super().__init__(alpha)
        self.fn_cost_weight = fn_cost_weight
        self.fp_cost_weight = fp_cost_weight

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """Compute threshold using cost-weighted nonconformity scores."""
        # For fraudulent transactions (y=1): score = fn_weight * (1 - p_fraud)
        # For legitimate transactions (y=0): score = fp_weight * p_fraud
        scores = np.where(
            y_true == 1,
            self.fn_cost_weight * (1 - probs),
            self.fp_cost_weight * probs,
        )

        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)

        self.threshold = np.quantile(scores, quantile_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Prediction sets with cost-asymmetric thresholds."""
        sets = np.zeros((len(probs), 2), dtype=bool)
        sets[:, 0] = (self.fp_cost_weight * probs) <= self.threshold
        sets[:, 1] = (self.fn_cost_weight * (1 - probs)) <= self.threshold
        return sets
