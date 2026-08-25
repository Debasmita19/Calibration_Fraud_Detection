"""
Calibration evaluation metrics.

Implements ECE, MCE, Brier score, and reliability diagram data computation.
"""

from typing import Dict, List, Tuple

import numpy as np


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE).

    Weighted average of |accuracy - confidence| across probability bins.
    Lower is better; 0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece


def adaptive_expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """
    Adaptive ECE with equal-mass (equal-count) bins.

    Standard ECE uses equal-width bins, which are unstable under extreme
    class imbalance: nearly all fraud predictions sit near 0, leaving most
    bins empty. Equal-mass binning puts the same number of samples in every
    bin (quantile bin edges), so each bin's accuracy/confidence estimate is
    statistically stable. Recommended alongside standard ECE (Nixon et al.,
    2019, "Measuring Calibration in Deep Learning").
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_y = y_true[order]

    # Equal-count bin boundaries over the sorted predictions.
    bin_assignments = np.floor(np.arange(n) * n_bins / n).astype(int)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_assignments == b
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = sorted_probs[mask].mean()
        avg_accuracy = sorted_y[mask].mean()
        ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)

    return ece


def maximum_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """Maximum Calibration Error (MCE) — worst-case bin miscalibration."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = y_true[mask].mean()
        mce = max(mce, abs(avg_accuracy - avg_confidence))

    return mce


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Score — mean squared error of probabilistic predictions."""
    return np.mean((probs - y_true) ** 2)


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Negative log-likelihood (cross-entropy loss)."""
    # Cast to float64 BEFORE clipping: in float32, 1 - 1e-10 rounds to
    # exactly 1.0, so log(1 - p) becomes log(0) = -inf for confident
    # predictions (models like XGBoost emit float32).
    probs64 = np.asarray(probs, dtype=np.float64)
    probs_clipped = np.clip(probs64, 1e-10, 1 - 1e-10)
    return -np.mean(
        y_true * np.log(probs_clipped)
        + (1 - y_true) * np.log(1 - probs_clipped)
    )


def reliability_diagram_data(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> Dict[str, np.ndarray]:
    """
    Compute data for plotting a reliability diagram.

    Returns:
        Dictionary with bin_centers, bin_accuracies, bin_confidences,
        bin_counts for plotting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_bin = mask.sum()
        bin_counts.append(n_bin)

        if n_bin > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(y_true[mask].mean())
            bin_confidences.append(probs[mask].mean())
        else:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(0)
            bin_confidences.append(0)

    return {
        "bin_centers": np.array(bin_centers),
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts),
        "bin_edges": bin_edges,
    }


def compute_all_calibration_metrics(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> Dict[str, float]:
    """Compute all calibration metrics in one call."""
    return {
        "ece": expected_calibration_error(probs, y_true, n_bins),
        "adaptive_ece": adaptive_expected_calibration_error(probs, y_true, n_bins),
        "mce": maximum_calibration_error(probs, y_true, n_bins),
        "brier_score": brier_score(probs, y_true),
        "nll": negative_log_likelihood(probs, y_true),
    }
