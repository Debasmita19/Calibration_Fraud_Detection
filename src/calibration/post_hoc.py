"""
Post-hoc calibration methods for fraud detection.

Implements:
- Temperature Scaling (Guo et al., 2017)
- Platt Scaling (logistic regression on logits)
- Isotonic Regression
- Beta Calibration
- Histogram Binning
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratorBase:
    """Abstract base for all calibration methods."""

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TemperatureScaling(CalibratorBase):
    """
    Temperature Scaling (Guo et al., 2017).
    Learns a single scalar T that rescales logits: p_cal = sigma(logit(p) / T).
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        logits = logit(probs_clipped)

        def nll_loss(T):
            scaled_logits = logits / T[0]
            scaled_probs = expit(scaled_logits)
            scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
            return -np.mean(
                y_true * np.log(scaled_probs)
                + (1 - y_true) * np.log(1 - scaled_probs)
            )

        result = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 100.0)])
        self.temperature = result.x[0]
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        logits = logit(probs_clipped)
        return expit(logits / self.temperature)


class PlattScaling(CalibratorBase):
    """Platt Scaling: fit a logistic regression on model outputs."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self.lr.fit(probs.reshape(-1, 1), y_true)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibration(CalibratorBase):
    """Isotonic Regression calibration (non-parametric, monotonic)."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self.iso.fit(probs, y_true)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return self.iso.predict(probs)


class BetaCalibration(CalibratorBase):
    """
    Beta Calibration (Kull et al., 2017).
    Fits: P(y=1|s) = 1 / (1 + 1/exp(a*log(s/(1-s)) + b*log(s) + c))
    Simplified to logistic on [log(s/(1-s)), log(s)].
    """

    def __init__(self):
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)

    def _features(self, probs: np.ndarray) -> np.ndarray:
        probs64 = np.asarray(probs, dtype=np.float64)
        probs_clipped = np.clip(probs64, 1e-10, 1.0 - 1e-10)
        log_odds = np.log(probs_clipped / (1.0 - probs_clipped))
        log_probs = np.log(probs_clipped)
        return np.column_stack([log_odds, log_probs])

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        features = self._features(probs)
        self.lr.fit(features, y_true)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        features = self._features(probs)
        return self.lr.predict_proba(features)[:, 1]


class HistogramBinning(CalibratorBase):
    """
    Histogram Binning calibration.
    Assigns each bin the empirical fraction of positives.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_values = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self.bin_values = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (probs >= self.bin_edges[i]) & (probs < self.bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = (probs >= self.bin_edges[i]) & (probs <= self.bin_edges[i + 1])
            if mask.sum() > 0:
                self.bin_values[i] = y_true[mask].mean()
            else:
                self.bin_values[i] = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probs)
        for i in range(self.n_bins):
            mask = (probs >= self.bin_edges[i]) & (probs < self.bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = (probs >= self.bin_edges[i]) & (probs <= self.bin_edges[i + 1])
            calibrated[mask] = self.bin_values[i]
        return calibrated


CALIBRATORS = {
    "temperature_scaling": TemperatureScaling,
    "platt_scaling": PlattScaling,
    "isotonic_regression": IsotonicCalibration,
    "beta_calibration": BetaCalibration,
    "histogram_binning": HistogramBinning,
}


def get_calibrator(name: str, **kwargs) -> CalibratorBase:
    """Factory function to create a calibrator by name."""
    if name not in CALIBRATORS:
        raise ValueError(
            f"Unknown calibrator: {name}. Choose from {list(CALIBRATORS.keys())}"
        )
    return CALIBRATORS[name](**kwargs)
