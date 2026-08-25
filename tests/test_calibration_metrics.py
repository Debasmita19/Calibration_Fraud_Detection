"""
Tests for src/evaluation/calibration_metrics.py.

Each metric is verified against values computed by hand on tiny examples,
so a regression in the implementation is caught immediately.
"""

import numpy as np
import pytest

from src.evaluation.calibration_metrics import (
    adaptive_expected_calibration_error,
    brier_score,
    compute_all_calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_diagram_data,
)


# Hand-computed example (10 bins):
#   probs  = [0.9, 0.9, 0.1, 0.1], labels = [1, 0, 0, 0]
#   bin [0.9, 1.0]: conf 0.9, acc 0.5 -> gap 0.4, weight 2/4
#   bin [0.1, 0.2): conf 0.1, acc 0.0 -> gap 0.1, weight 2/4
#   ECE = 0.5*0.4 + 0.5*0.1 = 0.25 ; MCE = 0.4
PROBS = np.array([0.9, 0.9, 0.1, 0.1])
LABELS = np.array([1, 0, 0, 0])


class TestECE:
    def test_hand_computed_value(self):
        assert expected_calibration_error(PROBS, LABELS, n_bins=10) == pytest.approx(0.25)

    def test_perfectly_calibrated_is_zero(self):
        # 10 predictions of 0.3 with exactly 3 positives: conf == acc in its bin.
        probs = np.full(10, 0.3)
        labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)

    def test_maximally_wrong_is_one(self):
        # Model says ~1.0 for guaranteed negatives.
        probs = np.full(100, 0.999)
        labels = np.zeros(100)
        assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(
            0.999, abs=1e-3
        )


class TestMCE:
    def test_hand_computed_value(self):
        assert maximum_calibration_error(PROBS, LABELS, n_bins=10) == pytest.approx(0.4)

    def test_mce_at_least_ece(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 500)
        labels = (rng.uniform(0, 1, 500) < probs**2).astype(int)
        ece = expected_calibration_error(probs, labels)
        mce = maximum_calibration_error(probs, labels)
        assert mce >= ece


class TestBrier:
    def test_hand_computed_value(self):
        # ((0.9-1)^2 + 0.9^2 + 0.1^2 + 0.1^2) / 4 = 0.84 / 4 = 0.21
        assert brier_score(PROBS, LABELS) == pytest.approx(0.21)

    def test_perfect_predictions(self):
        assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


class TestNLL:
    def test_hand_computed_value(self):
        probs = np.array([0.8, 0.2])
        labels = np.array([1, 0])
        expected = -np.mean([np.log(0.8), np.log(0.8)])
        assert negative_log_likelihood(probs, labels) == pytest.approx(expected)

    def test_finite_on_hard_zero_one_probs(self):
        # Isotonic regression emits exact 0s and 1s; NLL must not become inf/NaN.
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        labels = np.array([1, 0, 0, 1])
        nll = negative_log_likelihood(probs, labels)
        assert np.isfinite(nll)

    def test_finite_on_float32_confident_probs(self):
        # Regression test: XGBoost emits float32, where clip(p, ., 1 - 1e-10)
        # rounds to exactly 1.0 and log(1 - p) exploded to -inf.
        probs = np.array([0.9999999, 1.0, 0.0000001], dtype=np.float32)
        labels = np.array([1, 1, 0])
        nll = negative_log_likelihood(probs, labels)
        assert np.isfinite(nll)


class TestReliabilityDiagram:
    def test_bin_counts_sum_to_n(self):
        data = reliability_diagram_data(PROBS, LABELS, n_bins=10)
        assert data["bin_counts"].sum() == len(PROBS)

    def test_bin_stats_match_hand_example(self):
        data = reliability_diagram_data(PROBS, LABELS, n_bins=10)
        top_bin = -1  # [0.9, 1.0]
        assert data["bin_confidences"][top_bin] == pytest.approx(0.9)
        assert data["bin_accuracies"][top_bin] == pytest.approx(0.5)


class TestAdaptiveECE:
    def test_perfectly_calibrated_is_near_zero(self):
        rng = np.random.default_rng(3)
        n = 50_000
        probs = rng.uniform(0, 1, n)
        labels = (rng.uniform(0, 1, n) < probs).astype(int)
        assert adaptive_expected_calibration_error(probs, labels) < 0.02

    def test_detects_miscalibration_missed_by_equal_width_bins(self):
        # Fraud-like skew: all predictions in [0, 0.02) land in ONE equal-width
        # bin, hiding structure. Equal-mass bins split them into 15 groups.
        rng = np.random.default_rng(4)
        n = 50_000
        probs = rng.uniform(0, 0.02, n)
        # True fraud rate is 2x the predicted probability everywhere.
        labels = (rng.uniform(0, 1, n) < 2 * probs).astype(int)
        standard = expected_calibration_error(probs, labels)
        adaptive = adaptive_expected_calibration_error(probs, labels)
        # Both should detect it, but adaptive must not be blind to it.
        assert adaptive > 0.005
        assert adaptive >= 0.8 * standard

    def test_all_samples_used(self):
        # Sanity on the tiny hand example: weights sum to 1, value finite.
        val = adaptive_expected_calibration_error(PROBS, LABELS, n_bins=2)
        assert np.isfinite(val)
        # With 2 equal-mass bins: low bin = [0.1, 0.1] (acc 0, conf 0.1),
        # high bin = [0.9, 0.9] (acc 0.5, conf 0.9) -> 0.5*0.1 + 0.5*0.4 = 0.25
        assert val == pytest.approx(0.25)


class TestComputeAll:
    def test_returns_all_keys_and_matches_individuals(self):
        metrics = compute_all_calibration_metrics(PROBS, LABELS, n_bins=10)
        assert metrics["ece"] == pytest.approx(0.25)
        assert metrics["mce"] == pytest.approx(0.4)
        assert metrics["brier_score"] == pytest.approx(0.21)
        assert "adaptive_ece" in metrics
        assert np.isfinite(metrics["nll"])
