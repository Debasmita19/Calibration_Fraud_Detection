"""
Tests for src/calibration/post_hoc.py.

Strategy: manufacture a miscalibrated model with a known ground truth.
  1. Draw true probabilities p_true and labels y ~ Bernoulli(p_true).
  2. Give the calibrator distorted probabilities (overconfident: logits x 3).
  3. A working calibrator must recover most of the lost ECE on held-out data.
"""

import numpy as np
import pytest
from scipy.special import expit, logit

from src.calibration.post_hoc import CALIBRATORS, get_calibrator
from src.evaluation.calibration_metrics import expected_calibration_error


@pytest.fixture(scope="module")
def miscalibrated_data():
    rng = np.random.default_rng(42)
    n = 40_000
    p_true = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    # Overconfident model: sharpen logits by 3x (what temp scaling T=3 undoes).
    probs_model = expit(3.0 * logit(p_true))

    half = n // 2
    return {
        "cal_probs": probs_model[:half],
        "cal_y": y[:half],
        "test_probs": probs_model[half:],
        "test_y": y[half:],
    }


@pytest.mark.parametrize("name", list(CALIBRATORS.keys()))
def test_calibrator_reduces_ece(name, miscalibrated_data):
    d = miscalibrated_data
    ece_before = expected_calibration_error(d["test_probs"], d["test_y"])
    assert ece_before > 0.05, "sanity: input must be clearly miscalibrated"

    calibrator = get_calibrator(name)
    calibrator.fit(d["cal_probs"], d["cal_y"])
    calibrated = calibrator.calibrate(d["test_probs"])

    ece_after = expected_calibration_error(calibrated, d["test_y"])
    assert ece_after < 0.5 * ece_before, (
        f"{name}: ECE {ece_before:.4f} -> {ece_after:.4f}, expected >50% reduction"
    )


@pytest.mark.parametrize("name", list(CALIBRATORS.keys()))
def test_calibrated_output_in_unit_interval(name, miscalibrated_data):
    d = miscalibrated_data
    calibrator = get_calibrator(name)
    calibrator.fit(d["cal_probs"], d["cal_y"])
    out = calibrator.calibrate(d["test_probs"])
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert np.all(np.isfinite(out))


def test_temperature_scaling_recovers_known_temperature(miscalibrated_data):
    # Logits were multiplied by 3, so the fitted temperature should be ~3.
    d = miscalibrated_data
    ts = get_calibrator("temperature_scaling")
    ts.fit(d["cal_probs"], d["cal_y"])
    assert ts.temperature == pytest.approx(3.0, rel=0.15)


def test_temperature_scaling_preserves_ranking(miscalibrated_data):
    # Monotone map: order of predictions (hence AUROC) must be unchanged.
    d = miscalibrated_data
    ts = get_calibrator("temperature_scaling")
    ts.fit(d["cal_probs"], d["cal_y"])
    out = ts.calibrate(d["test_probs"])
    order_before = np.argsort(d["test_probs"])
    order_after = np.argsort(out)
    assert np.array_equal(order_before, order_after)


def test_get_calibrator_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown calibrator"):
        get_calibrator("does_not_exist")
