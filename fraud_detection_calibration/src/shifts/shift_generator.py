"""
Synthetic distribution shift generators for controlled experiments.

Creates various types of shifts to test calibration robustness:
- Feature drift (covariate shift)
- New fraud modes (prior shift + concept drift)
- Label noise
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def apply_covariate_shift(
    X: pd.DataFrame,
    columns: list = None,
    shift_magnitude: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply covariate shift by perturbing feature distributions.

    Adds Gaussian noise scaled by each feature's standard deviation,
    simulating gradual drift in transaction characteristics.
    """
    rng = np.random.RandomState(seed)
    X_shifted = X.copy()

    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        std = X[col].std()
        noise = rng.normal(0, shift_magnitude * std, size=len(X))
        X_shifted[col] = X_shifted[col] + noise

    return X_shifted


def inject_new_fraud_mode(
    X: pd.DataFrame,
    y: np.ndarray,
    new_fraud_fraction: float = 0.05,
    n_shifted_features: int = 5,
    shift_scale: float = 3.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Inject a new fraud pattern that differs from training distribution.

    Creates synthetic fraud samples with shifted feature values,
    simulating a new attack strategy that the model hasn't seen.
    """
    rng = np.random.RandomState(seed)
    n_new_fraud = int(len(X) * new_fraud_fraction)

    # Sample base patterns from existing fraud
    fraud_idx = np.where(y == 1)[0]
    if len(fraud_idx) == 0:
        raise ValueError("No fraud samples to base new patterns on")

    base_idx = rng.choice(fraud_idx, size=n_new_fraud, replace=True)
    new_fraud_X = X.iloc[base_idx].copy().reset_index(drop=True)

    # Shift a subset of features to create a "new" attack pattern
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    shift_cols = rng.choice(
        numeric_cols,
        size=min(n_shifted_features, len(numeric_cols)),
        replace=False,
    )

    for col in shift_cols:
        col_std = X[col].std()
        direction = rng.choice([-1, 1])
        new_fraud_X[col] = new_fraud_X[col] + direction * shift_scale * col_std

    new_fraud_y = np.ones(n_new_fraud, dtype=y.dtype)

    X_combined = pd.concat([X, new_fraud_X], ignore_index=True)
    y_combined = np.concatenate([y, new_fraud_y])

    return X_combined, y_combined


def apply_label_noise(
    y: np.ndarray,
    noise_rate: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """
    Flip a fraction of labels to simulate labeling errors,
    which is common in fraud detection (delayed chargebacks, etc.).
    """
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


def create_shift_scenarios(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Generate a suite of shifted datasets for systematic evaluation.

    Returns a dict mapping scenario_name -> (X_shifted, y_shifted).
    """
    scenarios = {"no_shift": (X.copy(), y.copy())}

    # Mild covariate shift
    scenarios["mild_covariate_shift"] = (
        apply_covariate_shift(X, shift_magnitude=0.1),
        y.copy(),
    )

    # Strong covariate shift
    scenarios["strong_covariate_shift"] = (
        apply_covariate_shift(
            X, shift_magnitude=config.get("feature_drift_magnitude", 0.3)
        ),
        y.copy(),
    )

    # New fraud mode
    X_new, y_new = inject_new_fraud_mode(
        X, y,
        new_fraud_fraction=config.get("new_fraud_fraction", 0.05),
    )
    scenarios["new_fraud_mode"] = (X_new, y_new)

    # Label noise
    scenarios["label_noise"] = (
        X.copy(),
        apply_label_noise(y, noise_rate=config.get("label_noise_rate", 0.02)),
    )

    # Combined: covariate shift + new fraud mode
    X_combined, y_combined = inject_new_fraud_mode(
        apply_covariate_shift(X, shift_magnitude=0.2),
        y,
        new_fraud_fraction=0.03,
    )
    scenarios["combined_shift"] = (X_combined, y_combined)

    return scenarios
