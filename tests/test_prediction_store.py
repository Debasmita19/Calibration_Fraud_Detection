"""Round-trip tests for the per-sample prediction store."""

import numpy as np
import pytest

from src.utils.prediction_store import (
    list_available_runs,
    load_predictions,
    save_predictions,
)


@pytest.fixture
def toy_run(tmp_path):
    rng = np.random.default_rng(7)
    n = 100
    splits = {
        "train": np.arange(0, 60),
        "calibration": np.arange(60, 75),
        "val": np.arange(75, 85),
        "test": np.arange(85, 100),
    }
    kwargs = {
        "dataset": "toyset",
        "model": "xgboost",
        "seed": 0,
        "sample_ids": np.arange(1000, 1000 + n),
        "time_values": np.arange(n, dtype=float),
        "y_true": rng.integers(0, 2, n),
        "y_prob": rng.uniform(0, 1, n),
        "splits": splits,
        "store_dir": str(tmp_path),
    }
    save_predictions(**kwargs)
    return kwargs


def test_round_trip_preserves_values(toy_run):
    df = load_predictions("toyset", "xgboost", 0, store_dir=toy_run["store_dir"])
    assert len(df) == 100
    np.testing.assert_array_equal(df["y_true"].values, toy_run["y_true"])
    np.testing.assert_allclose(df["y_prob"].values, toy_run["y_prob"])
    np.testing.assert_array_equal(df["sample_id"].values, toy_run["sample_ids"])


def test_split_labels_assigned_correctly(toy_run):
    test_df = load_predictions(
        "toyset", "xgboost", 0, split="test", store_dir=toy_run["store_dir"]
    )
    assert len(test_df) == 15
    assert set(test_df["sample_id"]) == set(range(1085, 1100))


def test_missing_run_raises_helpful_error(toy_run):
    with pytest.raises(FileNotFoundError, match="No prediction store"):
        load_predictions("toyset", "mlp", 99, store_dir=toy_run["store_dir"])


def test_list_available_runs(toy_run):
    runs = list_available_runs(store_dir=toy_run["store_dir"])
    assert len(runs) == 1
    assert runs.iloc[0]["dataset"] == "toyset"
    assert runs.iloc[0]["model"] == "xgboost"
    assert runs.iloc[0]["seed"] == 0


def test_list_available_runs_dataset_with_underscore(tmp_path):
    save_predictions(
        dataset="ieee_cis", model="lightgbm", seed=43,
        sample_ids=np.arange(4), time_values=np.arange(4, dtype=float),
        y_true=np.array([0, 1, 0, 1]), y_prob=np.array([0.1, 0.9, 0.2, 0.8]),
        splits={"test": np.arange(4)}, store_dir=str(tmp_path),
    )
    runs = list_available_runs(store_dir=str(tmp_path))
    assert runs.iloc[0]["dataset"] == "ieee_cis"
    assert runs.iloc[0]["model"] == "lightgbm"
    assert runs.iloc[0]["seed"] == 43
