"""
Tests for src/data/splits.py.

The single most important invariant in the whole thesis: temporal splits must
never leak the future into the past. If train contains timestamps later than
test, every "shift" result is invalid.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    domain_shift_split,
    sliding_window_splits,
    temporal_split_creditcard,
    temporal_split_ieee_cis,
)


@pytest.fixture
def creditcard_like_df():
    rng = np.random.default_rng(0)
    n = 10_000
    # Unsorted arrival order, like the real CSV after any shuffling.
    times = rng.permutation(np.arange(n)).astype(float)
    return pd.DataFrame({"Time": times, "Class": rng.integers(0, 2, n)})


@pytest.fixture
def ieee_like_df():
    rng = np.random.default_rng(1)
    n = 10_000
    times = rng.permutation(np.arange(n)).astype(float)
    return pd.DataFrame(
        {
            "TransactionDT": times,
            "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n),
        }
    )


def _assert_temporal_order(df, splits, time_col):
    """max(time) of each earlier split must precede min(time) of the next."""
    order = ["train", "calibration", "val", "test"]
    for earlier, later in zip(order[:-1], order[1:]):
        t_earlier = df[time_col].values[splits[earlier]]
        t_later = df[time_col].values[splits[later]]
        assert t_earlier.max() < t_later.min(), (
            f"temporal leakage: {earlier} overlaps {later}"
        )


class TestTemporalSplits:
    def test_creditcard_no_leakage(self, creditcard_like_df):
        splits = temporal_split_creditcard(creditcard_like_df)
        _assert_temporal_order(creditcard_like_df, splits, "Time")

    def test_ieee_no_leakage(self, ieee_like_df):
        splits = temporal_split_ieee_cis(ieee_like_df)
        _assert_temporal_order(ieee_like_df, splits, "TransactionDT")

    def test_splits_partition_all_rows(self, creditcard_like_df):
        splits = temporal_split_creditcard(creditcard_like_df)
        all_idx = np.concatenate([splits[k] for k in splits])
        assert len(all_idx) == len(creditcard_like_df)
        assert len(np.unique(all_idx)) == len(creditcard_like_df)

    def test_split_fractions_respected(self, creditcard_like_df):
        n = len(creditcard_like_df)
        splits = temporal_split_creditcard(
            creditcard_like_df, train_frac=0.6, cal_frac=0.15, val_frac=0.1, test_frac=0.15
        )
        assert len(splits["train"]) == pytest.approx(0.6 * n, rel=0.01)
        assert len(splits["calibration"]) == pytest.approx(0.15 * n, rel=0.01)


class TestSlidingWindows:
    def test_test_windows_after_calibration(self, creditcard_like_df):
        times = creditcard_like_df["Time"].values
        windows = sliding_window_splits(times, n_windows=5)
        assert len(windows) > 0
        for w in windows:
            assert times[w["train"]].max() < times[w["calibration"]].min()
            assert times[w["calibration"]].max() < times[w["test"]].min()

    def test_test_windows_move_forward_in_time(self, creditcard_like_df):
        times = creditcard_like_df["Time"].values
        windows = sliding_window_splits(times, n_windows=5)
        starts = [times[w["test"]].min() for w in windows]
        assert starts == sorted(starts)


class TestDomainShiftSplit:
    def test_source_and_target_are_disjoint_domains(self, ieee_like_df):
        splits = domain_shift_split(
            ieee_like_df, "ProductCD", source_values=["W"], target_values=["C"]
        )
        source_products = ieee_like_df["ProductCD"].values[
            np.concatenate([splits["train"], splits["calibration"]])
        ]
        target_products = ieee_like_df["ProductCD"].values[splits["test"]]
        assert set(source_products) == {"W"}
        assert set(target_products) == {"C"}

    def test_train_and_calibration_disjoint(self, ieee_like_df):
        splits = domain_shift_split(
            ieee_like_df, "ProductCD", source_values=["W"], target_values=["C"]
        )
        assert len(np.intersect1d(splits["train"], splits["calibration"])) == 0
