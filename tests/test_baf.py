"""Tests for BAF loading/splitting/preprocessing on synthetic BAF-shaped data."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import preprocess_baf
from src.data.splits import temporal_split_baf


@pytest.fixture
def baf_like_df():
    rng = np.random.default_rng(11)
    n = 8_000
    return pd.DataFrame(
        {
            "fraud_bool": rng.choice([0, 1], n, p=[0.989, 0.011]),
            "month": rng.integers(0, 8, n),
            "income": rng.uniform(0, 1, n),
            "customer_age": rng.integers(18, 90, n),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD"], n),
            "employment_status": rng.choice(["CA", "CB", "CC"], n),
            "device_os": rng.choice(["windows", "linux", "macintosh", "other"], n),
        }
    )


class TestTemporalSplitBAF:
    def test_month_boundaries_respected(self, baf_like_df):
        splits = temporal_split_baf(baf_like_df)
        month = baf_like_df["month"].values
        assert set(month[splits["train"]]) == {0, 1, 2, 3}
        assert set(month[splits["calibration"]]) == {4}
        assert set(month[splits["val"]]) == {5}
        assert set(month[splits["test"]]) == {6, 7}

    def test_no_temporal_leakage(self, baf_like_df):
        splits = temporal_split_baf(baf_like_df)
        month = baf_like_df["month"].values
        assert month[splits["train"]].max() < month[splits["calibration"]].min()
        assert month[splits["calibration"]].max() < month[splits["val"]].min()
        assert month[splits["val"]].max() < month[splits["test"]].min()

    def test_all_months_partition_rows(self, baf_like_df):
        splits = temporal_split_baf(baf_like_df)
        total = sum(len(v) for v in splits.values())
        assert total == len(baf_like_df)


class TestPreprocessBAF:
    def test_month_and_target_excluded_from_features(self, baf_like_df):
        splits = temporal_split_baf(baf_like_df)
        X, y = preprocess_baf(baf_like_df, train_idx=splits["train"])
        assert "month" not in X.columns
        assert "fraud_bool" not in X.columns
        assert len(y) == len(baf_like_df)

    def test_categoricals_are_numeric_after_encoding(self, baf_like_df):
        splits = temporal_split_baf(baf_like_df)
        X, _ = preprocess_baf(baf_like_df, train_idx=splits["train"])
        assert X.select_dtypes(include=["object"]).empty

    def test_unseen_test_category_maps_to_missing_token(self, baf_like_df):
        # Inject a category that exists ONLY in the test months: the encoder
        # (fitted on train) must not crash and must map it like "_missing_".
        df = baf_like_df.copy()
        test_rows = df.index[df["month"] >= 6][:50]
        df.loc[test_rows, "device_os"] = "brand_new_os"
        splits = temporal_split_baf(df)
        X, _ = preprocess_baf(df, train_idx=splits["train"])
        assert X.select_dtypes(include=["object"]).empty
        assert np.isfinite(X["device_os"].values).all()
