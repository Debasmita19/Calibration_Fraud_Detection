"""
Temporal and domain-based data splitting strategies.

These splits simulate realistic deployment conditions where models are trained
on historical data and evaluated on future (shifted) data.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split_ieee_cis(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    cal_frac: float = 0.15,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Split IEEE-CIS data by TransactionDT to simulate temporal deployment.

    Returns dict with keys: 'train', 'calibration', 'val', 'test',
    each containing row indices.
    """
    assert abs(train_frac + cal_frac + val_frac + test_frac - 1.0) < 1e-6

    sorted_idx = df["TransactionDT"].argsort().values
    n = len(sorted_idx)

    boundaries = np.cumsum([train_frac, cal_frac, val_frac, test_frac])
    splits = np.split(
        sorted_idx,
        (boundaries[:-1] * n).astype(int),
    )

    return {
        "train": splits[0],
        "calibration": splits[1],
        "val": splits[2],
        "test": splits[3],
    }


def temporal_split_creditcard(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    cal_frac: float = 0.15,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Split Credit Card data by Time column to simulate temporal deployment.
    """
    assert abs(train_frac + cal_frac + val_frac + test_frac - 1.0) < 1e-6

    sorted_idx = df["Time"].argsort().values
    n = len(sorted_idx)

    boundaries = np.cumsum([train_frac, cal_frac, val_frac, test_frac])
    splits = np.split(
        sorted_idx,
        (boundaries[:-1] * n).astype(int),
    )

    return {
        "train": splits[0],
        "calibration": splits[1],
        "val": splits[2],
        "test": splits[3],
    }


def temporal_split_baf(
    df: pd.DataFrame,
    train_months: Tuple[int, ...] = (0, 1, 2, 3),
    cal_months: Tuple[int, ...] = (4,),
    val_months: Tuple[int, ...] = (5,),
    test_months: Tuple[int, ...] = (6, 7),
) -> Dict[str, np.ndarray]:
    """
    Split a BAF variant by its explicit `month` column (0-7).

    Unlike fraction-based splits, these are semantic calendar boundaries:
    the model trains on months 0-3 and is tested on months 6-7, so test
    metrics measure genuine month-scale temporal drift.
    """
    month = df["month"].values

    def idx_for(months):
        return np.where(np.isin(month, months))[0]

    splits = {
        "train": idx_for(train_months),
        "calibration": idx_for(cal_months),
        "val": idx_for(val_months),
        "test": idx_for(test_months),
    }

    all_assigned = np.concatenate(list(splits.values()))
    assert len(np.unique(all_assigned)) == len(all_assigned), (
        "Split month groups must not overlap"
    )
    return splits


def sliding_window_splits(
    time_values: np.ndarray,
    n_windows: int = 5,
    train_window_frac: float = 0.5,
    cal_frac: float = 0.1,
) -> List[Dict[str, np.ndarray]]:
    """
    Create sliding-window temporal splits for evaluating calibration
    degradation over time.

    Each window uses historical data for training, a calibration set
    immediately after, and then a test window further in time.

    Returns a list of split dicts, one per window.
    """
    sorted_idx = np.argsort(time_values)
    n = len(sorted_idx)

    test_size = int(n * (1 - train_window_frac - cal_frac) / n_windows)
    train_end = int(n * train_window_frac)
    cal_end = int(n * (train_window_frac + cal_frac))

    windows = []
    for i in range(n_windows):
        test_start = cal_end + i * test_size
        test_end = min(test_start + test_size, n)
        if test_end <= test_start:
            break

        windows.append({
            "train": sorted_idx[:train_end],
            "calibration": sorted_idx[train_end:cal_end],
            "test": sorted_idx[test_start:test_end],
            "window_id": i,
        })

    return windows


def domain_shift_split(
    df: pd.DataFrame,
    split_column: str,
    source_values: list,
    target_values: list,
    cal_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create a domain-shift split by grouping on a categorical column.

    E.g., train on card_type='visa', test on card_type='mastercard'.
    """
    source_mask = df[split_column].isin(source_values)
    target_mask = df[split_column].isin(target_values)

    source_idx = np.where(source_mask)[0]
    target_idx = np.where(target_mask)[0]

    train_idx, cal_idx = train_test_split(
        source_idx, test_size=cal_frac, random_state=seed
    )

    return {
        "train": train_idx,
        "calibration": cal_idx,
        "test": target_idx,
    }
