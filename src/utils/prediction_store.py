"""
Per-sample prediction store.

Every base-model run saves its raw predicted probabilities once, per sample,
alongside identifiers, timestamps, labels, split membership, and the seed.
All calibrators, conformal methods, and decision policies then operate on
these frozen predictions, guaranteeing that method comparisons are not
confounded by training randomness, and that expensive base models never need
refitting just to evaluate a new (cheap) calibration method.

Schema: sample_id, time_value, y_true, y_prob, split, dataset, model, seed
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_STORE_DIR = "results/predictions"


def _store_path(store_dir: str, dataset: str, model: str, seed: int) -> Path:
    return Path(store_dir) / f"{dataset}_{model}_seed{seed}.parquet"


def save_predictions(
    dataset: str,
    model: str,
    seed: int,
    sample_ids: np.ndarray,
    time_values: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    splits: Dict[str, np.ndarray],
    store_dir: str = DEFAULT_STORE_DIR,
) -> Path:
    """
    Save one model run's per-sample predictions to parquet.

    Args:
        sample_ids: unique identifier per row (e.g., TransactionID or row index).
        time_values: temporal ordering value per row (TransactionDT / Time / month).
        y_true, y_prob: labels and predicted fraud probabilities, aligned to rows.
        splits: dict mapping split name -> row indices (as produced by
            src.data.splits). Rows not present in any split get split="unused".

    Returns the path written.
    """
    n = len(y_prob)
    split_labels = np.full(n, "unused", dtype=object)
    for split_name, idx in splits.items():
        if split_name == "window_id":
            continue
        split_labels[np.asarray(idx)] = split_name

    df = pd.DataFrame(
        {
            "sample_id": np.asarray(sample_ids),
            "time_value": np.asarray(time_values, dtype=float),
            "y_true": np.asarray(y_true, dtype=int),
            "y_prob": np.asarray(y_prob, dtype=float),
            "split": split_labels,
            "dataset": dataset,
            "model": model,
            "seed": seed,
        }
    )

    path = _store_path(store_dir, dataset, model, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_predictions(
    dataset: str,
    model: str,
    seed: int,
    split: Optional[str] = None,
    store_dir: str = DEFAULT_STORE_DIR,
) -> pd.DataFrame:
    """
    Load a saved prediction table, optionally filtered to one split.

    Raises FileNotFoundError with a helpful message if the run wasn't saved.
    """
    path = _store_path(store_dir, dataset, model, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction store at {path}. "
            f"Run the experiment for dataset={dataset}, model={model}, "
            f"seed={seed} first."
        )

    df = pd.read_parquet(path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def list_available_runs(store_dir: str = DEFAULT_STORE_DIR) -> pd.DataFrame:
    """Return a DataFrame of (dataset, model, seed) runs present in the store."""
    store = Path(store_dir)
    rows = []
    if store.exists():
        for f in sorted(store.glob("*_seed*.parquet")):
            stem = f.stem  # {dataset}_{model}_seed{seed}
            base, _, seed_part = stem.rpartition("_seed")
            dataset, _, model = base.partition("_")
            rows.append({"dataset": dataset, "model": model, "seed": int(seed_part)})
    return pd.DataFrame(rows, columns=["dataset", "model", "seed"])
