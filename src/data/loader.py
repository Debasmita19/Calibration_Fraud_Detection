"""Dataset loading and preprocessing for fraud detection experiments."""

from pathlib import Path
from typing import Tuple

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_ieee_cis(
    data_dir: str = "data/raw/ieee-fraud-detection",
) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS transaction + identity tables.

    Returns a single DataFrame with all features and the isFraud target.
    """
    data_path = Path(data_dir)

    train_txn = pd.read_csv(data_path / "train_transaction.csv")
    train_id = pd.read_csv(data_path / "train_identity.csv")

    df = train_txn.merge(train_id, on="TransactionID", how="left")
    return df


def load_creditcard(
    data_dir: str = "data/raw/creditcardfraud",
) -> pd.DataFrame:
    """Load the European credit card fraud dataset."""
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / "creditcard.csv")
    return df


def preprocess_ieee_cis(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    max_cat_cardinality: int = 50,
) -> Tuple[pd.DataFrame, np.ndarray, list, list]:
    """
    Preprocess IEEE-CIS data:
    - Encode categoricals with cardinality <= max_cat_cardinality
    - Drop very high-cardinality categoricals
    - Fill missing values
    - Return (X, y, numeric_cols, categorical_cols)
    """
    target = "isFraud"
    drop_cols = ["TransactionID", "TransactionDT"]

    y = df[target].values

    # Separate TransactionDT for temporal splitting before dropping
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in [target] + drop_cols
    ]

    kept_cat_cols = []
    for col in cat_cols:
        if df[col].nunique() <= max_cat_cardinality:
            df[col] = df[col].fillna("_missing_").astype(str)
            # Categories fitted on train only; unseen test categories -> _missing_
            train_categories = set(df[col].iloc[train_idx].unique())
            df[col] = df[col].where(df[col].isin(train_categories), "_missing_")
            le = LabelEncoder()
            # Include "_missing_" so unseen test categories can always be encoded
            le.fit(pd.concat([df[col].iloc[train_idx], pd.Series(["_missing_"])]))
            df[col] = le.transform(df[col])
            kept_cat_cols.append(col)
    feature_cols = num_cols + kept_cat_cols
    X = df[feature_cols].copy()
    # Fill numeric NaN with train-only median
    train_medians = X.iloc[train_idx][num_cols].median()
    X[num_cols] = X[num_cols].fillna(train_medians)
    return X, y, num_cols, kept_cat_cols


def preprocess_creditcard(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess Credit Card Fraud dataset.
    Features V1-V28 are already PCA-transformed; only scale Amount and Time.
    Scaler is fit on train rows only to avoid temporal leakage.
    """
    target = "Class"
    y = df[target].values

    X = df.drop(columns=[target]).copy()

    scaler = StandardScaler()
    scaler.fit(X.iloc[train_idx][["Time", "Amount"]])
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    return X, y


def get_dataset(
    name: str, data_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load raw data, compute temporal splits, then preprocess with
    statistics fitted on the train partition only.

    Returns:
        (X, y, splits) tuple
    """
    from src.data.splits import temporal_split_creditcard, temporal_split_ieee_cis

    if name == "ieee_cis":
        df = load_ieee_cis(f"{data_dir}/ieee-fraud-detection")
        splits = temporal_split_ieee_cis(df)
        X, y, _, _ = preprocess_ieee_cis(df, train_idx=splits["train"])
    elif name == "creditcard":
        df = load_creditcard(f"{data_dir}/creditcardfraud")
        splits = temporal_split_creditcard(df)
        X, y = preprocess_creditcard(df, train_idx=splits["train"])
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'ieee_cis' or 'creditcard'.")

    return X, y, splits