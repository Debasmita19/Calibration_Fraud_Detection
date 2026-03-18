"""Dataset loading and preprocessing for fraud detection experiments."""

from pathlib import Path
from typing import Tuple

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
    label_encoders = {}
    for col in cat_cols:
        if df[col].nunique() <= max_cat_cardinality:
            le = LabelEncoder()
            df[col] = df[col].fillna("_missing_").astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            kept_cat_cols.append(col)

    feature_cols = num_cols + kept_cat_cols
    X = df[feature_cols].copy()

    # Fill numeric NaN with median
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    return X, y, num_cols, kept_cat_cols


def preprocess_creditcard(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess Credit Card Fraud dataset.
    Features V1-V28 are already PCA-transformed; only scale Amount and Time.
    """
    target = "Class"
    y = df[target].values

    X = df.drop(columns=[target]).copy()

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    return X, y


def get_dataset(
    name: str, data_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Unified dataset loading interface.

    Args:
        name: "ieee_cis" or "creditcard"
        data_dir: root directory containing raw data subfolders

    Returns:
        (X, y) tuple
    """
    if name == "ieee_cis":
        df = load_ieee_cis(f"{data_dir}/ieee-fraud-detection")
        X, y, _, _ = preprocess_ieee_cis(df)
        return X, y
    elif name == "creditcard":
        df = load_creditcard(f"{data_dir}/creditcardfraud")
        X, y = preprocess_creditcard(df)
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'ieee_cis' or 'creditcard'.")
