"""Dataset loading and preprocessing for fraud detection experiments."""

from pathlib import Path
from typing import Tuple

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_ieee_cis(
    data_dir: str = "data/raw/ieee-fraud-detection",
    cache_dir: str = "data/processed",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS transaction + identity tables.

    The merged table (~650MB of CSV parsing + a join) is cached to parquet on
    first load; later loads take seconds instead of minutes. Pass
    use_cache=False to force a rebuild (e.g., after changing the merge logic).

    Returns a single DataFrame with all features and the isFraud target.
    """
    cache_path = Path(cache_dir) / "ieee_cis_raw.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    data_path = Path(data_dir)

    train_txn = pd.read_csv(data_path / "train_transaction.csv")
    train_id = pd.read_csv(data_path / "train_identity.csv")

    df = train_txn.merge(train_id, on="TransactionID", how="left")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def load_creditcard(
    data_dir: str = "data/raw/creditcardfraud",
) -> pd.DataFrame:
    """Load the European credit card fraud dataset."""
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / "creditcard.csv")
    return df


BAF_VARIANTS = ["Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]


def load_baf(
    variant: str = "Base",
    data_dir: str = "data/raw/baf",
    cache_dir: str = "data/processed",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load one variant of the Bank Account Fraud (BAF) suite (NeurIPS 2022).

    BAF is realistic synthetic data generated from a real bank-account-opening
    fraud dataset. Each variant has ~1M rows, a `month` column (0-7) for
    temporal experiments, and `fraud_bool` as the target. The 6 variants have
    deliberately different distributions, enabling controlled domain-shift
    experiments (train on Base, test on a variant).
    """
    if variant not in BAF_VARIANTS:
        raise ValueError(f"Unknown BAF variant: {variant}. Choose from {BAF_VARIANTS}")

    cache_name = f"baf_{variant.lower().replace(' ', '_')}.parquet"
    cache_path = Path(cache_dir) / cache_name

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(data_dir) / f"{variant}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download with: kaggle datasets download "
            f"-d sgpjesus/bank-account-fraud-dataset-neurips-2022 "
            f"-p data/raw/baf --unzip"
        )
    df = pd.read_csv(csv_path)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

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


def preprocess_baf(df, train_idx=None, encoders=None):
    """
    Preprocess a BAF variant.

    Fit mode (encoders=None): requires train_idx; label-encoders are fitted
    on train rows only. Apply mode (encoders given): reuses encoders fitted
    on another dataframe (e.g., Base), so all variants share one encoding.
    Categories unseen at fit time map to "_missing_". The `month` column is
    the temporal axis, not a predictor, so it is excluded from features.

    Returns (X, y, encoders).
    """
    target = "fraud_bool"
    y = df[target].values
    X = df.drop(columns=[target, "month"]).copy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    fit_mode = encoders is None
    if fit_mode:
        if train_idx is None:
            raise ValueError("train_idx is required when fitting encoders")
        encoders = {}

    for col in cat_cols:
        X[col] = X[col].fillna("_missing_").astype(str)
        if fit_mode:
            train_categories = set(X[col].iloc[train_idx].unique())
            le = LabelEncoder()
            le.fit(pd.concat([X[col].iloc[train_idx], pd.Series(["_missing_"])]))
            encoders[col] = (le, train_categories)
        le, train_categories = encoders[col]
        X[col] = X[col].where(X[col].isin(train_categories), "_missing_")
        X[col] = le.transform(X[col])

    return X, y, encoders

def get_dataset(
    name: str, data_dir: str = "data/raw", return_meta: bool = False
):
    """
    Load raw data, compute temporal splits, then preprocess with
    statistics fitted on the train partition only.

    Args:
        return_meta: if True, additionally return a dict with per-row
            'sample_ids' and raw (unscaled) 'time_values' for the
            per-sample prediction store.

    Returns:
        (X, y, splits) or (X, y, splits, meta)
    """
    from src.data.splits import temporal_split_creditcard, temporal_split_ieee_cis

    from src.data.splits import temporal_split_baf

    if name == "ieee_cis":
        df = load_ieee_cis(f"{data_dir}/ieee-fraud-detection")
        splits = temporal_split_ieee_cis(df)
        meta = {
            "sample_ids": df["TransactionID"].values,
            "time_values": df["TransactionDT"].values.astype(float),
        }
        X, y, _, _ = preprocess_ieee_cis(df, train_idx=splits["train"])
    elif name == "creditcard":
        df = load_creditcard(f"{data_dir}/creditcardfraud")
        splits = temporal_split_creditcard(df)
        meta = {
            "sample_ids": df.index.values,
            "time_values": df["Time"].values.astype(float),
        }
        X, y = preprocess_creditcard(df, train_idx=splits["train"])
    elif name.startswith("baf"):
        # "baf" -> Base variant; "baf_variant_i" ... "baf_variant_v" for others.
        suffix = name[len("baf"):].strip("_")
        variant = "Base" if not suffix else "Variant " + suffix.split("_")[-1].upper()
        df = load_baf(variant, f"{data_dir}/baf")
        splits = temporal_split_baf(df)
        meta = {
            "sample_ids": df.index.values,
            "time_values": df["month"].values.astype(float),
        }
        X, y, _ = preprocess_baf(df, train_idx=splits["train"])
    else:
        raise ValueError(
            f"Unknown dataset: {name}. Use 'ieee_cis', 'creditcard', or 'baf'."
        )

    if return_meta:
        return X, y, splits, meta
    return X, y, splits