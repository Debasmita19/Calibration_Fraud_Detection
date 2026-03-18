"""
Download fraud detection datasets from Kaggle.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Place your kaggle.json API key in ~/.kaggle/kaggle.json
       (Get it from https://www.kaggle.com/settings -> API -> Create New Token)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_ieee_cis(output_dir: Path):
    """Download IEEE-CIS Fraud Detection dataset from Kaggle competition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IEEE-CIS Fraud Detection to {output_dir}...")
    os.system(
        f'kaggle competitions download -c ieee-fraud-detection -p "{output_dir}"'
    )
    # Unzip
    import zipfile
    zip_path = output_dir / "ieee-fraud-detection.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        zip_path.unlink()
        print("IEEE-CIS dataset extracted.")
    else:
        print("Warning: ZIP file not found. Check Kaggle API credentials.")


def download_creditcard(output_dir: Path):
    """Download European Credit Card Fraud dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Credit Card Fraud dataset to {output_dir}...")
    os.system(
        f'kaggle datasets download -d mlg-ulb/creditcardfraud -p "{output_dir}"'
    )
    import zipfile
    zip_path = output_dir / "creditcardfraud.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        zip_path.unlink()
        print("Credit Card Fraud dataset extracted.")
    else:
        print("Warning: ZIP file not found. Check Kaggle API credentials.")


if __name__ == "__main__":
    raw_dir = PROJECT_ROOT / "data" / "raw"

    print("=" * 60)
    print("Fraud Detection Dataset Downloader")
    print("=" * 60)

    download_ieee_cis(raw_dir / "ieee-fraud-detection")
    print()
    download_creditcard(raw_dir / "creditcardfraud")

    print("\nDone! Datasets saved to:", raw_dir)
