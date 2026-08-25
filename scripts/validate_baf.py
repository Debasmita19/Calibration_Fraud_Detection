"""One-shot validation of the BAF pipeline: load, split, preprocess, report."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import get_dataset

t0 = time.time()
X, y, splits, meta = get_dataset("baf", return_meta=True)
t1 = time.time()

print(f"load+preprocess (cold): {t1 - t0:.1f}s")
print(f"X: {X.shape}, overall fraud rate: {y.mean():.4f}")
print(f"months present: {sorted(set(meta['time_values']))}")
for name, idx in splits.items():
    months = sorted(set(meta["time_values"][idx]))
    print(f"  {name:<12} n={len(idx):>7}  fraud={y[idx].mean():.4f}  months={months}")

t2 = time.time()
get_dataset("baf")
t3 = time.time()
print(f"warm reload (parquet cache): {t3 - t2:.1f}s")
