# Confidence-Calibrated and Deployment-Safe Fraud Detection under Distribution Shift

## Overview

This project investigates how to build fraud detection models that produce **reliable confidence estimates** and use those estimates in **risk-aware decision policies**, even when fraud patterns evolve over time (distribution shift).

### Key Research Questions

1. **Calibration under shift** — How do standard fraud models (XGBoost, deep tabular, GNNs) behave in terms of calibration when fraud patterns shift temporally?
2. **Calibration methods** — Which post-hoc calibration methods work best under distribution shift (not just i.i.d.)?
3. **Risk-aware decision policies** — How can calibrated confidence drive tiered decisions (auto-approve / step-up auth / manual review / auto-block)?
4. **Robustness** — How robust are calibration methods to novel attack patterns? Can we design monitoring + recalibration strategies?
5. **Novel contribution** — Can a fraud-specific calibration + decision framework outperform generic methods on both calibration and business metrics?

## Project Structure

```
fraud_detection_calibration/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml              # Global experiment configuration
├── data/
│   ├── raw/                     # Original downloaded datasets
│   ├── processed/               # Cleaned, split datasets
│   └── synthetic/               # Synthetically shifted data
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_baselines.ipynb       # Baseline model training
│   ├── 03_calibration.ipynb     # Calibration experiments
│   ├── 04_shift.ipynb           # Distribution shift analysis
│   └── 05_decisions.ipynb       # Decision policy evaluation
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Dataset loading & preprocessing
│   │   └── splits.py            # Temporal & domain split logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── baselines.py         # XGBoost, LightGBM, MLP, etc.
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── post_hoc.py          # Temperature, isotonic, beta, Dirichlet
│   │   ├── conformal.py         # Conformal prediction wrappers
│   │   └── ensemble.py          # MC Dropout, deep ensembles
│   ├── shifts/
│   │   ├── __init__.py
│   │   └── shift_generator.py   # Synthetic shift creation
│   ├── policies/
│   │   ├── __init__.py
│   │   └── decision.py          # Risk-aware decision policies
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── calibration_metrics.py  # ECE, MCE, Brier, reliability
│   │   ├── business_metrics.py     # Fraud loss, review cost, friction
│   │   └── visualization.py        # Reliability diagrams, plots
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── scripts/
│   ├── download_data.py         # Download datasets
│   └── run_experiment.py        # Full pipeline runner
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
└── tests/
```

## Datasets

| Dataset | Source | Features | Key Properties |
|---------|--------|----------|----------------|
| IEEE-CIS Fraud Detection | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | ~400 anonymized | Large, temporal, severe imbalance |
| Credit Card Fraud (European) | [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) | 30 (PCA) | Simpler benchmark, highly imbalanced |

## Core References

1. Guo et al. (2017) — [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
2. Kull et al. (2019) — [Beyond temperature scaling: Dirichlet calibration](https://arxiv.org/abs/1906.02530)
3. Lakshminarayanan et al. (2017) — [Simple and Scalable Predictive Uncertainty Estimation](https://arxiv.org/abs/1612.01474)
4. Cost-aware conformal prediction for fraud — [arXiv:2507.07780](https://arxiv.org/abs/2507.07780)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires Kaggle API key)
python scripts/download_data.py
```
