# Confidence-Calibrated and Deployment-Safe Fraud Detection under Distribution Shift

## Overview

This project investigates how to build fraud detection models that produce **reliable confidence estimates** and use those estimates in **risk-aware decision policies**, even when fraud patterns evolve over time (distribution shift).

### Key Research Questions

1. **Calibration under shift** — How do standard fraud models (XGBoost, deep tabular, GNNs) behave in terms of calibration when fraud patterns shift temporally?
2. **Calibration methods** — Which post-hoc calibration methods work best under distribution shift (not just i.i.d.)?
3. **Risk-aware decision policies** — How can calibrated confidence drive tiered decisions (auto-approve / step-up auth / manual review / auto-block)?
4. **Robustness** — How robust are calibration methods to novel attack patterns? Can we design monitoring + recalibration strategies?
5. **Novel contribution** — Can a fraud-specific calibration + decision framework outperform generic methods on both calibration and business metrics?

---

## Research Phases

### Phase 1 — Data Acquisition & Exploratory Analysis (Week 1–2)
- Download IEEE-CIS Fraud Detection and Credit Card Fraud Detection datasets
- Perform EDA: class distributions, feature correlations, temporal patterns
- Create temporal train/val/test splits to simulate deployment over time
- Create synthetic distribution shifts (region, channel, new fraud modes)
- **Deliverable**: EDA notebook, processed datasets with temporal splits

### Phase 2 — Baseline Models & Uncalibrated Evaluation (Week 2–3)
- Train baseline models: XGBoost, LightGBM, MLP (deep tabular), optional TabNet
- Evaluate: AUC-ROC, AUC-PR, F1, precision, recall
- Evaluate calibration: ECE, MCE, Brier score, reliability diagrams
- Compare i.i.d. vs temporal-split performance
- **Deliverable**: Baseline results table, reliability diagrams showing miscalibration

### Phase 3 — Calibration Methods (Week 3–5)
- Implement post-hoc calibration: Platt/temperature scaling, isotonic regression, beta calibration, Dirichlet calibration
- Implement uncertainty-aware methods: MC Dropout, deep ensembles (if using neural models)
- Implement conformal prediction wrappers for coverage guarantees
- Evaluate all methods on calibration metrics under i.i.d. and shifted settings
- **Deliverable**: Calibration comparison tables and reliability diagrams

### Phase 4 — Distribution Shift Experiments (Week 5–7)
- Temporal shift: train on months 1–4, calibrate on month 5, test on months 6+
- Domain shift: train on one card type/region, test on another
- Synthetic shift: inject new fraud modes into test set
- Measure calibration degradation curves over time
- Test recalibration strategies with limited labeled data
- **Deliverable**: Shift degradation plots, recalibration effectiveness analysis

### Phase 5 — Risk-Aware Decision Policies (Week 7–8)
- Design cost-sensitive decision framework with configurable thresholds
- Implement tiered policies: auto-approve / step-up auth / manual review / auto-block
- Define business metrics: expected fraud loss, review cost, customer friction rate
- Optimize thresholds using calibrated probabilities
- Compare: uncalibrated vs calibrated policy performance
- **Deliverable**: Decision policy comparison, Pareto frontiers of fraud-loss vs review-cost

### Phase 6 — Novel Framework & Final Evaluation (Week 8–10)
- Design fraud-specific contribution: e.g., cost-aware conformal wrapper, risk-adaptive recalibration
- Comprehensive evaluation on both calibration metrics and business metrics
- Ablation studies
- **Deliverable**: Final results, paper draft, visualizations

---

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
