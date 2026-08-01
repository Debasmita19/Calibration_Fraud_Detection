"""
Full experimental pipeline runner.

Usage:
    python scripts/run_experiment.py --config configs/config.yaml --dataset creditcard
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, set_seed, get_class_weight
from src.data.loader import get_dataset
from src.data.splits import temporal_split_creditcard
from src.models.baselines import get_model
from src.calibration.post_hoc import get_calibrator, CALIBRATORS
from src.calibration.conformal import SplitConformalClassifier, CostAwareConformal
from src.evaluation.calibration_metrics import compute_all_calibration_metrics
from src.evaluation.business_metrics import compute_business_metrics
from src.evaluation.visualization import (
    plot_reliability_diagram,
    plot_decision_distribution,
)
from src.shifts.shift_generator import create_shift_scenarios
from src.policies.decision import (
    compare_calibrated_vs_uncalibrated_policies,
)


def run_pipeline(config_path: str, dataset_name: str = "creditcard"):
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])

    results_dir = Path(config["experiment"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    print("=" * 60)
    print(f"Loading dataset: {dataset_name}")
    print("=" * 60)

    X, y, splits = get_dataset(dataset_name)
    print(f"Dataset shape: {X.shape}, Fraud rate: {y.mean():.4f}")

  
    X_train = X.iloc[splits["train"]].values if hasattr(X, "iloc") else X[splits["train"]]
    y_train = y[splits["train"]]
    X_cal = X.iloc[splits["calibration"]].values if hasattr(X, "iloc") else X[splits["calibration"]]
    y_cal = y[splits["calibration"]]
    X_val = X.iloc[splits["val"]].values if hasattr(X, "iloc") else X[splits["val"]]
    y_val = y[splits["val"]]
    X_test = X.iloc[splits["test"]].values if hasattr(X, "iloc") else X[splits["test"]]
    y_test = y[splits["test"]]

    print(f"Train: {len(y_train)}, Cal: {len(y_cal)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ---- Step 3: Train baseline models ----
    all_results = {}

    for model_name in ["xgboost", "lightgbm"]:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        model_config = config["models"].get(model_name, {})
        model = get_model(model_name, model_config)
        model.fit(X_train, y_train, X_val, y_val)

        # Raw (uncalibrated) predictions
        raw_probs_test = model.predict_proba(X_test)
        raw_probs_cal = model.predict_proba(X_cal)

        raw_metrics = compute_all_calibration_metrics(raw_probs_test, y_test)
        raw_clf_metrics = model.evaluate(X_test, y_test)
        print(f"\n[{model_name}] Uncalibrated:")
        print(f"  AUROC={raw_clf_metrics['auroc']:.4f}, AUPRC={raw_clf_metrics['auprc']:.4f}")
        print(f"  ECE={raw_metrics['ece']:.4f}, Brier={raw_metrics['brier_score']:.4f}")

        model_results = {"uncalibrated": {**raw_metrics, **raw_clf_metrics}}

        # ---- Step 4: Calibrate ----
        for cal_name in config["calibration"]["methods"]:
            print(f"\n  Calibrating with: {cal_name}")
            try:
                calibrator = get_calibrator(cal_name)
                calibrator.fit(raw_probs_cal, y_cal)
                cal_probs = calibrator.calibrate(raw_probs_test)

                cal_metrics = compute_all_calibration_metrics(cal_probs, y_test)
                print(f"    ECE={cal_metrics['ece']:.4f}, Brier={cal_metrics['brier_score']:.4f}")

                model_results[cal_name] = cal_metrics
            except Exception as e:
                print(f"    Error: {e}")
                model_results[cal_name] = {"error": str(e)}

        # ---- Step 5: Conformal prediction ----
        print(f"\n  Conformal prediction (alpha={config['calibration']['conformal_alpha']}):")
        conformal = SplitConformalClassifier(alpha=config["calibration"]["conformal_alpha"])
        conformal.fit(raw_probs_cal, y_cal)
        conformal_eval = conformal.evaluate_coverage(raw_probs_test, y_test)
        print(f"    Coverage={conformal_eval['empirical_coverage']:.4f}, "
              f"Avg set size={conformal_eval['avg_set_size']:.3f}")
        model_results["conformal"] = conformal_eval

        cost_conformal = CostAwareConformal(
            alpha=config["calibration"]["conformal_alpha"],
            fn_cost_weight=10.0,
            fp_cost_weight=1.0,
        )
        cost_conformal.fit(raw_probs_cal, y_cal)
        cost_conformal_eval = cost_conformal.evaluate_coverage(raw_probs_test, y_test)
        print(f"    Cost-aware coverage={cost_conformal_eval['empirical_coverage']:.4f}")
        model_results["cost_aware_conformal"] = cost_conformal_eval

        # ---- Step 6: Business metrics comparison ----
        print(f"\n  Decision policy comparison:")
        best_cal_name = min(
            [k for k in model_results if k not in ["uncalibrated", "conformal", "cost_aware_conformal"]
             and isinstance(model_results.get(k), dict) and "ece" in model_results.get(k, {})],
            key=lambda k: model_results[k]["ece"],
            default=None,
        )

        if best_cal_name:
            best_calibrator = get_calibrator(best_cal_name)
            best_calibrator.fit(raw_probs_cal, y_cal)
            best_cal_probs = best_calibrator.calibrate(raw_probs_test)

            policy_comparison = compare_calibrated_vs_uncalibrated_policies(
                raw_probs_test, best_cal_probs, y_test, costs=config["evaluation"]["costs"]
            )
            print(f"    Best calibrator: {best_cal_name}")
            print(f"    Cost reduction: {policy_comparison['cost_reduction_pct']:.2f}%")
            model_results["policy_comparison"] = {
                "best_calibrator": best_cal_name,
                **{k: v for k, v in policy_comparison.items()
                   if not isinstance(v, dict)},
            }

        all_results[model_name] = model_results

    # ---- Save results ----
    output_path = results_dir / f"results_{dataset_name}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default="creditcard", choices=["creditcard", "ieee_cis"])
    args = parser.parse_args()
    run_pipeline(args.config, args.dataset)
