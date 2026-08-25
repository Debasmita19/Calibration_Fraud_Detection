"""
Full experimental pipeline runner with multi-seed support.

For each seed: train base models, save per-sample predictions to the
prediction store, evaluate uncalibrated/calibrated/conformal metrics and the
decision-policy comparison. Results are saved per-seed and aggregated
(mean/std across seeds) so every reported number carries error bars.

Usage:
    python scripts/run_experiment.py --config configs/config.yaml --dataset creditcard
    python scripts/run_experiment.py --dataset ieee_cis --seeds 42 43 44 45 46
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, set_seed
from src.utils.prediction_store import save_predictions
from src.data.loader import get_dataset
from src.models.baselines import get_model
from src.calibration.post_hoc import get_calibrator
from src.calibration.conformal import SplitConformalClassifier, CostAwareConformal
from src.evaluation.calibration_metrics import compute_all_calibration_metrics
from src.policies.decision import compare_calibrated_vs_uncalibrated_policies


def _slice(X, idx):
    return X.iloc[idx].values if hasattr(X, "iloc") else X[idx]


def run_single_seed(config, dataset_name, seed, X, y, splits, meta, model_names):
    """Train, calibrate, and evaluate all models for one seed."""
    set_seed(seed)

    X_train, y_train = _slice(X, splits["train"]), y[splits["train"]]
    X_cal, y_cal = _slice(X, splits["calibration"]), y[splits["calibration"]]
    X_val, y_val = _slice(X, splits["val"]), y[splits["val"]]
    X_test, y_test = _slice(X, splits["test"]), y[splits["test"]]

    seed_results = {}

    for model_name in model_names:
        print(f"\n{'='*60}\n[seed {seed}] Training: {model_name}\n{'='*60}")

        model_config = dict(config["models"].get(model_name, {}))
        # GBDTs are deterministic without an explicit random_state; inject the
        # run seed so subsampling produces genuine seed-to-seed variance.
        if model_name in ("xgboost", "lightgbm"):
            model_config["random_state"] = seed

        model = get_model(model_name, model_config)
        model.fit(X_train, y_train, X_val, y_val)

        # Predict once for ALL rows; downstream methods reuse these frozen
        # predictions from the store instead of refitting the model.
        all_probs = model.predict_proba(X.values if hasattr(X, "values") else X)
        store_path = save_predictions(
            dataset=dataset_name,
            model=model_name,
            seed=seed,
            sample_ids=meta["sample_ids"],
            time_values=meta["time_values"],
            y_true=y,
            y_prob=all_probs,
            splits=splits,
        )
        print(f"  Predictions saved to {store_path}")

        raw_probs_cal = all_probs[splits["calibration"]]
        raw_probs_test = all_probs[splits["test"]]

        raw_metrics = compute_all_calibration_metrics(raw_probs_test, y_test)
        raw_clf_metrics = model.evaluate(X_test, y_test)
        print(f"  Uncalibrated: AUROC={raw_clf_metrics['auroc']:.4f}, "
              f"AUPRC={raw_clf_metrics['auprc']:.4f}, ECE={raw_metrics['ece']:.4f}, "
              f"adaptive ECE={raw_metrics['adaptive_ece']:.4f}")

        model_results = {"uncalibrated": {**raw_metrics, **raw_clf_metrics}}

        # ---- Post-hoc calibration ----
        for cal_name in config["calibration"]["methods"]:
            try:
                calibrator = get_calibrator(cal_name)
                calibrator.fit(raw_probs_cal, y_cal)
                cal_probs = calibrator.calibrate(raw_probs_test)
                model_results[cal_name] = compute_all_calibration_metrics(
                    cal_probs, y_test
                )
            except Exception as e:
                print(f"  {cal_name} failed: {e}")
                model_results[cal_name] = {"error": str(e)}

        # ---- Conformal prediction ----
        alpha = config["calibration"]["conformal_alpha"]
        conformal = SplitConformalClassifier(alpha=alpha)
        conformal.fit(raw_probs_cal, y_cal)
        model_results["conformal"] = conformal.evaluate_coverage(
            raw_probs_test, y_test
        )

        cost_conformal = CostAwareConformal(
            alpha=alpha, fn_cost_weight=10.0, fp_cost_weight=1.0
        )
        cost_conformal.fit(raw_probs_cal, y_cal)
        model_results["cost_aware_conformal"] = cost_conformal.evaluate_coverage(
            raw_probs_test, y_test
        )

        # ---- Decision policy comparison (best-ECE calibrator vs raw) ----
        cal_candidates = [
            k for k in config["calibration"]["methods"]
            if "ece" in model_results.get(k, {})
        ]
        if cal_candidates:
            best_cal_name = min(
                cal_candidates, key=lambda k: model_results[k]["ece"]
            )
            best_calibrator = get_calibrator(best_cal_name)
            best_calibrator.fit(raw_probs_cal, y_cal)
            best_cal_probs = best_calibrator.calibrate(raw_probs_test)

            policy_comparison = compare_calibrated_vs_uncalibrated_policies(
                raw_probs_test, best_cal_probs, y_test,
                costs=config["evaluation"]["costs"],
            )
            model_results["policy_comparison"] = {
                "best_calibrator": best_cal_name,
                **{k: v for k, v in policy_comparison.items()
                   if not isinstance(v, dict)},
            }

        seed_results[model_name] = model_results

    return seed_results


def aggregate_across_seeds(per_seed_results):
    """
    Recursively aggregate a list of identically-structured nested dicts.
    Numeric leaves become {mean, std, n_seeds, values}; non-numeric leaves
    are kept as a list when they differ across seeds.
    """
    def _aggregate(dicts):
        out = {}
        for key in dicts[0]:
            vals = [d[key] for d in dicts if key in d]
            if all(isinstance(v, dict) for v in vals):
                out[key] = _aggregate(vals)
            elif all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in vals
            ):
                arr = np.asarray(vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                out[key] = {
                    "mean": float(finite.mean()) if len(finite) else None,
                    "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                    "n_seeds": len(arr),
                    "values": [float(v) for v in arr],
                }
            else:
                unique = {str(v) for v in vals}
                out[key] = vals[0] if len(unique) == 1 else vals
        return out

    return _aggregate(per_seed_results)


def run_pipeline(config_path, dataset_name="creditcard", seeds=None,
                 model_names=("xgboost", "lightgbm")):
    config = load_config(config_path)
    if seeds is None:
        seeds = config["experiment"].get("seeds", [config["experiment"]["seed"]])

    results_dir = Path(config["experiment"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Dataset: {dataset_name} | Seeds: {seeds} | Models: {list(model_names)}")
    print("=" * 60)

    # Data and splits are deterministic; load once, reuse across seeds.
    X, y, splits, meta = get_dataset(dataset_name, return_meta=True)
    print(f"Dataset shape: {X.shape}, Fraud rate: {y.mean():.4f}")
    print(f"Train: {len(splits['train'])}, Cal: {len(splits['calibration'])}, "
          f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    per_seed = {}
    for seed in seeds:
        per_seed[seed] = run_single_seed(
            config, dataset_name, seed, X, y, splits, meta, model_names
        )

    aggregated = aggregate_across_seeds(list(per_seed.values()))

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    per_seed_path = results_dir / f"results_{dataset_name}_per_seed.json"
    with open(per_seed_path, "w") as f:
        json.dump({str(s): r for s, r in per_seed.items()}, f, indent=2,
                  default=convert)

    agg_path = results_dir / f"results_{dataset_name}.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=convert)

    print(f"\n{'='*60}")
    print(f"Per-seed results: {per_seed_path}")
    print(f"Aggregated (mean/std over {len(seeds)} seeds): {agg_path}")
    print(f"{'='*60}")

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default="creditcard",
                        choices=["creditcard", "ieee_cis", "baf",
                                 "baf_variant_i", "baf_variant_ii",
                                 "baf_variant_iii", "baf_variant_iv",
                                 "baf_variant_v"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: experiment.seeds from config)")
    parser.add_argument("--models", nargs="+", default=["xgboost", "lightgbm"],
                        choices=["xgboost", "lightgbm", "mlp"])
    args = parser.parse_args()
    run_pipeline(args.config, args.dataset, seeds=args.seeds,
                 model_names=args.models)
