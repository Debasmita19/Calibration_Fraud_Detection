"""Print calibrator comparison (ECE mean ± std) from aggregated results."""
import json

for name in ["ieee_cis", "baf"]:
    d = json.load(open(f"results/results_{name}.json"))["xgboost"]
    print(f"\n{name}")
    for method, v in d.items():
        if isinstance(v, dict) and "ece" in v:
            e = v["ece"]
            print(f"  {method:<22} ECE {e['mean']:.5f} ± {e['std']:.5f}")
            for extra in ["mce", "nll", "brier_score"]:
                if extra in v:
                    print(f"    {extra:<12} {v[extra]['mean']:.5f} ± {v[extra]['std']:.5f}")

