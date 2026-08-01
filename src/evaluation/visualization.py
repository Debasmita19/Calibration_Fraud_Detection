"""
Visualization utilities for calibration and fraud detection evaluation.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.calibration_metrics import reliability_diagram_data


def plot_reliability_diagram(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot a reliability (calibration) diagram."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    data = reliability_diagram_data(probs, y_true, n_bins)

    mask = data["bin_counts"] > 0
    ax.bar(
        data["bin_centers"][mask],
        data["bin_accuracies"][mask],
        width=1.0 / n_bins,
        alpha=0.6,
        edgecolor="black",
        label=label,
        color=color,
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    return ax


def plot_reliability_comparison(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    n_bins: int = 15,
    title: str = "Calibration Comparison",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot reliability diagrams for multiple calibration methods side by side.

    Args:
        results: Dict mapping method_name -> {"probs": np.ndarray}
        y_true: True labels
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for ax, (name, res), color in zip(axes, results.items(), colors):
        plot_reliability_diagram(
            res["probs"], y_true, n_bins, title=name, ax=ax, color=color
        )

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_calibration_over_time(
    window_metrics: List[Dict[str, float]],
    metric_name: str = "ece",
    method_names: Optional[List[str]] = None,
    title: str = "Calibration Degradation Over Time",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot how a calibration metric evolves across temporal windows.

    Args:
        window_metrics: List of dicts, one per time window.
            Each dict maps method_name -> metric_value.
        metric_name: Name of the metric being plotted
    """
    fig, ax = plt.subplots(figsize=figsize)

    if method_names is None:
        method_names = list(window_metrics[0].keys())

    x = np.arange(len(window_metrics))
    for name in method_names:
        values = [w.get(name, np.nan) for w in window_metrics]
        ax.plot(x, values, marker="o", label=name)

    ax.set_xlabel("Time Window")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_decision_distribution(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: Dict[str, float] = None,
    title: str = "Decision Zone Distribution",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Visualize how transactions are distributed across decision zones."""
    if thresholds is None:
        thresholds = {
            "auto_approve_max": 0.10,
            "step_up_auth_max": 0.50,
            "manual_review_max": 0.85,
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram of predicted probabilities by true label
    ax1.hist(probs[y_true == 0], bins=50, alpha=0.6, label="Legit", density=True)
    ax1.hist(probs[y_true == 1], bins=50, alpha=0.6, label="Fraud", density=True)

    for name, val in thresholds.items():
        ax1.axvline(x=val, color="red", linestyle="--", alpha=0.7)

    ax1.set_xlabel("Predicted P(Fraud)")
    ax1.set_ylabel("Density")
    ax1.set_title("Score Distribution by Class")
    ax1.legend()

    # Decision zone bar chart
    zones = ["Auto\nApprove", "Step-up\nAuth", "Manual\nReview", "Auto\nBlock"]
    t = thresholds
    masks = [
        probs < t["auto_approve_max"],
        (probs >= t["auto_approve_max"]) & (probs < t["step_up_auth_max"]),
        (probs >= t["step_up_auth_max"]) & (probs < t["manual_review_max"]),
        probs >= t["manual_review_max"],
    ]

    fraud_counts = [y_true[m].sum() for m in masks]
    legit_counts = [(1 - y_true)[m].sum() for m in masks]

    x = np.arange(len(zones))
    width = 0.35
    ax2.bar(x - width / 2, legit_counts, width, label="Legit", color="#4CAF50")
    ax2.bar(x + width / 2, fraud_counts, width, label="Fraud", color="#F44336")
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones)
    ax2.set_ylabel("Count")
    ax2.set_title("Transactions per Decision Zone")
    ax2.legend()

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_pareto_frontier(
    pareto_data: Dict[str, np.ndarray],
    title: str = "Fraud Loss vs Operational Cost",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot the Pareto frontier of fraud loss vs operational cost."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        pareto_data["operational_cost"],
        pareto_data["fraud_loss"],
        c=pareto_data["review_load"],
        cmap="viridis",
        alpha=0.5,
        s=20,
    )
    plt.colorbar(ax.collections[0], ax=ax, label="Review Load")

    ax.set_xlabel("Operational Cost ($)")
    ax.set_ylabel("Fraud Loss ($)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
