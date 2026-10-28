"""Named-label evaluation and reusable presentation figures."""
from __future__ import annotations

from pathlib import Path

from src.config import setup_cache

setup_cache()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.dataset import LABELS

COLORS = {"negative": "#da6c67", "neutral": "#8192ae", "positive": "#3eac91"}


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Expected equally sized, nonempty true and predicted label lists")
    if (set(y_true) | set(y_pred)) - set(LABELS):
        raise ValueError("Evaluation requires named sentiment labels")
    report = classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0)
    for row in report.values():
        if isinstance(row, dict) and "support" in row:
            row["support"] = int(row["support"])
    return {
        "n_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0)),
        "classification_report": report,
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": {"labels": list(LABELS), "rows": "true", "columns": "predicted", "counts": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()},
    }


def plot_confusion(metrics: dict, path: str | Path, title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"]["counts"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    labels = list(LABELS)
    ax.set(xticks=np.arange(3), yticks=np.arange(3), xticklabels=labels, yticklabels=labels, xlabel="Predicted sentiment", ylabel="True sentiment", title=title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.55 else "#162840", fontsize=14, fontweight="bold")
    _save(fig, path)


def plot_distribution(counts: dict[str, int], path: str | Path, title: str = "Financial PhraseBank: cleaned class distribution") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    values = [counts[label] for label in LABELS]
    bars = ax.bar(list(LABELS), values, color=[COLORS[label] for label in LABELS], width=0.6)
    total = sum(values)
    ax.bar_label(bars, labels=[f"{count:,}\n({count / total:.1%})" for count in values], padding=5)
    ax.set(title=title, ylabel="Sentences", ylim=(0, max(values) * 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_model_comparison(baseline: dict, majority: dict, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    positions = np.arange(2)
    for offset, (label, result, color) in zip((-0.18, 0.18), (("TF-IDF + logistic regression", baseline, "#387ba6"), ("Majority class", majority, "#9ca7b6"))):
        bars = ax.bar(positions + offset, [result["accuracy"], result["macro_f1"]], width=0.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(xticks=positions, xticklabels=["Accuracy", "Macro F1"], ylim=(0, 1.16), title="Held-out PhraseBank test: independently trained baselines", ylabel="Score")
    ax.legend(loc="upper center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def _save(fig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
