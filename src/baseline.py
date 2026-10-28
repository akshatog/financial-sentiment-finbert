"""Training and inference for the independently trained TF-IDF baseline."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.dataset import LABELS, ROOT, assert_disjoint_splits, frame_fingerprint

DEFAULT_MODEL_PATH = ROOT / "models" / "tfidf_logreg.joblib"


def make_pipeline(c_value: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=seed, solver="lbfgs")),
    ])


def train_baseline(train: pd.DataFrame, validation: pd.DataFrame, c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0), seed: int = 42) -> tuple[Pipeline, dict]:
    """Fit vocabulary and weights on train only; select C on validation macro F1.

    The selected model remains fitted on train only. There is no test argument.
    """
    assert_disjoint_splits({"train": train, "validation": validation})
    for name, frame in {"train": train, "validation": validation}.items():
        if set(frame["label"]) != set(LABELS):
            raise ValueError(f"{name} must include all three sentiment classes")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C candidates must be positive")
    trials = []
    best_model, best_score, best_c = None, -1.0, None
    for c_value in sorted(set(c_values)):
        candidate = make_pipeline(float(c_value), seed)
        candidate.fit(train["text"].tolist(), train["label"].tolist())
        predicted = candidate.predict(validation["text"].tolist())
        score = float(f1_score(validation["label"], predicted, labels=list(LABELS), average="macro", zero_division=0))
        trials.append({"C": float(c_value), "validation_macro_f1": score, "validation_accuracy": float(accuracy_score(validation["label"], predicted))})
        if score > best_score:
            best_model, best_score, best_c = candidate, score, float(c_value)
    metadata = {
        "schema_version": 1,
        "model_type": "TF-IDF (word unigrams/bigrams) + class-balanced logistic regression",
        "labels": list(LABELS),
        "seed": seed,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_class_counts": {label: int((train["label"] == label).sum()) for label in LABELS},
        "majority_label": str(train["label"].value_counts().idxmax()),
        "train_fingerprint": frame_fingerprint(train),
        "validation_fingerprint": frame_fingerprint(validation),
        "selection_metric": "validation macro F1; smaller C wins exact ties",
        "selected_C": best_c,
        "validation_macro_f1": best_score,
        "candidates": trials,
        "fit_data": "Training partition only, including TF-IDF vocabulary and IDF weights; no refit on validation or test",
        "test_used_for_training_or_selection": False,
        "feature_count": int(len(best_model.named_steps["tfidf"].vocabulary_)),
        "parameters": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.98, "sublinear_tf": True, "class_weight": "balanced", "max_iter": 2000},
    }
    return best_model, metadata


class BaselinePredictor:
    """Load a saved pipeline and return named-label model scores."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Baseline model not found at {self.model_path}. Run python scripts/train_baseline.py first.")
        # Only load project-created trusted joblib files: pickle is executable.
        self.model = joblib.load(self.model_path)
        self.classes = [str(label) for label in self.model.classes_]
        if set(self.classes) != set(LABELS):
            raise ValueError(f"Baseline must use named classes {LABELS}; found {self.classes}")

    def predict(self, texts: list[str]) -> list[dict]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every input must be a nonblank string")
        if not texts:
            return []
        scores = self.model.predict_proba(texts)
        results = []
        for row in scores:
            named_scores = {label: float(row[self.classes.index(label)]) for label in LABELS}
            winner = self.classes[int(np.argmax(row))]
            results.append({"label": winner, "score": named_scores[winner], **named_scores})
        return results
