"""Verify trained vocabulary isolation and named probability mapping."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from src.baseline import BaselinePredictor, train_baseline
from src.dataset import ROOT


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        phrases = {"positive": "profits grow gains strong", "negative": "losses deepen decline weak", "neutral": "meeting scheduled report unchanged"}
        cls.train = pd.DataFrame([{"id": f"train_{label}_{i}", "text": f"{phrase} company trainingnumber{i}", "label": label} for label, phrase in phrases.items() for i in range(12)])
        cls.validation = pd.DataFrame([{"id": f"validation_{label}_{i}", "text": f"{phrase} company validationonlytoken number{i}", "label": label} for label, phrase in phrases.items() for i in range(3)])
        cls.model, cls.metadata = train_baseline(cls.train, cls.validation, c_values=(0.5, 1.0))

    def test_vocabulary_is_fitted_only_on_training_partition(self):
        vocabulary = self.model.named_steps["tfidf"].vocabulary_
        self.assertNotIn("validationonlytoken", vocabulary)
        self.assertTrue(self.metadata["test_used_for_training_or_selection"] is False)
        self.assertEqual(self.metadata["training_rows"], len(self.train))
        self.assertEqual(len(self.metadata["candidates"]), 2)

    def test_saved_prediction_preserves_named_probability_columns(self):
        texts = ["profits grow strong gains", "losses deepen weak decline", "meeting report scheduled"]
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "baseline.joblib"
            joblib.dump(self.model, path)
            predictor = BaselinePredictor(path)
            actual = predictor.predict(texts)
            expected = self.model.predict_proba(texts)
            for row, original in zip(actual, expected):
                self.assertAlmostEqual(sum(row[label] for label in ("positive", "neutral", "negative")), 1.0)
                for index, label in enumerate(self.model.classes_):
                    self.assertAlmostEqual(row[label], original[index])
                self.assertEqual(row["label"], max(("positive", "neutral", "negative"), key=lambda label: row[label]))
                self.assertEqual(row["score"], row[row["label"]])
            self.assertEqual(predictor.predict([]), [])
            with self.assertRaises(ValueError):
                predictor.predict(["valid", "   "])
            with self.assertRaises(ValueError):
                predictor.predict([None])

    def test_training_rejects_leaking_validation_text(self):
        contaminated = self.validation.copy()
        contaminated.loc[0, "text"] = self.train.loc[0, "text"].upper()
        with self.assertRaisesRegex(ValueError, "overlap"):
            train_baseline(self.train, contaminated, c_values=(1.0,))


if __name__ == "__main__":
    unittest.main()
