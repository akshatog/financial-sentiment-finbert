"""Training safeguards without downloading or claiming to train FinBERT."""
import json
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dataset import ROOT, frame_fingerprint
from src.finetuning import EncodedSentimentDataset, ID2LABEL, LABEL2ID, load_audited_splits, named_prediction_records, reusable_evaluation, validate_completed_manifest, validate_label_mapping


class ToyTokenizer:
    """Only exercises dataset policy; this is not the FinBERT tokenizer."""
    def __call__(self, texts, truncation=False, max_length=None, **kwargs):
        rows = [[1, *range(3, 3 + len(text.split())), 2] for text in texts]
        if truncation:
            rows = [row[:max_length - 1] + [2] if len(row) > max_length else row for row in rows]
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}


class FinetuningTests(unittest.TestCase):
    def test_named_labels_and_recorded_truncation(self):
        frame = pd.DataFrame({"text": ["one two three four five", "one"], "label": ["negative", "positive"]})
        dataset = EncodedSentimentDataset(frame, ToyTokenizer(), max_length=4)
        self.assertEqual(dataset.length_audit["rows_truncated"], 1)
        self.assertEqual(dataset.length_audit["max_original_tokens"], 7)
        self.assertEqual(dataset[0]["labels"], 0)
        self.assertEqual(dataset[1]["labels"], 2)
        self.assertEqual(len(dataset[0]["input_ids"]), 4)

    def test_label_permutation_is_rejected(self):
        validate_label_mapping(SimpleNamespace(id2label=ID2LABEL, label2id=LABEL2ID))
        with self.assertRaisesRegex(ValueError, "mapping"):
            validate_label_mapping(SimpleNamespace(id2label={0: "positive", 1: "negative", 2: "neutral"}, label2id=LABEL2ID))
        rows = named_prediction_records(np.asarray([[0.0, -1.0, 3.0]]))
        self.assertEqual(rows[0]["predicted"], "positive")
        self.assertAlmostEqual(sum(rows[0][label] for label in LABEL2ID), 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            named_prediction_records(np.asarray([[float("nan"), 1, 2]]))

    def test_training_can_read_train_validation_without_test_file(self):
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            partitions = {}
            for name in ("train", "validation"):
                frame = pd.DataFrame([{"id": f"{name}_{label}", "text": f"{name} unique {label}", "label": label} for label in LABEL2ID])
                frame.to_csv(root / f"{name}.csv", index=False)
                partitions[name] = {"rows": len(frame), "fingerprint": frame_fingerprint(frame)}
            audit = root / "audit.json"
            audit.write_text(json.dumps({"splits": {"partitions": partitions}}), encoding="utf-8")
            actual = load_audited_splits(root, audit, ("train", "validation"))
            self.assertEqual(set(actual), {"train", "validation"})
            self.assertFalse((root / "test.csv").exists())
            tampered = pd.read_csv(root / "train.csv")
            tampered.loc[0, "label"] = "positive"
            tampered.to_csv(root / "train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "audit"):
                load_audited_splits(root, audit, ("train", "validation"))

    def test_evaluation_reuse_requires_same_frozen_model_data_and_policy(self):
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "evaluation.json"
            result = {"test_fingerprint": "test", "inference_file_sha256": {"model.safetensors": "model"}, "training_manifest_sha256": "manifest", "max_length": 128}
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(reusable_evaluation(path, "test", {"model.safetensors": "model"}, "manifest", 128), result)
            for values in [("changed", {"model.safetensors": "model"}, "manifest", 128), ("test", {"model.safetensors": "changed"}, "manifest", 128), ("test", {"model.safetensors": "model"}, "manifest", 256)]:
                with self.assertRaisesRegex(ValueError, "different"):
                    reusable_evaluation(path, *values)
            (Path(directory) / "training_manifest.json").write_text(json.dumps({"status": "smoke_completed", "smoke_test": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke"):
                validate_completed_manifest(Path(directory))

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "Separate training dependencies are optional for the app test suite")
    def test_tiny_local_fixture_trains_saves_and_evaluates_without_downloads(self):
        # A randomly initialized miniature BERT verifies the real Trainer API,
        # checkpoint selection, export, and evaluation. It is never FinBERT
        # training evidence and all of its files stay in ignored temporary data.
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        from scripts import train_finbert, evaluate_finetuned
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with TemporaryDirectory(dir=temp_root) as directory:
                root = Path(directory)
                fixture = root / "random_tiny_bert_fixture"
                fixture.mkdir()
                vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "train", "validation", "test", "negative", "neutral", "positive", "one", "two", "company"]
                (fixture / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
                BertTokenizerFast(vocab_file=str(fixture / "vocab.txt"), do_lower_case=True).save_pretrained(fixture)
                BertForSequenceClassification(BertConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24, max_position_embeddings=128, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL)).save_pretrained(fixture, safe_serialization=True)
                splits_dir = root / "splits"
                splits_dir.mkdir()
                partitions = {}
                for name in ("train", "validation", "test"):
                    frame = pd.DataFrame([{"id": f"{name}_{label}_{number}", "text": f"{name} company {label} {number}", "label": label} for label in LABEL2ID for number in ("one", "two")])
                    frame.to_csv(splits_dir / f"{name}.csv", index=False)
                    partitions[name] = {"rows": len(frame), "fingerprint": frame_fingerprint(frame)}
                audit = root / "audit.json"
                audit.write_text(json.dumps({"splits": {"partitions": partitions}}), encoding="utf-8")
                output = root / "model"
                common = ["--splits-dir", str(splits_dir), "--audit-path", str(audit)]
                with patch.object(train_finbert, "PRETRAINED_MODEL_ID", str(fixture)), patch("sys.argv", ["train_finbert.py", *common, "--output-dir", str(output), "--epochs", "1", "--batch-size", "2", "--gradient-accumulation", "1", "--max-length", "32", "--allow-cpu"]):
                    train_finbert.main()
                manifest, _ = validate_completed_manifest(output)
                self.assertEqual(manifest["source_split_rows"], {"train": 6, "validation": 6})
                self.assertTrue(manifest["encoder_update_evidence"]["changed"])
                self.assertEqual(manifest["parameter_counts"]["trainable"], manifest["parameter_counts"]["total"])
                results_dir = root / "results"
                arguments = ["evaluate_finetuned.py", *common, "--model-dir", str(output), "--output-dir", str(results_dir), "--device", "cpu"]
                with patch("sys.argv", arguments):
                    evaluate_finetuned.main()
                    evaluate_finetuned.main()
                result = json.loads((results_dir / "evaluation.json").read_text(encoding="utf-8"))
                self.assertEqual(result["test_rows"], 6)
                self.assertEqual(result["test_evaluation_runs"], 1)
                self.assertFalse(result["provenance"]["test_used_for_training_or_selection"])
                self.assertEqual(result["metrics"]["confusion_matrix"]["labels"], list(LABEL2ID))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
