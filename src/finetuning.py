"""Shared safeguards for supervised FinBERT training and frozen evaluation.

These helpers do not import Trainer or require a GPU. Training reads only the
saved training and validation partitions; the separate evaluator reads test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import LABELS, assert_disjoint_splits, file_sha256, frame_fingerprint

PRETRAINED_MODEL_ID = "yiyanghkust/finbert-pretrain"
PRETRAINED_REVISION = "88ab954a39ea6d3ce2b62cff086dd5ad1172c664"
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def load_audited_splits(splits_dir: Path, audit_path: Path, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Read only requested partitions and verify the original prepared split."""
    if not names or set(names) - {"train", "validation", "test"}:
        raise ValueError("Unknown or empty split selection")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = audit["splits"]["partitions"]
    splits = {name: pd.read_csv(splits_dir / f"{name}.csv", keep_default_na=False) for name in names}
    assert_disjoint_splits(splits)
    for name, frame in splits.items():
        if len(frame) != expected[name]["rows"] or frame_fingerprint(frame) != expected[name]["fingerprint"]:
            raise ValueError(f"{name} split differs from the prepared dataset audit")
        if set(frame["label"]) != set(LABELS):
            raise ValueError(f"{name} must contain all three sentiment classes")
    return splits


class EncodedSentimentDataset:
    """Map-style dataset with explicit named labels and recorded truncation.

    Dynamic batch padding happens in DataCollatorWithPadding. Length audits use
    the same tokenizer, including special tokens, before right truncation.
    """

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        if max_length < 4:
            raise ValueError("max_length must leave room for special tokens and text")
        if set(frame["label"]) - set(LABELS):
            raise ValueError("Unexpected named sentiment label")
        texts = frame["text"].tolist()
        original = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
        lengths = np.asarray([len(row) for row in original["input_ids"]])
        self.length_audit = {
            "rows": len(texts),
            "max_original_tokens": int(lengths.max()) if len(lengths) else 0,
            "rows_truncated": int((lengths > max_length).sum()),
            "max_length": max_length,
            "policy": "Right truncation including special tokens; dynamic padding per batch",
        }
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False, add_special_tokens=True)
        self.labels = [LABEL2ID[label] for label in frame["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{name: values[index] for name, values in self.encodings.items()}, "labels": self.labels[index]}


def validate_label_mapping(config) -> None:
    actual = {int(index): label for index, label in config.id2label.items()}
    if actual != ID2LABEL or config.label2id != LABEL2ID:
        raise ValueError("Saved checkpoint has an unexpected named label mapping")


def inference_file_hashes(model_dir: Path) -> dict[str, str]:
    """Hash only the exported inference files, never optimizer states or data."""
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "added_tokens.json", "model.safetensors", "model.safetensors.index.json"}
    paths = [path for path in model_dir.iterdir() if path.is_file() and (path.name in names or (path.name.startswith("model-") and path.suffix == ".safetensors"))]
    if not (model_dir / "config.json").is_file() or not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("Expected a saved configuration and safetensors inference weights")
    return {path.name: file_sha256(path) for path in sorted(paths)}


def validate_completed_manifest(model_dir: Path) -> tuple[dict, dict[str, str]]:
    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("smoke_test") is not False:
        raise ValueError("A completed full training run is required; smoke runs cannot be evaluated as final models")
    if manifest.get("method") != "full_fine_tuning" or manifest.get("test_used_for_training_or_selection") is not False:
        raise ValueError("Unexpected training method or test-use declaration")
    hashes = inference_file_hashes(model_dir)
    if hashes != manifest.get("inference_file_sha256"):
        raise ValueError("Saved inference files differ from the completed training manifest")
    return manifest, hashes


def reusable_evaluation(result_path: Path, test_fingerprint: str, model_hashes: dict, manifest_hash: str, max_length: int) -> dict | None:
    """Reuse a frozen result; a changed model or test must use another output."""
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = (test_fingerprint, model_hashes, manifest_hash, max_length)
    actual = (result.get("test_fingerprint"), result.get("inference_file_sha256"), result.get("training_manifest_sha256"), result.get("max_length"))
    if actual != expected:
        raise ValueError("Existing evaluation belongs to different data, model, or tokenization. Use a new output directory.")
    return result


def named_prediction_records(logits: np.ndarray, id2label: dict[int, str] = ID2LABEL) -> list[dict]:
    if {int(index): label for index, label in id2label.items()} != ID2LABEL:
        raise ValueError("Unexpected model label mapping")
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(LABELS) or not np.isfinite(values).all():
        raise ValueError("Expected finite three-class logits")
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    rows = []
    for row in probabilities:
        label = ID2LABEL[int(row.argmax())]
        scores = {name: float(row[index]) for index, name in ID2LABEL.items()}
        rows.append({"predicted": label, "score": scores[label], **scores})
    return rows
