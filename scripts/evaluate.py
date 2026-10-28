"""Evaluate the frozen baseline once on held-out test data and save evidence.

Existing matching baseline results are reused. --force deliberately recomputes
the evaluation and records that action. FinBERT scores, if requested, are an
overlapping-training-data diagnostic and cannot be used as an unseen benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.baseline import BaselinePredictor, DEFAULT_MODEL_PATH
from src.dataset import LABELS, ROOT, assert_disjoint_splits, file_sha256, frame_fingerprint
from src.metrics import classification_metrics, plot_confusion, plot_model_comparison


def predict_and_save(predictor, test: pd.DataFrame, name: str, output_dir: Path) -> tuple[dict, pd.DataFrame]:
    records = predictor.predict(test["text"].tolist())
    if len(records) != len(test):
        raise ValueError(f"{name} returned the wrong number of predictions")
    predictions = pd.DataFrame(records).rename(columns={"label": "predicted"})
    required = {"predicted", "score", *LABELS}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{name} prediction schema is missing fields")
    metrics = classification_metrics(test["label"].tolist(), predictions["predicted"].tolist())
    combined = pd.concat([test.reset_index(drop=True), predictions], axis=1)
    private_dir = output_dir / "predictions"
    private_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(private_dir / f"{name}_test.csv", index=False, encoding="utf-8")
    errors = combined.loc[combined["label"] != combined["predicted"]].sort_values("score", ascending=False)
    errors.to_csv(private_dir / f"{name}_errors.csv", index=False, encoding="utf-8")
    return metrics, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--finbert", action="store_true", help="Also run pretrained FinBERT as a separately labeled overlapping-data diagnostic")
    parser.add_argument("--force", action="store_true", help="Explicitly repeat a previously saved baseline test evaluation")
    args = parser.parse_args()
    splits = {name: pd.read_csv(args.splits_dir / f"{name}.csv", keep_default_na=False) for name in ("train", "validation", "test")}
    assert_disjoint_splits(splits)
    metadata_path = args.model_path.parent / "baseline_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name in ("train", "validation"):
        if frame_fingerprint(splits[name]) != metadata[f"{name}_fingerprint"]:
            raise ValueError(f"{name} split does not match the data used to train/select the saved model")
    model_hash = file_sha256(args.model_path)
    if model_hash != metadata["model_sha256"]:
        raise ValueError("Saved model does not match baseline_metadata.json")
    test_hash = frame_fingerprint(splits["test"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "evaluation.json"
    previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    if previous and not args.force:
        if previous.get("test_fingerprint") != test_hash or previous.get("baseline_model_sha256") != model_hash:
            raise ValueError("Existing evaluation belongs to different data/model. Use a separate output directory, or explicitly --force after reviewing the change.")
        results = previous
        print("Reusing saved matching held-out baseline evaluation; no baseline test inference repeated.")
    else:
        baseline, errors = predict_and_save(BaselinePredictor(args.model_path), splits["test"], "baseline", args.output_dir)
        baseline["provenance"] = {
            "interpretation": "Held-out test evaluation of our independently trained conventional baseline",
            "training_data": "Financial PhraseBank 50Agree, saved training partition only",
            "selection_data": "Saved validation partition only; C selected by macro F1",
            "test_data": "Saved test partition, excluded from TF-IDF fitting, model fitting and selection",
            "named_labels": True,
            "normalized_sentence_overlap_with_train_or_validation": 0,
            "test_used_for_training_or_selection": False,
        }
        majority_label = metadata["majority_label"]
        majority = classification_metrics(splits["test"]["label"].tolist(), [majority_label] * len(splits["test"]))
        majority["label"] = majority_label
        majority["provenance"] = {"interpretation": "Majority class selected from training labels only", "test_used_for_class_selection": False}
        results = {
            "schema_version": 1,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_evaluation_runs": int(previous.get("baseline_evaluation_runs", 1) + 1) if previous else 1,
            "explicit_repeat_requested": bool(args.force),
            "dataset": "Financial PhraseBank v1.0, cleaned 50Agree, seeded stratified split",
            "test_fingerprint": test_hash,
            "test_rows": len(splits["test"]),
            "baseline_model_sha256": model_hash,
            "baseline": baseline,
            "majority_baseline": majority,
            "finbert": {"status": "not_evaluated", "checkpoint": "ProsusAI/finbert", "interpretation": "FinBERT was fine-tuned on Financial PhraseBank. Any score here is an overlapping-data diagnostic, never a fair unseen-data comparison with our baseline."},
        }
        pairs = Counter((row.label, row.predicted) for row in errors.itertuples())
        analysis = {
            "model": "baseline",
            "total_errors": len(errors),
            "correct_predictions": int(len(splits["test"]) - len(errors)),
            "error_pairs": [{"true": true, "predicted": predicted, "count": count} for (true, predicted), count in sorted(pairs.items())],
            "high_confidence_error_ids": errors["id"].head(12).tolist(),
            "private_error_file": "outputs/predictions/baseline_errors.csv",
            "publication_policy": "Public report includes aggregate counts and record IDs only. Full licensed dataset sentences remain in ignored private local files.",
            "review_prompts": ["Inspect errors involving negation and changes over time", "Inspect mixed information and missing financial context", "Review neutral versus weak positive/negative confusion without claiming an automatic cause"],
        }
        (args.output_dir / "error_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        plot_confusion(baseline, args.output_dir / "figures" / "confusion_baseline.png", "TF-IDF + logistic regression: held-out test")
        plot_model_comparison(baseline, majority, args.output_dir / "figures" / "baseline_comparison.png")
    # Persist the frozen baseline before any optional model download/inference.
    # A diagnostic failure must not cause a later invocation to repeat this test.
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.finbert and (results["finbert"].get("status") != "evaluated" or args.force):
        # Import lazily so baseline evaluation never requires torch/model weights.
        from src.inference import FinBERTPredictor
        diagnostic, _ = predict_and_save(FinBERTPredictor(), splits["test"], "finbert", args.output_dir)
        diagnostic.update({"status": "evaluated", "checkpoint": "ProsusAI/finbert", "provenance": {"interpretation": "Overlapping-data diagnostic; checkpoint was fine-tuned on Financial PhraseBank", "independent_test_claim": False, "fair_unseen_comparison_to_baseline": False, "checkpoint_training_overlap": "Known dataset-level overlap; exact original checkpoint row membership is not established"}})
        manifest_path = ROOT / "outputs" / "model_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            diagnostic["provenance"]["checkpoint_revision"] = manifest.get("revision")
        results["finbert"] = diagnostic
        plot_confusion(diagnostic, args.output_dir / "figures" / "confusion_finbert_diagnostic.png", "FinBERT: overlapping-data diagnostic")
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_accuracy": results["baseline"]["accuracy"], "baseline_macro_f1": results["baseline"]["macro_f1"], "majority_accuracy": results["majority_baseline"]["accuracy"], "majority_macro_f1": results["majority_baseline"]["macro_f1"], "test_rows": results["test_rows"], "finbert_status": results["finbert"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
