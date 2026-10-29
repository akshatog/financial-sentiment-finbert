"""Evaluate a completed, frozen student-fine-tuned FinBERT on held-out test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ROOT, setup_cache
from src.dataset import file_sha256, frame_fingerprint
from src.finetuning import EncodedSentimentDataset, load_audited_splits, named_prediction_records, reusable_evaluation, validate_completed_manifest, validate_label_mapping
from src.metrics import classification_metrics, plot_confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "finbert_finetuned")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--audit-path", type=Path, default=ROOT / "outputs" / "dataset_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "finetuning")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="Explicitly repeat matching frozen-model test inference")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    setup_cache()
    manifest, hashes = validate_completed_manifest(args.model_dir)
    splits = load_audited_splits(args.splits_dir, args.audit_path, ("train", "validation", "test"))
    for name in ("train", "validation"):
        if frame_fingerprint(splits[name]) != manifest["split_fingerprints"][name]:
            raise ValueError(f"{name} does not match the training manifest")
    max_length = int(manifest["settings"]["max_length"])
    test = splits["test"]
    test_hash = frame_fingerprint(test)
    manifest_hash = file_sha256(args.model_dir / "training_manifest.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "evaluation.json"
    previous = reusable_evaluation(result_path, test_hash, hashes, manifest_hash, max_length)
    if previous and not args.force:
        print("Reusing saved matching frozen-model held-out evaluation; no test inference repeated.")
        print(json.dumps({"accuracy": previous["metrics"]["accuracy"], "macro_f1": previous["metrics"]["macro_f1"], "test_rows": len(test)}, indent=2))
        return
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    gpu = torch.cuda.is_available()
    if args.device == "cuda" and not gpu:
        parser.error("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and gpu) else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False).to(device).eval()
    validate_label_mapping(model.config)
    dataset = EncodedSentimentDataset(test, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer), num_workers=0)
    records = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
            logits = model(**inputs).logits.float().cpu().numpy()
            records.extend(named_prediction_records(logits, model.config.id2label))
    predictions = pd.DataFrame(records)
    if len(predictions) != len(test):
        raise ValueError("Wrong number of held-out predictions")
    metrics = classification_metrics(test["label"].tolist(), predictions["predicted"].tolist())
    result = {
        "schema_version": 1, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Student-fine-tuned HKUST FinBERT", "base_model": manifest["base_model"], "base_revision": manifest["base_revision"],
        "training_run_id": manifest["run_id"], "test_fingerprint": test_hash, "test_rows": len(test),
        "inference_file_sha256": hashes, "training_manifest_sha256": manifest_hash, "max_length": max_length,
        "test_tokenization": dataset.length_audit, "metrics": metrics,
        "test_evaluation_runs": int(previous.get("test_evaluation_runs", 1) + 1) if previous else 1,
        "explicit_repeat_requested": args.force,
        "provenance": {"test_used_for_training_or_selection": False, "normalized_sentence_overlap_with_train_or_validation": 0, "selection": manifest["selection"], "limitation": "One seeded PhraseBank split. Original Prosus checkpoint's PhraseBank diagnostic is not a fair unseen-data benchmark. Unpublished language-model pretraining overlap is not established."},
    }
    baseline_path = ROOT / "outputs" / "evaluation.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("test_fingerprint") == test_hash:
            reference = baseline["baseline"]
            result["same_test_baseline_comparison"] = {
                "baseline": "TF-IDF + logistic regression",
                "baseline_accuracy": reference["accuracy"], "baseline_macro_f1": reference["macro_f1"],
                "finetuned_accuracy": metrics["accuracy"], "finetuned_macro_f1": metrics["macro_f1"],
                "accuracy_change": metrics["accuracy"] - reference["accuracy"],
                "macro_f1_change": metrics["macro_f1"] - reference["macro_f1"],
                "interpretation": "Descriptive comparison on the same seeded test split; no statistical significance or improvement is assumed.",
            }
    # Save the frozen result before ancillary plotting, so plot failures do not
    # silently rerun inference on the test set at the next invocation.
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    private_dir = args.output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([test.reset_index(drop=True), predictions], axis=1)
    combined.to_csv(private_dir / "finetuned_test_predictions.csv", index=False, encoding="utf-8")
    combined.loc[combined["label"] != combined["predicted"]].to_csv(private_dir / "finetuned_errors.csv", index=False, encoding="utf-8")
    plot_confusion(metrics, args.output_dir / "confusion_finetuned.png", "Student-fine-tuned FinBERT: held-out test")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "test_rows": len(test), "result_path": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
