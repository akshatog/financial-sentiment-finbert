"""Fine-tune the HKUST finance language model; test data is never loaded here."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, setup_cache
from src.dataset import frame_fingerprint
from src.finetuning import EncodedSentimentDataset, ID2LABEL, LABEL2ID, PRETRAINED_MODEL_ID, PRETRAINED_REVISION, inference_file_hashes, load_audited_splits, validate_label_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--audit-path", type=Path, default=ROOT / "outputs" / "dataset_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "finbert_finetuned")
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional persistent checkpoint directory, e.g. a mounted Drive folder")
    parser.add_argument("--revision", default=PRETRAINED_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Five optimizer steps and small validation; never a final trained model")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicit opt-in for slow CPU development; GPU is recommended")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation) < 1 or not 4 <= args.max_length <= 512 or args.learning_rate <= 0:
        parser.error("Positive training settings and max-length in [4,512] are required")
    if len(args.revision) != 40 or any(character not in "0123456789abcdef" for character in args.revision):
        parser.error("--revision must be an immutable 40-character checkpoint commit SHA")
    setup_cache()
    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("CUDA GPU unavailable. In Colab select a GPU runtime. CPU execution needs explicit --allow-cpu.")
    try:
        version("accelerate")
    except PackageNotFoundError:
        parser.error("Training requires accelerate. Install the separate training requirements, not the app's CPU requirements.")
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    # Deliberately do not open test.csv, including during tokenization audits.
    splits = load_audited_splits(args.splits_dir, args.audit_path, ("train", "validation"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir = ROOT / ".cache" / "finbert-smoke" / run_id if args.smoke_test else args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output directory: {output_dir}")
    checkpoint_dir = args.checkpoint_dir or output_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        parser.error(f"Checkpoint directory must be new and empty: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    hardware = {"device": "cuda" if gpu else "cpu", "gpu_name": torch.cuda.get_device_name(0) if gpu else None, "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if gpu else None}
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "started", "smoke_test": args.smoke_test,
        "method": "full_fine_tuning", "base_model": PRETRAINED_MODEL_ID, "base_revision": args.revision,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "hardware": hardware,
        "label2id": LABEL2ID, "id2label": ID2LABEL, "test_used_for_training_or_selection": False,
        "source_split_rows": {name: len(frame) for name, frame in splits.items()},
        "split_fingerprints": {name: frame_fingerprint(frame) for name, frame in splits.items()},
        "settings": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation, "effective_batch_size": args.batch_size * args.gradient_accumulation, "max_length": args.max_length, "learning_rate": args.learning_rate, "weight_decay": 0.01, "warmup_ratio": 0.1, "fp16": gpu, "gradient_checkpointing": gpu},
        "selection": "Best epoch by validation macro-F1; test is excluded",
        "provenance": "HKUST financial language-model pretraining plus our supervised training partition; distinct from ProsusAI/finbert, which already received Financial PhraseBank sentiment fine-tuning. No assertion that unpublished pretraining corpora have zero sentence overlap.",
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate", "numpy", "pandas", "scikit-learn")},
    }
    manifest_path = output_dir / "training_manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save_manifest()
    try:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_ID, revision=args.revision, trust_remote_code=False)
        tokenizer.truncation_side = "right"
        tokenizer.padding_side = "right"
        if args.smoke_test:
            # Balanced diagnostic subsets only. The full source hashes stay recorded.
            splits = {name: frame.groupby("label", group_keys=False).head(16 if name == "train" else 4).reset_index(drop=True) for name, frame in splits.items()}
        datasets = {name: EncodedSentimentDataset(frame, tokenizer, args.max_length) for name, frame in splits.items()}
        model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_ID, revision=args.revision, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL, trust_remote_code=False)
        for parameter in model.parameters():
            parameter.requires_grad = True
        validate_label_mapping(model.config)
        manifest["parameter_counts"] = {"total": sum(parameter.numel() for parameter in model.parameters()), "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
        manifest["tokenization"] = {name: dataset.length_audit for name, dataset in datasets.items()}
        manifest["actual_rows"] = {name: len(frame) for name, frame in splits.items()}
        # Track an encoder tensor, not just the fresh classification head. This
        # detects runs whose optimizer updates were all skipped by AMP.
        encoder_name = "bert.encoder.layer.0.attention.self.query.weight"
        encoder_before = dict(model.named_parameters())[encoder_name].detach().float().cpu().clone()
        save_manifest()

        def compute_metrics(prediction):
            logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
            if not np.isfinite(logits).all():
                raise ValueError("Non-finite validation logits; training cannot be marked completed")
            predicted = np.asarray(logits).argmax(axis=1)
            return {"accuracy": float(accuracy_score(prediction.label_ids, predicted)), "macro_f1": float(f1_score(prediction.label_ids, predicted, labels=list(ID2LABEL), average="macro", zero_division=0))}

        arguments = TrainingArguments(
            output_dir=str(checkpoint_dir), num_train_epochs=args.epochs,
            max_steps=5 if args.smoke_test else -1,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.1,
            eval_strategy="steps" if args.smoke_test else "epoch", eval_steps=5 if args.smoke_test else None,
            save_strategy="steps" if args.smoke_test else "epoch", save_steps=5 if args.smoke_test else 500,
            load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
            save_total_limit=2, save_safetensors=True, optim="adamw_torch",
            logging_strategy="steps", logging_steps=1 if args.smoke_test else 20,
            logging_nan_inf_filter=False,
            fp16=gpu, bf16=False, gradient_checkpointing=gpu,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gpu else None,
            seed=args.seed, data_seed=args.seed, report_to=[], push_to_hub=False,
            dataloader_num_workers=0, dataloader_pin_memory=gpu, use_cpu=not gpu,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=datasets["train"], eval_dataset=datasets["validation"], processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if gpu else None), compute_metrics=compute_metrics)
        trained = trainer.train()
        validation_metrics = trainer.evaluate()
        losses = [float(trained.training_loss)] + [float(row[key]) for row in trainer.state.log_history for key in ("loss", "eval_loss") if key in row]
        if not all(np.isfinite(value) for value in losses) or not all(np.isfinite(value) for value in validation_metrics.values() if isinstance(value, (int, float))):
            raise ValueError("Non-finite training loss or validation metrics")
        encoder_after = dict(trainer.model.named_parameters())[encoder_name].detach().float().cpu()
        if trainer.state.global_step < 1 or torch.equal(encoder_before, encoder_after):
            raise ValueError("No encoder parameter change was observed; this is not a completed fine-tuning run")
        if not all(bool(torch.isfinite(parameter).all()) for parameter in trainer.model.parameters()):
            raise ValueError("Model parameters contain non-finite values")
        manifest["encoder_update_evidence"] = {"parameter": encoder_name, "changed": True, "before_sha256": hashlib.sha256(encoder_before.numpy().tobytes()).hexdigest(), "after_sha256": hashlib.sha256(encoder_after.numpy().tobytes()).hexdigest(), "l2_change": float(torch.linalg.vector_norm(encoder_after - encoder_before))}
        trainer.save_state()
        (output_dir / "training_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")
        manifest.update({"status": "smoke_completed" if args.smoke_test else "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.perf_counter() - started, 3), "global_optimizer_steps": trainer.state.global_step, "selected_checkpoint": trainer.state.best_model_checkpoint, "best_validation_macro_f1": trainer.state.best_metric, "validation_metrics": validation_metrics, "training_metrics": trained.metrics, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else None, "checkpoint_directory": str(checkpoint_dir)})
        if not args.smoke_test:
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(output_dir)
            manifest["inference_file_sha256"] = inference_file_hashes(output_dir)
        save_manifest()
        print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir), "elapsed_seconds": manifest["elapsed_seconds"], "optimizer_steps": trainer.state.global_step, "validation_macro_f1": manifest["best_validation_macro_f1"], "test_evaluated": False}, indent=2))
    except Exception as error:
        manifest.update({"status": "failed", "error_type": type(error).__name__, "elapsed_seconds": round(time.perf_counter() - started, 3)})
        save_manifest()
        raise


if __name__ == "__main__":
    main()
