"""Train TF-IDF + logistic regression using saved train/validation data only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.baseline import DEFAULT_MODEL_PATH, train_baseline
from src.dataset import ROOT, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    args = parser.parse_args()
    train_path, validation_path = args.splits_dir / "train.csv", args.splits_dir / "validation.csv"
    if not train_path.is_file() or not validation_path.is_file():
        parser.error("Saved train/validation splits are missing. Run python scripts/prepare_data.py first.")
    train = pd.read_csv(train_path, keep_default_na=False)
    validation = pd.read_csv(validation_path, keep_default_na=False)
    start = time.perf_counter()
    model, metadata = train_baseline(train, validation, tuple(args.c_values), args.seed)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path, compress=3)
    metadata.update({
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "model_file": args.model_path.name,
        "model_sha256": file_sha256(args.model_path),
        "dataset": "Financial PhraseBank v1.0, 50Agree; cleaned canonical split",
        "versions": {package: version(package) for package in ("scikit-learn", "numpy", "pandas", "joblib")},
    })
    metadata_path = args.model_path.parent / "baseline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.model_path), "selected_C": metadata["selected_C"], "validation_macro_f1": metadata["validation_macro_f1"], "training_rows": metadata["training_rows"], "features": metadata["feature_count"]}, indent=2))


if __name__ == "__main__":
    main()
