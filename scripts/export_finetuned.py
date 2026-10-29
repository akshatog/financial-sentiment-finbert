"""Package a completed classifier and evidence; exclude private text and optimizer state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import ROOT, file_sha256
from src.finetuning import reusable_evaluation, validate_completed_manifest


def export(model_dir: Path, evaluation_dir: Path, destination: Path) -> dict:
    manifest, model_hashes = validate_completed_manifest(model_dir)
    evaluation_path = evaluation_dir / "evaluation.json"
    if not evaluation_path.is_file():
        raise ValueError("Run the separate held-out evaluation before exporting")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not evaluation.get("test_fingerprint") or not evaluation.get("metrics"):
        raise ValueError("A completed held-out evaluation is required")
    reusable_evaluation(evaluation_path, evaluation["test_fingerprint"], model_hashes,
                        file_sha256(model_dir / "training_manifest.json"),
                        int(manifest["settings"]["max_length"]))
    payloads = {}
    for path in model_dir.iterdir():
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt"}:
            payloads[f"models/finbert_finetuned/{path.name}"] = path
    if "models/finbert_finetuned/config.json" not in payloads or not any(name.endswith(".safetensors") for name in payloads):
        raise ValueError("Classifier config and safe model weights are required")
    for path in evaluation_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".png"}:
            relative = path.relative_to(evaluation_dir)
            if any(part in {"predictions", "private", "errors"} for part in relative.parts):
                continue
            payloads[f"outputs/finetuning/{relative.as_posix()}"] = path
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Results bundle already exists: {destination}. Preserve it or choose a new destination.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    temporary = destination.with_suffix(destination.suffix + ".part")
    with ZipFile(temporary, "x", compression=ZIP_DEFLATED) as archive:
        for name, path in sorted(payloads.items()):
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, name)
        archive.writestr("export_manifest.json", json.dumps({"schema_version": 1, "contents": "Completed model + aggregate evidence; excludes licensed dataset text, private predictions and resumable optimizer checkpoints", "files": hashes}, indent=2) + "\n")
    temporary.replace(destination)
    return {"archive": str(destination), "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "files": len(hashes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "finbert_finetuned")
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "outputs" / "finetuning")
    parser.add_argument("--destination", type=Path, default=ROOT / ".cache" / "training_exports" / "finbert_finetuned_results.zip")
    args = parser.parse_args()
    print(json.dumps(export(args.model_dir, args.evaluation_dir, args.destination), indent=2))


if __name__ == "__main__":
    main()
