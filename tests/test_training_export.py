"""Check that exported evidence is bound to a completed frozen checkpoint."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from scripts.export_finetuned import export
from src.dataset import ROOT, file_sha256
from src.finetuning import inference_file_hashes


class TrainingExportTests(unittest.TestCase):
    def fixture(self, root):
        model, evaluation = root / "model", root / "evaluation"
        model.mkdir()
        evaluation.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        # Synthetic bytes exercise packaging guards, never model inference.
        (model / "model.safetensors").write_bytes(b"synthetic packaging fixture")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        hashes = inference_file_hashes(model)
        manifest = {"status": "completed", "smoke_test": False, "method": "full_fine_tuning", "test_used_for_training_or_selection": False, "settings": {"max_length": 128}, "inference_file_sha256": hashes}
        (model / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = {"test_fingerprint": "frozen-test", "metrics": {"accuracy": 0.5}, "inference_file_sha256": hashes, "training_manifest_sha256": file_sha256(model / "training_manifest.json"), "max_length": 128}
        (evaluation / "evaluation.json").write_text(json.dumps(result), encoding="utf-8")
        return model, evaluation

    def test_export_excludes_raw_predictions_and_optimizer_states(self):
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            model, evaluation = self.fixture(root)
            (model / "checkpoints").mkdir()
            (model / "checkpoints" / "optimizer.pt").write_bytes(b"not for export")
            (evaluation / "private").mkdir()
            (evaluation / "private" / "predictions.csv").write_text("private text", encoding="utf-8")
            destination = root / "result.zip"
            export(model, evaluation, destination)
            with ZipFile(destination) as archive:
                names = archive.namelist()
                self.assertIn("models/finbert_finetuned/model.safetensors", names)
                self.assertIn("outputs/finetuning/evaluation.json", names)
                self.assertFalse(any("optimizer" in name or "private" in name or name.endswith(".csv") for name in names))
            with self.assertRaises(FileExistsError):
                export(model, evaluation, destination)

    def test_export_rejects_smoke_or_tampered_model(self):
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            model, evaluation = self.fixture(root)
            (model / "model.safetensors").write_bytes(b"changed after evaluation")
            with self.assertRaisesRegex(ValueError, "differ"):
                export(model, evaluation, root / "result.zip")
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            model, evaluation = self.fixture(root)
            manifest = json.loads((model / "training_manifest.json").read_text(encoding="utf-8"))
            manifest["smoke_test"] = True
            (model / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke"):
                export(model, evaluation, root / "result.zip")

    def test_export_rejects_evaluation_from_another_model(self):
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            model, evaluation = self.fixture(root)
            result = json.loads((evaluation / "evaluation.json").read_text(encoding="utf-8"))
            result["training_manifest_sha256"] = "another run"
            (evaluation / "evaluation.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different"):
                export(model, evaluation, root / "result.zip")
