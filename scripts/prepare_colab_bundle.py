"""Build a private, checksum-verified Colab input ZIP and its companion notebook.

The ZIP contains the existing local splits; never publish it. The notebook has
no credentials or dataset sentences and can be committed to the repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.dataset import ROOT, assert_disjoint_splits, frame_fingerprint


INPUT_FILES = (
    "requirements-training.txt",
    "src/__init__.py", "src/config.py", "src/dataset.py", "src/metrics.py", "src/finetuning.py",
    "scripts/__init__.py", "scripts/train_finbert.py", "scripts/evaluate_finetuned.py",
    "scripts/export_finetuned.py",
    "outputs/dataset_audit.json", "outputs/evaluation.json",
    "data/splits/train.csv", "data/splits/validation.csv", "data/splits/test.csv",
)


def _cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(bundle_hash: str) -> dict:
    cells = [
        _cell("markdown", """
# Financial News Sentiment Analysis Using FinBERT — actual fine-tuning

Akshat Agarwal · 202401100100028 · CS-5A · KIET Group of Institutions, Ghaziabad

**Before running:** Runtime → Change runtime type → GPU (T4 if available).
Then choose Runtime → Run all and upload `finbert_colab_training.zip` when asked.
No API key, GitHub token, or training code changes are required.

This trains a new sentiment classifier and the encoder of **HKUST
`yiyanghkust/finbert-pretrain`**, a different FinBERT release from the app's
existing Prosus reference. Only train/validation data is used during training.
The held-out test is evaluated after the best checkpoint is selected.
The short smoke test is a technical check and has no final accuracy claim.

Save the downloaded **finbert_finetuned_results.zip** in `D:\\Desktop\\CA2_DL`.
It contains the trained model, tokenizer, settings, logs, metrics, and figures.
GPU availability and runtime lifetime are controlled by Colab; download results
before disconnecting. Optional Drive backup is explained in the final notes.
"""),
        _cell("markdown", "## 1. Upload the private input bundle"),
        _cell("code", f'''
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, subprocess, sys, zipfile
from google.colab import files

EXPECTED_SHA256 = "{bundle_hash}"
uploaded = files.upload()
if set(uploaded) != {{"finbert_colab_training.zip"}}:
    raise ValueError("Upload only finbert_colab_training.zip from the project .cache/colab folder.")
payload = uploaded.pop("finbert_colab_training.zip")
if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
    raise ValueError("Bundle mismatch. Use the ZIP that accompanies this notebook.")
ROOT = Path("/content/financial_sentiment_training")
if ROOT.exists() and any(ROOT.iterdir()):
    raise FileExistsError("This runtime already has an extracted project. Continue with the next cell; do not replace it.")
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    manifest = json.loads(archive.read("bundle_manifest.json"))
    expected = set(manifest["files"]) | {{"bundle_manifest.json"}}
    if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
        raise ValueError("Unexpected ZIP members.")
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in manifest["files"].items():
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or "\\\\" in name or ":" in name:
            raise ValueError("Unsafe ZIP member.")
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"File checksum mismatch: {{name}}")
        destination = ROOT.joinpath(*member.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (ROOT / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
del payload, uploaded
os.chdir(ROOT)
print("Verified input bundle. Exact saved splits: train 3385 / validation 725 / test 726.")
'''),
        _cell("markdown", "## 2. Install training libraries and verify GPU"),
        _cell("code", '''
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements-training.txt"], check=True)
subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'No CUDA GPU: change runtime type to GPU, then rerun'; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__); print('GPU memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))"], check=True)
'''),
        _cell("markdown", """
## 3. Check real training, validation, and checkpoint saving

This short run uses only training/validation subsets. It must finish successfully
before the main run. It does not produce a model for the submitted app.
Warnings about newly initialized classifier/pooler weights and unused pretraining
head weights are expected. A traceback or failed cell needs attention.
"""),
        _cell("code", '''
subprocess.run([sys.executable, "scripts/train_finbert.py", "--smoke-test"], check=True)
'''),
        _cell("markdown", """
## 4. Fine-tune on all training rows

Full encoder + classifier fine-tuning, seed 42, 3 epochs, maximum 128 tokens,
learning rate 2e-5, microbatch 2, gradient accumulation 8 (effective batch 16 on
one GPU), mixed precision, dynamic padding, and gradient checkpointing.
Select the saved checkpoint with the best **validation macro-F1**.

Do not change settings based on test results. A new run uses a new output folder.
If this cell is interrupted, preserve any saved checkpoints before disconnecting
and ask for help resuming rather than repeating the test.
"""),
        _cell("code", '''
MODEL_DIR = "models/finbert_finetuned"
subprocess.run([sys.executable, "scripts/train_finbert.py", "--output-dir", MODEL_DIR], check=True)
'''),
        _cell("markdown", """
## 5. Evaluate the selected frozen checkpoint

The 726 test sentences are now used for evaluation, after training/selection.
Save accuracy, macro-F1, per-class results, and the confusion matrix. Compare
with our TF-IDF baseline on the exact same test membership. The older Prosus
score remains an overlapping-data diagnostic because Prosus was already
fine-tuned on Financial PhraseBank. This experiment does not prove that every
test sentence was absent from HKUST's large pretraining corpus.
"""),
        _cell("code", '''
subprocess.run([sys.executable, "scripts/evaluate_finetuned.py", "--model-dir", MODEL_DIR, "--output-dir", "outputs/finetuning"], check=True)
'''),
        _cell("markdown", "## 6. Export and download the result bundle"),
        _cell("code", '''
RESULT_ZIP = ROOT / "finbert_finetuned_results.zip"
if not RESULT_ZIP.exists():
    subprocess.run([sys.executable, "scripts/export_finetuned.py", "--model-dir", MODEL_DIR, "--evaluation-dir", "outputs/finetuning", "--destination", str(RESULT_ZIP)], check=True)
with zipfile.ZipFile(RESULT_ZIP) as archive:
    if archive.testzip() is not None:
        raise ValueError("The export ZIP is damaged; preserve the model and ask for help exporting again.")
print("Save this ZIP in D:\\\\Desktop\\\\CA2_DL. Keep its model and manifest together.")
files.download(str(RESULT_ZIP))
'''),
        _cell("markdown", """
## After downloading

Tell Codex the local ZIP path so it can verify hashes, inspect the training
evidence, load the saved classifier, and update the application, README and PPT
with measured results. Preparation or smoke-test success is not completed
FinBERT fine-tuning. Do not claim improvement until the final metrics exist.

If browser downloads are blocked, allow the download and rerun only the last
cell. Colab VM files can disappear when the runtime is deleted. For a longer run,
you can optionally mount Drive and use `--checkpoint-dir` pointing to a new
Drive folder before training; Drive authorization is your choice. Epoch
checkpoints are excluded from the final download to keep it smaller, so preserve
them separately if you need resumable optimizer state.

Sources: [HKUST checkpoint](https://huggingface.co/yiyanghkust/finbert-pretrain),
[authors' fine-tuning example](https://github.com/yya518/FinBERT/blob/master/finetune.ipynb),
[Colab resource/lifetime information](https://research.google.com/colaboratory/faq.html).
"""),
    ]
    return {"cells": cells, "metadata": {"colab": {"name": "finetune_finbert_colab.ipynb", "provenance": []}, "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def main() -> None:
    splits = {name: pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv", keep_default_na=False) for name in ("train", "validation", "test")}
    assert_disjoint_splits(splits)
    audit = json.loads((ROOT / "outputs" / "dataset_audit.json").read_text(encoding="utf-8"))
    for name, frame in splits.items():
        if frame_fingerprint(frame) != audit["splits"]["partitions"][name]["fingerprint"]:
            raise ValueError(f"{name} differs from the audited split")
    payloads = {name: (ROOT / name).read_bytes() for name in INPUT_FILES}
    manifest = {"schema_version": 1, "purpose": "Private Colab training input; do not publish dataset sentences", "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}, "split_fingerprints": {name: frame_fingerprint(frame) for name, frame in splits.items()}}
    payloads["bundle_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    directory = ROOT / ".cache" / "colab"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "finbert_colab_training.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in sorted(payloads.items()):
            entry = ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, payload)
    bundle_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    notebook_path = ROOT / "notebooks" / "finetune_finbert_colab.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook(bundle_hash), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(destination), "bytes": destination.stat().st_size, "sha256": bundle_hash, "notebook": str(notebook_path)}, indent=2))


if __name__ == "__main__":
    main()
