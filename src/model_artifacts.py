"""Fetch only the pinned student-trained checkpoint; never substitute a model."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path, PurePosixPath
import tempfile
import threading
from urllib.request import Request, urlopen
from zipfile import ZipFile

from src.config import (FINETUNED_ARCHIVE_SHA256, FINETUNED_ARCHIVE_URL,
                        FINETUNED_BASE_ID, FINETUNED_BASE_REVISION,
                        FINETUNED_MANIFEST_SHA256, FINETUNED_MODEL_PATH,
                        FINETUNED_PARTS_SHA256, FINETUNED_PARTS_URL,
                        FINETUNED_WEIGHTS_SHA256, setup_cache)
from src.dataset import file_sha256
from src.finetuning import LABEL2ID, validate_completed_manifest

_LOCK = threading.Lock()
_MODEL_FILES = {
    'config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer.json',
    'tokenizer_config.json', 'training_history.json', 'training_manifest.json', 'vocab.txt',
}


def validate_finetuned_artifact(model_dir: Path) -> dict:
    """Bind local weights, tokenizer, label order and training record to the release."""
    manifest, hashes = validate_completed_manifest(model_dir)
    if file_sha256(model_dir / 'training_manifest.json') != FINETUNED_MANIFEST_SHA256:
        raise ValueError('Training record does not match the published fine-tuned checkpoint.')
    if hashes.get('model.safetensors') != FINETUNED_WEIGHTS_SHA256:
        raise ValueError('Fine-tuned weights do not match the published checkpoint.')
    if manifest.get('base_model') != FINETUNED_BASE_ID or manifest.get('base_revision') != FINETUNED_BASE_REVISION:
        raise ValueError('Unexpected starting checkpoint for the fine-tuned model.')
    if manifest.get('label2id') != LABEL2ID or manifest.get('settings', {}).get('max_length') != 128:
        raise ValueError('Unexpected fine-tuned label mapping or input policy.')
    return manifest


def _download_archive(destination: Path) -> None:
    if destination.exists():
        if file_sha256(destination) != FINETUNED_ARCHIVE_SHA256:
            raise ValueError('Cached checkpoint archive has an unexpected checksum. Preserve it and inspect before retrying.')
        return
    temporary = destination.with_suffix('.zip.part')
    try:
        request = Request(FINETUNED_PARTS_URL, headers={'User-Agent': 'FinancialSentimentLab/1.0'})
        with urlopen(request, timeout=60) as response:
            metadata = response.read(64 * 1024 + 1)
        if len(metadata) > 64 * 1024 or hashlib.sha256(metadata).hexdigest() != FINETUNED_PARTS_SHA256:
            raise ValueError('Model transfer manifest does not match the pinned release.')
        parts_manifest = json.loads(metadata)
        parts = parts_manifest['parts']
        if parts_manifest['archive_sha256'] != FINETUNED_ARCHIVE_SHA256 or not 1 <= len(parts) <= 100:
            raise ValueError('Unexpected model archive declaration.')
        if not 0 < parts_manifest['archive_bytes'] <= 550 * 1024 * 1024:
            raise ValueError('Unexpected model archive size.')
        total = 0
        base_url = FINETUNED_PARTS_URL.rsplit('/', 1)[0]
        with temporary.open('wb') as stream:
            for index, part in enumerate(parts):
                name = f'finbert_finetuned_results.zip.part{index:02d}'
                if part['name'] != name or not 0 < part['bytes'] <= 20 * 1024 * 1024:
                    raise ValueError('Unexpected model transfer part.')
                part_path = destination.parent / name
                if part_path.exists():
                    if part_path.stat().st_size != part['bytes'] or file_sha256(part_path) != part['sha256']:
                        raise ValueError('Cached model transfer part has an unexpected checksum; preserve it for inspection.')
                else:
                    part_temporary = part_path.with_suffix(part_path.suffix + '.part')
                    request = Request(f'{base_url}/{name}', headers={'User-Agent': 'FinancialSentimentLab/1.0'})
                    try:
                        with urlopen(request, timeout=60) as response, part_temporary.open('wb') as output:
                            received = 0
                            while chunk := response.read(4 * 1024 * 1024):
                                received += len(chunk)
                                if received > part['bytes']:
                                    raise ValueError('Model transfer part exceeded its declared size.')
                                output.write(chunk)
                        if received != part['bytes'] or file_sha256(part_temporary) != part['sha256']:
                            raise ValueError('Model transfer part checksum mismatch.')
                        part_temporary.replace(part_path)
                    finally:
                        if part_temporary.is_file():
                            part_temporary.unlink()
                with part_path.open('rb') as source:
                    while chunk := source.read(4 * 1024 * 1024):
                        total += len(chunk)
                        stream.write(chunk)
        if total != parts_manifest['archive_bytes']:
            raise ValueError('Reconstructed model archive has the wrong size.')
        if file_sha256(temporary) != FINETUNED_ARCHIVE_SHA256:
            raise ValueError('Downloaded checkpoint checksum does not match the pinned release.')
        temporary.replace(destination)
    finally:
        if temporary.is_file():
            temporary.unlink()


def _extract_model(archive_path: Path, destination: Path, cache: Path) -> None:
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        exported = json.loads(archive.read('export_manifest.json'))
        if len(names) != len(set(names)) or set(names) != set(exported['files']) | {'export_manifest.json'}:
            raise ValueError('Unexpected checkpoint archive members.')
        if sum(member.file_size for member in archive.infolist()) > 1024**3:
            raise ValueError('Unexpected uncompressed checkpoint size.')
        prefix = 'models/finbert_finetuned/'
        model_members = {name.removeprefix(prefix): name for name in names if name.startswith(prefix)}
        if set(model_members) != _MODEL_FILES:
            raise ValueError('Checkpoint archive is missing expected model files.')
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe checkpoint archive path.')
        with tempfile.TemporaryDirectory(prefix='finbert-import-', dir=cache) as directory:
            staging = Path(directory) / 'model'
            staging.mkdir()
            for filename, member in model_members.items():
                target = staging / filename
                with archive.open(member) as source, target.open('wb') as stream:
                    while chunk := source.read(4 * 1024 * 1024):
                        stream.write(chunk)
                if file_sha256(target) != exported['files'][member]:
                    raise ValueError(f'Checkpoint member checksum mismatch: {filename}')
            validate_finetuned_artifact(staging)
            if destination.exists():
                raise FileExistsError('A model directory appeared during import; preserve it and retry validation.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(destination)


def ensure_finetuned_model() -> Path:
    """Validate a local checkpoint, or download/import the exact release on demand."""
    with _LOCK:
        if FINETUNED_MODEL_PATH.exists():
            validate_finetuned_artifact(FINETUNED_MODEL_PATH)
            return FINETUNED_MODEL_PATH
        cache = setup_cache() / 'model-artifacts'
        cache.mkdir(parents=True, exist_ok=True)
        archive_path = cache / 'finbert_finetuned_results.zip'
        _download_archive(archive_path)
        _extract_model(archive_path, FINETUNED_MODEL_PATH, cache)
        return FINETUNED_MODEL_PATH
