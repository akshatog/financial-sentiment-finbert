"""Verify resumable multipart transfer without network requests or model downloads."""
import hashlib
import io
import json

import pytest
from src import model_artifacts as artifacts


def setup_transfer(monkeypatch, tmp_path, corrupt=False):
    blocks = [b'first model archive segment', b'second model archive segment']
    content = b''.join(blocks)
    parts = [{'name': f'finbert_finetuned_results.zip.part{index:02d}', 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()} for index, block in enumerate(blocks)]
    manifest = {'archive_sha256': hashlib.sha256(content).hexdigest(), 'archive_bytes': len(content), 'parts': parts}
    metadata = (json.dumps(manifest) + '\n').encode()
    base = 'https://example.com/model'
    monkeypatch.setattr(artifacts, 'FINETUNED_PARTS_URL', base + '/finbert_finetuned_parts.json')
    monkeypatch.setattr(artifacts, 'FINETUNED_PARTS_SHA256', hashlib.sha256(metadata).hexdigest())
    monkeypatch.setattr(artifacts, 'FINETUNED_ARCHIVE_SHA256', manifest['archive_sha256'])
    calls = []
    def fetch(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith('.json'):
            return io.BytesIO(metadata)
        index = int(request.full_url[-2:])
        return io.BytesIO(b'corrupt' if corrupt else blocks[index])
    monkeypatch.setattr(artifacts, 'urlopen', fetch)
    return blocks, content, calls


def test_multipart_transfer_reuses_verified_part_and_reconstructs_exact_archive(tmp_path, monkeypatch):
    blocks, content, calls = setup_transfer(monkeypatch, tmp_path)
    (tmp_path / 'finbert_finetuned_results.zip.part00').write_bytes(blocks[0])
    archive = tmp_path / 'model.zip'
    artifacts._download_archive(archive)
    assert archive.read_bytes() == content
    assert len(calls) == 2
    assert not any(url.endswith('part00') for url in calls)
    artifacts._download_archive(archive)
    assert len(calls) == 2


def test_corrupt_download_is_never_installed(tmp_path, monkeypatch):
    setup_transfer(monkeypatch, tmp_path, corrupt=True)
    archive = tmp_path / 'model.zip'
    with pytest.raises(ValueError, match='checksum mismatch'):
        artifacts._download_archive(archive)
    assert not archive.exists()
    assert not (tmp_path / 'finbert_finetuned_results.zip.part00').exists()
