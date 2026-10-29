import json
from zipfile import ZipFile

import pytest

from src.inference import (BaselineAdapter, FinBERTPredictor,
                           FineTunedFinBERTPredictor, ModelInputError,
                           ModelLoadError, get_predictor)


def test_model_selection_rejects_typo_instead_of_loading_a_different_model():
    with pytest.raises(ValueError, match='Unknown model'):
        get_predictor('Unknown')


def test_baseline_adapter_rejects_blank_input_before_prediction():
    adapter = object.__new__(BaselineAdapter)
    with pytest.raises(ModelInputError, match='non-empty'):
        adapter.predict([' '])
    assert adapter.predict([]) == []


def test_baseline_reports_missing_artifact(tmp_path):
    with pytest.raises(RuntimeError, match='Train the baseline'):
        BaselineAdapter(tmp_path / 'missing.joblib')


def test_explicit_missing_checkpoint_never_falls_back_to_a_remote_model(tmp_path, monkeypatch):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    def unexpected_download(*args, **kwargs):
        pytest.fail('An explicit missing checkpoint must not download or substitute a remote model')
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', unexpected_download)
    monkeypatch.setattr(AutoModelForSequenceClassification, 'from_pretrained', unexpected_download)
    with pytest.raises(ModelLoadError, match='missing config.json'):
        FinBERTPredictor(model_path=tmp_path / 'missing')


def test_unverified_finetuned_checkpoint_does_not_substitute_prosus(monkeypatch):
    import src.model_artifacts
    def invalid_artifact():
        raise ValueError('checksum mismatch')
    monkeypatch.setattr(src.model_artifacts, 'ensure_finetuned_model', invalid_artifact)
    with pytest.raises(ModelLoadError, match='no other model was substituted'):
        get_predictor()


def test_finetuned_rejects_over_policy_limit_before_inference():
    predictor = object.__new__(FineTunedFinBERTPredictor)
    predictor.max_tokens = 128
    predictor.token_lengths = lambda texts: [129] * len(texts)
    with pytest.raises(ModelInputError, match='at most 128 tokens'):
        predictor.predict(['An over-limit financial statement'])


def test_cached_artifact_checksum_mismatch_does_not_download_over_it(tmp_path, monkeypatch):
    import src.model_artifacts as artifacts
    archive = tmp_path / 'checkpoint.zip'
    archive.write_bytes(b'not the published model')
    def unexpected_network(*args, **kwargs):
        pytest.fail('A mismatched local archive must be preserved, not silently replaced')
    monkeypatch.setattr(artifacts, 'urlopen', unexpected_network)
    with pytest.raises(ValueError, match='unexpected checksum'):
        artifacts._download_archive(archive)
    assert archive.read_bytes() == b'not the published model'


def test_artifact_import_rejects_traversal_before_writing_model_files(tmp_path):
    import src.model_artifacts as artifacts
    names = {f'models/finbert_finetuned/{name}': 'dummy' for name in artifacts._MODEL_FILES}
    names['../../escape.txt'] = 'dummy'
    archive = tmp_path / 'unsafe.zip'
    with ZipFile(archive, 'w') as bundle:
        for name in names:
            bundle.writestr(name, b'dummy')
        bundle.writestr('export_manifest.json', json.dumps({'files': names}))
    with pytest.raises(ValueError, match='Unsafe checkpoint archive path'):
        artifacts._extract_model(archive, tmp_path / 'model', tmp_path)
    assert not (tmp_path / 'model').exists()
