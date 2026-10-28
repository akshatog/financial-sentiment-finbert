import pytest

from src.inference import BaselineAdapter, ModelInputError, get_predictor


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
