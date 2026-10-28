"""UI behavior checks use a local fake predictor and never download weights."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import src.inference


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FakePredictor:
    max_tokens = 512

    def token_lengths(self, texts):
        return [len(text.split()) + 2 for text in texts]

    def predict(self, texts):
        return [
            {"label": "positive", "score": 0.8, "positive": 0.8, "neutral": 0.15, "negative": 0.05}
            for _ in texts
        ]


@pytest.fixture
def app(monkeypatch):
    calls = []

    def fake_get_predictor(model_name="FinBERT", device="cpu"):
        calls.append((model_name, device))
        return FakePredictor()

    monkeypatch.setattr(src.inference, "get_predictor", fake_get_predictor)
    import streamlit as st

    st.cache_resource.clear()
    result = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not result.exception
    return result, calls


def test_startup_does_not_load_a_model(app):
    ui, calls = app
    assert calls == []
    assert [tab.label for tab in ui.tabs] == ["Analyze text", "Batch CSV", "Dataset & Results", "About"]


def test_analyze_and_clear_stale_result_when_text_changes(app):
    ui, calls = app
    ui.button(key="analyze_text").click().run()
    assert not ui.exception
    assert calls == [("FinBERT", "cpu")]
    assert "single_result" in ui.session_state
    ui.text_area(key="news_text").set_value("A revised statement about falling profit.").run()
    assert not ui.exception
    assert "single_result" not in ui.session_state


def test_blank_input_does_not_load_a_model(app):
    ui, calls = app
    ui.text_area(key="news_text").set_value("   ").run()
    ui.button(key="analyze_text").click().run()
    assert not ui.exception
    assert calls == []
    assert any("Enter a financial statement" in notice.value for notice in ui.warning)


def test_over_limit_statement_is_rejected(app):
    ui, _ = app
    ui.text_area(key="news_text").set_value("profit " * 512).run()
    ui.button(key="analyze_text").click().run()
    assert not ui.exception
    assert "single_result" not in ui.session_state
    assert any("supports at most 512" in notice.value for notice in ui.warning)


def test_demo_batch_preserves_invalid_row_and_model_change_clears_results(app):
    ui, _ = app
    ui.checkbox(key="use_demo_csv").check().run()
    ui.button(key="analyze_csv").click().run()
    assert not ui.exception
    result = ui.session_state["batch_result"]
    assert len(result) == 8
    assert result["prediction"].isin(["positive", "neutral", "negative"]).sum() == 7
    assert result.loc[result["id"] == "DEMO-08", "error"].fillna("").str.len().iloc[0] > 0
    ui.selectbox(key="model_name").select("TF-IDF baseline").run()
    assert not ui.exception
    assert "batch_result" not in ui.session_state
