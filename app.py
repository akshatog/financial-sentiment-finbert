"""Local Streamlit interface for the financial sentiment project.

Model resources are loaded only after a user requests analysis. All inference,
CSV validation, and safe export behavior lives in the reusable ``src`` modules.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.batch import analyze_dataframe, export_csv, read_csv_file
from src.config import LABELS, MODEL_ID, MODEL_REVISION, OUTPUT_DIR, ROOT, setup_cache
from src.inference import ModelInputError, get_predictor


MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_CSV_ROWS = 500
SAMPLES = {
    "Positive · stronger earnings": (
        "The company reported higher quarterly profit and raised its full-year "
        "revenue outlook after strong demand for its products."
    ),
    "Neutral · scheduled announcement": (
        "The company will publish its quarterly financial report on Thursday "
        "before the market opens."
    ),
    "Negative · weaker demand": (
        "The manufacturer reported a decline in operating profit as weaker "
        "demand and rising costs reduced its margins."
    ),
    "Mixed · a useful edge case": (
        "Revenue increased compared with last year, but operating profit fell "
        "as higher production costs offset the sales growth."
    ),
}
COLORS = {"positive": "#15725d", "neutral": "#64716c", "negative": "#a34838"}


st.set_page_config(
    page_title="Financial Sentiment Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
setup_cache()


@st.cache_resource(show_spinner=False)
def load_predictor(model_name: str, device: str):
    """Cache a model per explicit selection; importing the app downloads nothing."""
    return get_predictor(model_name, device=device)


def clear_results() -> None:
    for key in ("single_result", "batch_result", "batch_result_signature"):
        st.session_state.pop(key, None)


def clear_single_result() -> None:
    st.session_state.pop("single_result", None)


def choose_sample() -> None:
    selected = st.session_state.get("sample_picker")
    if selected in SAMPLES:
        st.session_state["news_text"] = SAMPLES[selected]
    clear_single_result()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def section_label(number: str, label: str) -> None:
    st.markdown(f'<div class="section-label">{number} / {label}</div>', unsafe_allow_html=True)


def explain_model_error(error: Exception) -> None:
    st.error(f"Analysis could not run: {error}")
    st.caption(
        "For FinBERT, the first run needs an internet connection to download the "
        "pinned model. For the baseline, run `python scripts/train_baseline.py` "
        "from the project folder. See the README for setup and recovery steps."
    )


def render_prediction(result: dict[str, Any], model_name: str) -> None:
    label = str(result["label"]).lower()
    safe_label = html.escape(label.capitalize())
    color = COLORS.get(label, COLORS["neutral"])
    st.markdown(
        f'<div class="result-label" style="color:{color}">{safe_label}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{model_name} · highest model score {float(result['score']):.1%}")
    for name in ("positive", "neutral", "negative"):
        value = max(0.0, min(1.0, float(result[name])))
        label_col, score_col = st.columns([4, 1])
        label_col.markdown(f"**{name.capitalize()}**")
        score_col.markdown(f'<div class="score-value">{value:.1%}</div>', unsafe_allow_html=True)
        st.progress(value)
    st.caption(
        "These are the model’s scores across three classes. A high score is not "
        "a guarantee that the label is correct."
    )


def extract_counts(audit: dict[str, Any]) -> dict[str, int] | None:
    """Accept audit variants while never substituting assumed dataset counts."""
    for key in ("canonical_class_counts", "clean_class_counts", "class_counts", "label_counts", "raw_class_counts"):
        candidate = audit.get(key)
        if isinstance(candidate, dict) and all(name in candidate for name in LABELS):
            return {name: int(candidate[name]) for name in LABELS}
    for key in ("main_dataset", "canonical", "cleaned", "clean", "raw", "dataset", "summary"):
        nested = audit.get(key)
        if isinstance(nested, dict):
            found = extract_counts(nested)
            if found:
                return found
    return None


def metric_summary(payload: dict[str, Any]) -> dict[str, float]:
    candidate = payload.get("metrics", payload)
    if not isinstance(candidate, dict):
        return {}
    values: dict[str, float] = {}
    accuracy = candidate.get("accuracy")
    if isinstance(accuracy, (int, float)):
        values["Accuracy"] = float(accuracy)
    macro = candidate.get("macro_f1", candidate.get("f1_macro"))
    if macro is None and isinstance(candidate.get("macro avg"), dict):
        macro = candidate["macro avg"].get("f1-score")
    report = candidate.get("classification_report", candidate.get("report", {}))
    if macro is None and isinstance(report, dict) and isinstance(report.get("macro avg"), dict):
        macro = report["macro avg"].get("f1-score")
    if isinstance(macro, (int, float)):
        values["Macro F1"] = float(macro)
    return values


def render_metrics(payload: dict[str, Any], label: str) -> None:
    st.markdown(f"#### {label}")
    values = metric_summary(payload)
    if values:
        metric_columns = st.columns(len(values))
        for column, (name, value) in zip(metric_columns, values.items()):
            column.metric(name, f"{value:.3f}")
    else:
        st.caption("No summary scores are available in this output yet.")
    candidate = payload.get("metrics", payload)
    if isinstance(candidate, dict) and isinstance(candidate.get("n_examples"), int):
        st.caption(f"{candidate['n_examples']:,} evaluated statements · Scores range from 0 to 1.")
    report = candidate.get("classification_report", candidate.get("report")) if isinstance(candidate, dict) else None
    if isinstance(report, dict):
        class_rows = {name: report[name] for name in LABELS if isinstance(report.get(name), dict)}
        if class_rows:
            st.dataframe(pd.DataFrame.from_dict(class_rows, orient="index"), width="stretch")
    confusion = candidate.get("confusion_matrix") if isinstance(candidate, dict) else None
    if isinstance(confusion, dict) and isinstance(confusion.get("counts"), list):
        labels = confusion.get("labels", list(LABELS))
        with st.expander(f"Confusion matrix · {label}"):
            st.caption("Rows are true labels; columns are predicted labels.")
            st.dataframe(pd.DataFrame(confusion["counts"], index=labels, columns=labels), width="stretch")
    with st.expander(f"Inspect saved {label.lower()} output"):
        st.json(payload)


st.markdown(
    """
    <style>
    :root { --ink:#142d2b; --muted:#64716c; --line:#dce3dc; --teal:#15725d; }
    .stApp { background:#f6f5ef; color:var(--ink); }
    .block-container { max-width:1190px; padding-top:2.25rem; padding-bottom:1.5rem; }
    [data-testid="stSidebar"] { background:#eaf0e9; border-right:1px solid #d7e0d7; }
    [data-testid="stSidebar"] .block-container { padding-top:2.1rem; }
    h1,h2,h3,h4 { color:var(--ink); letter-spacing:-.035em; }
    h1 { font-family:Georgia,'Times New Roman',serif; font-weight:500; font-size:3.1rem !important; line-height:1.12; padding-top:.2rem; }
    h3 { font-size:1.22rem !important; font-weight:650; }
    h4 { font-size:1.05rem !important; }
    .brand { display:flex; align-items:center; gap:.65rem; margin-bottom:1.6rem; }
    .brand-icon { width:31px; height:31px; display:grid; place-items:center; background:var(--ink); color:#f6f5ef; font-size:22px; border-radius:7px; }
    .brand-name { font-size:.77rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; }
    .eyebrow { color:var(--teal); letter-spacing:.14em; font-weight:750; font-size:.7rem; text-transform:uppercase; }
    .hero-deck { font-size:1.03rem; color:#63706a; max-width:650px; margin:.15rem 0 1.7rem; line-height:1.65; }
    .section-label { color:var(--teal); font-size:.68rem; letter-spacing:.12em; font-weight:750; margin:1.55rem 0 .85rem; text-transform:uppercase; }
    .result-label { font-family:Georgia,'Times New Roman',serif; font-size:3.3rem; line-height:1.12; margin:.5rem 0 .35rem; }
    .score-value { text-align:right; font-variant-numeric:tabular-nums; font-size:.88rem; }
    .quiet-note { color:var(--muted); font-size:.82rem; line-height:1.6; }
    .side-footer { border-top:1px solid #cfd9cf; margin-top:2rem; padding-top:1rem; }
    .app-footer { border-top:1px solid var(--line); color:var(--muted); margin-top:2rem; padding-top:1rem; font-size:.74rem; }
    [data-testid="stTabs"] [role="tablist"] { gap:1.8rem; border-bottom:1px solid var(--line); }
    [data-testid="stTabs"] [role="tab"] { padding-left:0; padding-right:0; font-size:.88rem; }
    [data-testid="stProgressBar"] { margin-top:-.6rem; margin-bottom:.65rem; }
    [data-testid="stMetricValue"] { font-size:1.65rem; font-weight:550; }
    [data-testid="stTextArea"] textarea { background:#fffefa; font-size:1rem; line-height:1.65; }
    [data-testid="stButton"] button[kind="primary"] { padding:.6rem 1.2rem; border-radius:7px; }
    @media (max-width:760px) { h1 { font-size:2.3rem !important; } .block-container { padding-top:1.4rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">◈</div>'
        '<div class="brand-name">Financial<br>Sentiment Lab</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Model workspace</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Analysis model", ["FinBERT", "TF-IDF baseline"], key="model_name", on_change=clear_results
    )
    if model_name == "FinBERT":
        st.caption("A pretrained BERT transformer adapted to financial language by ProsusAI.")
    else:
        st.caption("Our trained TF-IDF + logistic regression model, fitted on the saved training split.")
    with st.expander("Runtime settings"):
        runtime = st.selectbox(
            "Inference device", ["CPU", "Auto"], key="runtime", on_change=clear_results,
            help="CPU is the portable default. Auto uses an available accelerator when supported.",
        )
        st.caption("Models load on the first analysis request and are reused in this session.")
    device = "cpu" if runtime == "CPU" else "auto"
    st.divider()
    st.markdown("**Three possible labels**")
    st.caption("Positive · Neutral · Negative")
    st.markdown(
        '<div class="side-footer quiet-note">A Unit 4 application of BERT, tokenization, '
        'and transformer text classification.<br><br>Student baseline training is separate '
        'from the pretrained FinBERT checkpoint.</div>', unsafe_allow_html=True,
    )

st.markdown('<div class="eyebrow">Financial language · Natural language processing</div>', unsafe_allow_html=True)
st.title("The sentiment behind the statement.")
st.markdown(
    '<div class="hero-deck">Explore the tone of financial news with FinBERT and a student-trained '
    'baseline. Three labels, transparent scores, and room for context.</div>', unsafe_allow_html=True,
)

analyze_tab, batch_tab, results_tab, about_tab = st.tabs(
    ["Analyze text", "Batch CSV", "Dataset & Results", "About"]
)

with analyze_tab:
    input_column, output_column = st.columns([1.3, 1], gap="large")
    with input_column:
        section_label("01", "Your statement")
        st.selectbox(
            "Load a demo statement", ["Choose a sample…", *SAMPLES],
            key="sample_picker", on_change=choose_sample,
        )
        st.session_state.setdefault("news_text", next(iter(SAMPLES.values())))
        news_text = st.text_area(
            "Financial news text", key="news_text", height=205,
            on_change=clear_single_result,
            help="Use a short English financial statement. Over-limit inputs are rejected; text is never silently shortened.",
        )
        st.caption(f"{len(news_text.split()):,} words · {len(news_text):,} characters · English input")
        analyze_clicked = st.button("Analyze sentiment", type="primary", key="analyze_text")
        if analyze_clicked:
            clear_single_result()
            if not news_text.strip():
                st.warning("Enter a financial statement before analyzing.")
            else:
                try:
                    with st.spinner(f"Analyzing with {model_name}…"):
                        predictor = load_predictor(model_name, device)
                        length = predictor.token_lengths([news_text])[0]
                        if predictor.max_tokens is not None and length > predictor.max_tokens:
                            raise ModelInputError(
                                f"This text has {length:,} tokens; {model_name} supports at most "
                                f"{predictor.max_tokens:,}. Shorten the statement and try again."
                            )
                        prediction = predictor.predict([news_text])[0]
                    st.session_state["single_result"] = {
                        "prediction": prediction, "text": news_text,
                        "model": model_name, "device": device,
                    }
                except ModelInputError as error:
                    st.warning(str(error))
                except Exception as error:
                    explain_model_error(error)
    with output_column:
        section_label("02", "Model reading")
        stored = st.session_state.get("single_result")
        if stored and stored["text"] == news_text and stored["model"] == model_name and stored["device"] == device:
            render_prediction(stored["prediction"], model_name)
        else:
            st.markdown("### Language has a point of view.")
            st.write(
                "Choose a sample or paste a short financial statement, then run an analysis. "
                "The result shows its strongest label alongside all three model scores."
            )
            st.divider()
            st.markdown("**Read the wording and the context together.**")
            st.caption(
                "Mixed news, negation, unfamiliar names, and missing context can change the reading. "
                "Sentiment describes the statement; it does not forecast a stock’s price."
            )
    st.caption("Demo statements are self-authored illustrations. Their sample names are not evaluation ground truth.")

with batch_tab:
    section_label("01", "A file of statements")
    st.markdown("### Bring your news into one view.")
    st.write("Upload a CSV, choose its text column, and keep every source row in the exported results.")
    upload_col, sample_col = st.columns([2, 1], gap="large")
    with upload_col:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="csv_upload")
    with sample_col:
        st.markdown("**Start with a small example**")
        demo_path = ROOT / "data" / "examples" / "demo_news.csv"
        if demo_path.exists():
            st.download_button("Download demo CSV", demo_path.read_bytes(), "demo_news.csv", "text/csv")
        use_demo = st.checkbox("Use the bundled demo CSV", key="use_demo_csv")
        st.caption("Up to 2 MB and 500 rows. Invalid or over-limit statements are flagged in place.")
    content = demo_path.read_bytes() if use_demo and demo_path.exists() else uploaded.getvalue() if uploaded else None
    if content is None:
        st.session_state.pop("batch_result", None)
        st.session_state.pop("batch_result_signature", None)
        st.info("Your CSV needs one column containing financial text. Other columns are preserved.")
    elif len(content) > MAX_CSV_BYTES:
        clear_results()
        st.error("This file exceeds the 2 MB limit. Upload a smaller CSV.")
    else:
        try:
            frame = read_csv_file(content)
            if len(frame) > MAX_CSV_ROWS:
                raise ValueError(f"This file contains {len(frame):,} rows. The limit is {MAX_CSV_ROWS}; split the file and retry.")
            if frame.empty:
                raise ValueError("This CSV has no data rows. Add at least one financial statement.")
            preferred_index = list(frame.columns).index("text") if "text" in frame.columns else 0
            text_column = st.selectbox("Column containing the news text", list(frame.columns), index=preferred_index)
            signature = hashlib.sha256(content + f"|{text_column}|{model_name}|{device}".encode()).hexdigest()
            if st.session_state.get("batch_result_signature") != signature:
                st.session_state.pop("batch_result", None)
            source_name = "bundled demo CSV" if use_demo else uploaded.name
            st.caption(f"Source: {source_name} · {len(frame):,} rows · Column: {text_column} · Model: {model_name}")
            with st.expander("Preview source rows", expanded=False):
                st.dataframe(frame.head(10), hide_index=True, width="stretch")
            if st.button("Analyze CSV", type="primary", key="analyze_csv"):
                st.session_state.pop("batch_result", None)
                try:
                    with st.spinner(f"Analyzing {len(frame):,} rows with {model_name}…"):
                        predictor = load_predictor(model_name, device)
                        batch_result = analyze_dataframe(frame, text_column, predictor, batch_size=8)
                    st.session_state["batch_result"] = batch_result
                    st.session_state["batch_result_signature"] = signature
                except Exception as error:
                    explain_model_error(error)
            batch_result = st.session_state.get("batch_result")
            if batch_result is not None and st.session_state.get("batch_result_signature") == signature:
                section_label("02", "Batch reading")
                valid = batch_result["prediction"].isin(LABELS)
                if "error" in batch_result:
                    valid &= batch_result["error"].fillna("").astype(str).str.strip().eq("")
                total_col, valid_col, error_col = st.columns(3)
                total_col.metric("Source rows", len(batch_result))
                valid_col.metric("Analyzed", int(valid.sum()))
                error_col.metric("Needs attention", int((~valid).sum()))
                if (~valid).any():
                    st.warning("Some rows could not be analyzed. They remain in the results with an error message.")
                counts = batch_result.loc[valid, "prediction"].value_counts().reindex(LABELS, fill_value=0)
                chart_col, counts_col = st.columns([2, 1], gap="large")
                with chart_col:
                    st.bar_chart(counts.rename("Statements"), color="#15725d", horizontal=True, height=180)
                with counts_col:
                    st.dataframe(counts.rename("Statements"), width="stretch")
                st.dataframe(batch_result, hide_index=True, width="stretch")
                st.download_button(
                    "Download results CSV", export_csv(batch_result), "sentiment_results.csv", "text/csv",
                    type="primary", key="download_results",
                )
                st.caption("The export includes every row, status, error details, the predicted label, and all three model scores.")
        except (ValueError, UnicodeError, pd.errors.ParserError) as error:
            st.session_state.pop("batch_result", None)
            st.error(str(error))

with results_tab:
    section_label("01", "Dataset provenance")
    st.markdown("### Evidence, with its boundaries.")
    st.write(
        "Financial PhraseBank contains short financial statements labeled positive, neutral, or negative. "
        "This project audits the 50%-agreement version and saves separate train, validation, and test splits."
    )
    audit = read_json(OUTPUT_DIR / "dataset_audit.json")
    evaluation = read_json(OUTPUT_DIR / "evaluation.json")
    diagnostic = read_json(OUTPUT_DIR / "finbert_diagnostic.json")
    if not diagnostic and evaluation and isinstance(evaluation.get("finbert"), dict):
        if evaluation["finbert"].get("status") == "evaluated":
            diagnostic = evaluation["finbert"]
    if audit:
        main_dataset = audit.get("main_dataset", {})
        partitions = audit.get("splits", {}).get("partitions", {})
        if isinstance(main_dataset, dict) and main_dataset.get("clean_rows") is not None:
            raw_col, clean_col, overlap_col = st.columns(3)
            raw_col.metric("Raw statements", f"{main_dataset.get('raw_rows', main_dataset.get('rows', 0)):,}")
            clean_col.metric("After duplicate audit", f"{main_dataset['clean_rows']:,}")
            overlap_col.metric("Sentence overlap across splits", audit.get("splits", {}).get("normalized_sentence_overlap", "—"))
            if all(name in partitions for name in ("train", "validation", "test")):
                st.caption("Saved split sizes: " + " · ".join(
                    f"{name} {partitions[name]['rows']:,}" for name in ("train", "validation", "test")
                ))
        counts = extract_counts(audit)
        if counts:
            distribution_col, context_col = st.columns([1.2, 1], gap="large")
            with distribution_col:
                st.bar_chart(pd.Series(counts, name="Statements"), color="#15725d", horizontal=True, height=205)
            with context_col:
                st.markdown("**Three classes, uneven support.**")
                st.caption("Distribution shown from the saved audit. Class imbalance makes macro F1 useful alongside accuracy.")
                st.caption("Duplicate handling and split membership are recorded in the project outputs.")
        with st.expander("Inspect saved dataset audit"):
            st.json(audit)
    else:
        st.info("Dataset audit is not available yet. Run the data preparation script listed in the README.")
    section_label("02", "Two different kinds of evidence")
    baseline_col, finbert_col = st.columns(2, gap="large")
    with baseline_col:
        st.markdown("#### Student-trained baseline")
        st.caption("HELD-OUT PHRASEBANK TEST SPLIT")
        st.write("TF-IDF and logistic regression are fitted on training data. Validation selects settings; the saved test split measures held-out performance.")
        if evaluation:
            baseline_payload = evaluation.get("baseline", evaluation.get("tfidf_logistic_regression", evaluation))
            if isinstance(baseline_payload, dict):
                render_metrics(baseline_payload, "TF-IDF + logistic regression")
            majority = evaluation.get("majority_baseline", evaluation.get("majority"))
            if isinstance(majority, dict):
                render_metrics(majority, "Majority-class reference")
        else:
            st.info("Baseline evaluation has not been generated yet.")
    with finbert_col:
        st.markdown("#### Pretrained FinBERT")
        st.caption("TRAINING-OVERLAP DIAGNOSTIC")
        st.warning("FinBERT was fine-tuned on Financial PhraseBank. Its scores on this dataset do not establish unseen-data performance.")
        if diagnostic:
            render_metrics(diagnostic, "FinBERT diagnostic")
        else:
            st.info("No FinBERT PhraseBank diagnostic is available yet.")
    st.divider()
    st.markdown("**Independent evaluation: pending human label review.**")
    st.caption("A source-linked review set can support a separate check after labels are reviewed and frozen. No independent accuracy is claimed here.")

with about_tab:
    section_label("01", "From sentence to sentiment")
    st.markdown("### A focused application of Unit 4.")
    st.write("This project connects BERT, tokenization, transfer learning, and multiclass text classification to financial language.")
    first, second, third = st.columns(3, gap="large")
    with first:
        st.markdown("#### 1. Tokenize")
        st.write("FinBERT converts a statement into subword tokens and adds the special tokens expected by BERT.")
    with second:
        st.markdown("#### 2. Read in context")
        st.write("Transformer attention builds context-aware representations across the statement.")
    with third:
        st.markdown("#### 3. Classify")
        st.write("A classification head scores positive, neutral, and negative sentiment. The largest score determines the displayed label.")
    st.divider()
    st.markdown("#### What we trained")
    st.write("The student-trained model is TF-IDF plus logistic regression. FinBERT is the downloaded ProsusAI checkpoint; this project does not claim to have trained or fine-tuned it.")
    st.markdown("#### Read these labels carefully")
    st.markdown(
        "- **Positive:** the wording describes a favorable financial development.\n"
        "- **Negative:** the wording describes an unfavorable financial development.\n"
        "- **Neutral:** the wording is informational or lacks a clear financial direction.\n"
        "- Mixed statements, sarcasm, and missing context can produce ambiguous or incorrect labels."
    )
    st.caption("The app is designed for short English financial statements. Long inputs are flagged, and no sentence is silently truncated.")
    st.markdown("#### Reproducibility & sources")
    st.code(f"Model: {MODEL_ID}\nRevision: {MODEL_REVISION}\nDefault device: cpu", language=None)
    st.markdown(
        "[FinBERT model card](https://huggingface.co/ProsusAI/finbert) · "
        "[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) · "
        "[BERT paper](https://arxiv.org/abs/1810.04805) · "
        "[Streamlit documentation](https://docs.streamlit.io/)"
    )

st.markdown(
    '<div class="app-footer">Financial Sentiment Lab &nbsp; / &nbsp; Academic demonstrator '
    '&nbsp; · &nbsp; Sentiment describes language. Financial decisions require broader evidence.</div>',
    unsafe_allow_html=True,
)
