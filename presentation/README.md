# Presentation

`financial_sentiment.pptx` is the editable 12-slide submission deck. It contains native text, tables, and charts. Its numbers come from the generated audit and evaluation JSON files in `outputs/`.

The demo slide includes the verified [live application](https://financial-sentiment-akshatog.streamlit.app/) as a clickable link.

The deck covers the task, model roles, dataset imbalance, cleaning and partitions, BERT concepts, baseline training, app demo, held-out baseline metrics, errors, FinBERT training overlap, and limitations. Speaker notes include source references and presenter context.

The project trains the TF-IDF + logistic regression baseline. It does not train or fine-tune FinBERT. A PhraseBank result for the published checkpoint is a diagnostic with prior training exposure.

The deck builder is `build_deck.mjs`. It uses JavaScript and `@oai/artifact-tool`, reads the current evidence files, and exports slide previews to the ignored `.cache/presentation/` folder before finalization. It needs the Codex bundled authoring runtime or an explicitly configured `ARTIFACT_TOOL_MODULE` path. The application itself does not require this authoring dependency.
