# Presentation

`financial_sentiment_revised.pptx` is the current editable six-slide submission deck. It follows the user-supplied reference's navy/teal design, typography, workflow diagram, result panels, comparison tables, and conclusion layout. Its numbers come from the audit and evaluation JSON files in `outputs/`. The earlier extended deck remains available as `financial_sentiment.pptx`.

The conclusion slide includes the verified [live application](https://financial-sentiment-akshatog.streamlit.app/) as a clickable link.

The six slides cover the project overview, transformer workflow, three model routes, evaluation results, model choice and training exposure, and conclusion with Unit 4 mapping. Speaker notes include source references, data preparation details, and presenter context.

The project trains the TF-IDF + logistic regression baseline. It does not train or fine-tune FinBERT. A PhraseBank result for the published checkpoint is a diagnostic with prior training exposure.

The current builder is `build_reference_deck.mjs`. It uses JavaScript and `@oai/artifact-tool`, edits a copy of the supplied reference, reads current evidence files, and exports slide previews to the ignored `.cache/presentation/` folder before validation. It takes the original reference PPTX path as its first argument (or `REFERENCE_DECK`) and needs the Codex bundled authoring runtime, optionally selected by `CODEX_ARTIFACT_RUNTIME`. The friend's source file is not redistributed in this repository. `build_deck.mjs` builds the earlier extended version. The application itself does not require the presentation authoring dependency.
