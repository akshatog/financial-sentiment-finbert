# Five-minute demo guide

## Before presenting

The fine-tuned model is already trained. Launch the app from the project environment and select **Fine-tuned FinBERT** while online so the verified artifact can be cached. Keep the app open. Have the demo CSV, saved metrics and PowerPoint available locally. Do not start a new Colab run during the demonstration.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Presentation sequence

1. State the task: positive, negative or neutral sentiment in a short English financial statement.
2. Select **Fine-tuned FinBERT** and analyse `The company reported higher profit and raised its full-year guidance.` Explain the label and three scores.
3. Try `The company expects an operating loss after demand weakened.` Then `The company will publish its annual report on Friday.` These are self-authored behavior examples, not an accuracy benchmark.
4. Select **TF-IDF baseline** and repeat an example. Explain that both models were trained by this project on the same saved training partition; the transformer learns contextual representations while TF-IDF uses fitted text features.
5. Upload `data/examples/demo_news.csv`, analyse, inspect the blank row's validation status, and download the result. Fine-tuned inputs have a 128-token limit including special tokens.
6. Open dataset/results. Show 3,385/725/726 split sizes, validation-selected epoch 2, and test accuracy **85.12%** / macro-F1 **84.46%**, against baseline **75.76%** / **71.94%**. Explain why macro-F1 matters with class imbalance.
7. If showing **Prosus FinBERT reference**, state its known prior supervised PhraseBank exposure. Its diagnostic is separate from the held-out student-model comparison.
8. Close with limitations: one corpus split, uncalibrated scores, and no completed independent future-news evaluation.

## Fallback

If a transformer download or load fails, show the saved conventional baseline and recorded evidence, explaining the failed operation directly. Do not present saved metrics as a fresh successful prediction or label another checkpoint as the selected model.

CSV files need a selected text column and UTF-8 encoding. The app supports up to 2 MB and 500 rows. If the baseline artifact is missing, the README provides preparation/training commands. The completed transformer does not need retraining to recover a network download.
