# Five-minute demo guide

## Before presenting

Run the bootstrap and preparation/training commands in the README. Launch the app from the virtual environment. Open it once while connected to the internet so FinBERT can download and cache its weights. Keep the local app open during the presentation. Have the sample CSV and the PowerPoint available locally.

## Presentation sequence

1. State the task: classify short English financial statements into positive, negative, and neutral.
2. Select FinBERT and analyse: `The company reported higher profit and raised its full-year guidance.` Point out the predicted label and all three model scores.
3. Try: `The company expects an operating loss after demand weakened.` Then: `The company will publish its annual report on Friday.` Explain that dates or factual announcements need not imply positive sentiment.
4. Select the trained baseline and repeat one example. Explain that this project trained the TF-IDF + logistic regression model, while FinBERT uses existing pretrained weights.
5. Upload the demo CSV, analyse it, inspect individual rows, and download the result. A blank row should receive a validation status instead of becoming a sentiment prediction.
6. Open the dataset/results view. Explain neutral-class imbalance and macro F1. Show the saved baseline test result.
7. State the central evaluation caveat: FinBERT already learned from PhraseBank, so its result on this corpus is a diagnostic with possible training overlap.

## Demo fallback

If model download fails, confirm the network connection and leave the app running. If the cached transformer cannot load, demonstrate the saved trained baseline, show the saved evaluation files, and explain the transformer download problem directly. Avoid presenting cached metrics as a fresh successful inference.

If the app does not start, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the baseline artifact is missing, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.train_baseline
```

CSV files need a text column matching the interface's documented selection. Use UTF-8 CSV where possible. Keep the demonstration small enough for the displayed batch limits.

