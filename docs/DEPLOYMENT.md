# Deployment to Streamlit Community Cloud

The project is hosted at **https://financial-sentiment-akshatog.streamlit.app/** from `akshatog/financial-sentiment-finbert`, branch `main`, entry point `app.py`, using the pinned CPU app dependencies. The account owner completed Streamlit and GitHub authentication.

The original deployment was browser-verified for Prosus FinBERT, the trained conventional baseline and the bundled CSV workflow (7 valid rows plus 1 flagged blank). **The fine-tuned model update requires its own hosted verification.** Follow the current [delivery status](STATUS.md); the earlier check does not verify the new checkpoint.

## Fine-tuned model artifact

Large transformer weights are excluded from Git. When a local verified model is absent, the app uses the `finbert-finetuned-v1` project release. To avoid timeouts on large transfers, that release stores the original ZIP as twenty smaller parts and a pinned `finbert_finetuned_parts.json` manifest. The downloader checks every part, reconstructs the exact original ZIP, checks its checksum, and extracts only expected model files. The original exported archive SHA-256 is:

```text
155ca67e272fbaef5413b4cad7f44237a52a006b9ffee623ddd5013e5a862592
```

Release publication and hosted download/inference must be checked before calling the new live deployment complete. The tokenizer, label mapping and selected checkpoint must remain with the matching weights. A failed download must show a model-load error rather than silently using Prosus or the baseline.

The optional **Prosus FinBERT reference** uses its pinned Hugging Face snapshot. **TF-IDF baseline** uses the small included fitted artifact. No paid model API key is required. Cold starts and resource limits can affect download/load time.

## Deploy and verify

1. Sign in at https://share.streamlit.io/ and select the existing app/repository.
2. Use branch `main`, entry point `app.py`, and the repository's supported Python environment and CPU `requirements.txt`.
3. Confirm the trained artifact is available and its pinned checksum matches the returned model evidence.
4. Launch **Fine-tuned FinBERT** and test a self-authored sentence. Confirm the selected model label, three named scores and 128-token policy.
5. Test the bundled demo CSV, its blank-row validation, and CSV export.
6. Check the baseline and optional Prosus reference without labelling one model's output as another's.
7. Check the public URL and update README/status only with observed behavior. Model weights and dataset records stay out of Git.

If the host cannot fit a transformer, keep the verified local demo and accurately describe the hosted limitation. Do not silently substitute another model.

Official deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
