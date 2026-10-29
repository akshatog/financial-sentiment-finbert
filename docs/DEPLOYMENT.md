# Deployment to Streamlit Community Cloud

The live app is deployed at **https://financial-sentiment-akshatog.streamlit.app/** from `akshatog/financial-sentiment-finbert`, branch `main`, entry point `app.py`, using Python 3.13. The account owner completed Streamlit and GitHub authentication.

Browser verification succeeded for FinBERT and the trained baseline. The hosted demo CSV produced 7 valid predictions and retained its 1 blank row with an error status. The download control was available. The full uploaded-file/export workflow was also exercised locally, with the exported file saved as `data/examples/demo_predictions.csv`. GitHub Actions installed the pinned CPU dependencies and passed all tests on Linux.

To reproduce or manage the deployment:

1. Sign in at https://share.streamlit.io/ using the intended account.
2. Choose Create app and select the existing GitHub repository: `akshatog/financial-sentiment-finbert`.
3. Use branch `main`, main file `app.py`, and Python 3.13 in advanced settings when available.
4. Deploy with the repository's pinned CPU dependencies in `requirements.txt`.
5. The first FinBERT request downloads the pinned public checkpoint. No paid model API key is required. Allow time for its approximately 438 MB inference weights to load.
6. Test a single sentence and the bundled demo CSV. Verify that the blank demo row remains flagged and the CSV export works.
7. Test the live URL after any dependency or application change and keep README.md and docs/STATUS.md accurate.

The repository includes the small fitted conventional baseline, so that option can run without downloading transformer weights. Full PhraseBank records, the local Python environment, and downloaded FinBERT weights stay outside Git; the download/preparation scripts reproduce them. The current Linux deployment has been tested; cold-start latency and hosting resource availability can vary over time.

If the host cannot fit FinBERT within its resources, retain the verified local FinBERT demo and document any baseline-only deployment accurately. A live app must not silently replace FinBERT with another model.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
