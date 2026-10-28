# Delivery status

| Work | Status | Evidence |
| --- | --- | --- |
| Local Python environment | Complete | Project-local .venv; dependency check passes; exact lock saved. |
| PhraseBank audit and splits | Complete | 4,836 clean rows; 4 conflicting rows quarantined; 6 matching duplicates removed; no split overlap. |
| Baseline training | Complete | TF-IDF + logistic regression, validation-selected C=2. |
| FinBERT inference | Complete | Pinned revision, local safetensors download, CPU inference verified. |
| App and CSV workflow | Complete | Browser verified text prediction, upload, batch analysis, and download. |
| Tests | Complete | 25 passed; Windows Streamlit tests require loopback access. |
| Held-out baseline evaluation | Complete | 726 test examples; accuracy 75.76%, macro F1 71.94%. |
| FinBERT PhraseBank diagnostic | Complete | Accuracy 88.15%, macro F1 87.51%; existing dataset training exposure disclosed. |
| README and presentation | Complete | 12-slide editable PPTX with Akshat Agarwal's supplied details and actual results. |
| Public GitHub repository | Published | https://github.com/akshatog/financial-sentiment-finbert |
| Independent human-reviewed check | Pending reviewer | Fill human_label in data/review/candidates.csv before predictions. |
| Live deployment | Awaiting Streamlit sign-in and terms acceptance | Hosting opened at share.streamlit.io; local app works at http://127.0.0.1:8501 while its server is running. |

The main model is pretrained FinBERT. Student training applies to the conventional baseline. Full transformer fine-tuning is an optional extension, not an implemented feature.
