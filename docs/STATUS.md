# Delivery status

| Work | Status | Evidence |
| --- | --- | --- |
| Local Python environment | Complete | Project-local .venv; dependency check passes; exact lock saved. |
| PhraseBank audit and splits | Complete | 4,836 clean rows; 4 conflicting rows quarantined; 6 matching duplicates removed; no split overlap. |
| Baseline training | Complete | TF-IDF + logistic regression, validation-selected C=2. |
| FinBERT inference | Complete | Pinned revision, local safetensors download, CPU inference verified. |
| App and CSV workflow | Complete | Browser verified text prediction, upload, batch analysis, and download. |
| Tests | Complete | 25 passed locally; GitHub Actions Linux job passed. Windows Streamlit tests require loopback access. |
| Held-out baseline evaluation | Complete | 726 test examples; accuracy 75.76%, macro F1 71.94%. |
| FinBERT PhraseBank diagnostic | Complete | Accuracy 88.15%, macro F1 87.51%; existing dataset training exposure disclosed. |
| README and presentation | Complete | 12-slide editable PPTX with Akshat Agarwal's supplied details and actual results. |
| Public GitHub repository | Published | https://github.com/akshatog/financial-sentiment-finbert |
| Independent human-reviewed check | Pending reviewer | Fill human_label in data/review/candidates.csv before predictions. |
| Live deployment | Complete and browser verified | https://financial-sentiment-akshatog.streamlit.app/ — FinBERT and baseline predictions work; hosted demo CSV retains 8 rows, with 7 analyzed and 1 flagged. |

The main model is pretrained FinBERT. Student training applies to the conventional baseline. Full transformer fine-tuning is an optional extension, not an implemented feature.

The published project passed its [GitHub Actions checks](https://github.com/akshatog/financial-sentiment-finbert/actions/runs/40775828023). The live application uses Python 3.13 and the pinned CPU dependencies. First model loading and hosting cold starts may take longer than cached requests.
