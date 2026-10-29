# Delivery status

| Work | Status | Evidence |
| --- | --- | --- |
| Local CPU app environment | Complete | Project-local .venv and pinned dependencies; separate from Colab GPU training. |
| PhraseBank audit and split | Complete | 4,836 clean rows; 3,385/725/726 partitions; no normalized sentence overlap. |
| Baseline training and evaluation | Complete | Validation-selected C=2; test accuracy 75.76%, macro-F1 71.94%. |
| Full student FinBERT fine-tuning | Complete and verified | HKUST finance LM + new head, all 109,754,115 parameters trainable; Tesla T4, 3 epochs, 636 updates. |
| Fine-tuned checkpoint selection | Complete | Epoch 2 / checkpoint 424, validation macro-F1 81.92%. |
| Fine-tuned held-out evaluation | Complete | 618/726 correct; accuracy 85.12%, macro-F1 84.46%; one recorded test evaluation. |
| Returned model verification | Complete | Export/model/evaluation hashes and original splits match; encoder tensor change verified; local CPU inference passes. |
| Original Prosus diagnostic | Complete | Accuracy 88.15%, macro-F1 87.51%; known prior PhraseBank training exposure disclosed. |
| Fine-tuned app integration | Complete locally | Student checkpoint is the default; 128-token limit; verified download and fail-closed model selection. |
| Submission presentation | Complete | Six editable slides in presentation/financial_sentiment_finetuned.pptx with actual fine-tuning and test results. |
| Current local checks | Passed | 37 passed, 1 optional training-dependency test skipped; the tiny Trainer integration was verified separately during workflow preparation. |
| Public GitHub repository | Published | https://github.com/akshatog/financial-sentiment-finbert |
| Independent human-reviewed check | Pending reviewer | Label data/review/candidates.csv before predictions. |
| Existing live deployment | Original models verified | https://financial-sentiment-akshatog.streamlit.app/; fine-tuned release verification pending. |

The transformer training is complete. A new Colab run is not needed. The selected model improves over the baseline by 9.37 percentage points accuracy and 12.52 points macro-F1 on the saved test split. These corpus results do not establish performance on future financial news.

The original live app was browser-verified for Prosus FinBERT, the conventional baseline, and its CSV workflow. That earlier check does not verify the new fine-tuned deployment. Update the app/presentation and hosted verification rows only after those checks finish.

Evidence: [training manifest](../outputs/finetuning/training_manifest.json), [history](../outputs/finetuning/training_history.json), [evaluation](../outputs/finetuning/evaluation.json), and [import verification](../outputs/finetuning/import_verification.json).
