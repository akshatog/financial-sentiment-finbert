# Measured results

The audit retained **4,836 clean sentences** after removing 6 matching duplicate occurrences and quarantining 4 rows in conflicting-label groups. The saved split contains **3,385 training / 725 validation / 726 test** examples, with zero normalized sentence overlap.

### Held-out baseline test

| Model | Accuracy | Macro F1 | Test examples |
| --- | ---: | ---: | ---: |
| Majority class (neutral) | 59.37% | 24.83% | 726 |
| TF-IDF + logistic regression | **75.76%** | **71.94%** | 726 |

Validation selected **C = 2**, with **11,138 TF-IDF features** fitted on training text only. These are results for our independently trained baseline on the saved PhraseBank test partition.

### FinBERT diagnostic with prior training exposure

FinBERT achieved 88.15% accuracy and 87.51% macro F1 on 726 PhraseBank examples. **The checkpoint was already fine-tuned on Financial PhraseBank.** These values are a diagnostic with known dataset-level training exposure, not independent accuracy or a fair unseen-data comparison against the baseline.

Evidence: [dataset audit](../outputs/dataset_audit.json), [training metadata](../models/baseline_metadata.json), [evaluation metrics](../outputs/evaluation.json), and [results detail](RESULTS.md).

## Clean class distribution

| Class | Sentences |
| --- | ---: |
| negative | 604 |
| neutral | 2,871 |
| positive | 1,361 |

## Baseline per-class test metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 70.45% | 68.13% | 69.27% | 91 |
| neutral | 81.78% | 83.29% | 82.53% | 431 |
| positive | 64.82% | 63.24% | 64.02% | 204 |

## Baseline confusion matrix

Rows contain true labels. Columns contain predicted labels.

| Actual / predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 62 | 17 | 12 |
| neutral | 14 | 359 | 58 |
| positive | 12 | 63 | 129 |

The baseline correctly classifies 550 examples and makes 176 errors. Confusions between positive and neutral account for 121 of those errors (68.8%). These aggregate counts describe the errors without claiming their cause.

An independent human-reviewed test remains pending. See the [review protocol](../data/review/README.md). These measured corpus results do not establish performance on future financial news or market returns.
