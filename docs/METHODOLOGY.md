# Methodology

## Task and label policy

Given a short English financial statement, predict `positive`, `negative`, or `neutral` from the financial perspective of the company or event described. A higher profit or improved outlook often supports positive sentiment. A loss or weakened outlook often supports negative sentiment. A factual announcement without an expressed financial effect may be neutral. The entity, comparison period, negation, and surrounding context can change the interpretation.

A single sentence can contain opposing effects. This project still returns one of the three available labels and exposes all three scores. Low confidence, a narrow score margin, or model disagreement can support a human review decision, but no chosen threshold establishes correctness. Scores have not been calibrated against independent data.

## Data provenance

Source: [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank), raw `FinancialPhraseBank-v1.0.zip`, `Sentences_50Agree.txt`. The parser reads ISO-8859-1 text and splits on the final `@` separator. The raw configuration has 4,846 rows with 2,879 neutral, 1,363 positive, and 604 negative labels. The dataset supplies agreement-based subsets rather than an official train/validation/test split.

The preparation step records the source checksum and audit. It detects blank text, unexpected labels, exact duplicates, normalized duplicates, and conflicting labels before creating the canonical dataset. Group normalization uses Unicode NFKC, case folding, and collapsed whitespace. Entire conflicting-label groups enter quarantine. Matching duplicate groups contribute one canonical row. Consult `outputs/` for the resulting counts. The different agreement subsets overlap and cannot act as independent evaluation corpora.

The completed audit retains **4,836 clean sentences**: 604 negative, 2,871 neutral, and 1,361 positive. It quarantines four rows across two conflicting-label groups and removes six repeated occurrences from matching-label groups. The raw audit's eight repeated occurrences includes occurrences in the conflicting groups, so it should not be added to the quarantine count.

## Split and leakage controls

Both student-trained models use the same saved, reproducible, stratified 70/15/15 partition. Data preparation keeps normalized duplicate sentences from crossing partitions. Conflicting duplicate groups require an explicit recorded exclusion or resolution. The saved row membership is the experiment definition, rather than a split re-created during each evaluation.

With seed 42, the saved partitions contain **3,385 training**, **725 validation**, and **726 test** rows. Integer rounding explains their small difference from exact 70/15/15 percentages. The audit records zero normalized sentence overlap and zero row-ID overlap across partitions.

The training pipeline fits the TF-IDF vocabulary and logistic regression on training rows only. Candidate model settings use validation macro F1. The held-out test set supplies final metrics after validation selection; it does not choose either model's settings. The majority predictor chooses the most frequent training label and evaluates on that same test partition.

These controls prevent direct normalized duplicate leakage across our saved partitions. They do not establish temporal independence, entity independence, or absence of close paraphrases. Random-split results should not be read as a guarantee for future news.

## Models and student contribution

**Fine-tuned FinBERT:** start from `yiyanghkust/finbert-pretrain`, immutable revision `88ab954a39ea6d3ce2b62cff086dd5ad1172c664`, attach a new three-class sentiment head, and fine-tune all 109,754,115 parameters. The matching tokenizer and explicit mapping negative=0, neutral=1, positive=2 are saved with the weights. The training script reads only the 3,385 training and 725 validation rows and verifies their canonical fingerprints.

Training used a Tesla T4, 3 epochs, 636 optimizer updates, learning rate 2e-5, microbatch 2, gradient accumulation 8, effective batch 16, FP16, dynamic padding, gradient checkpointing, and maximum length 128. Validation macro-F1 selected epoch 2/checkpoint 424. The final evaluator uses the frozen best checkpoint on 726 test rows, verifies file/data hashes, and reuses a matching saved result instead of silently repeating test inference. No training or test sentence was truncated; one validation sentence was. At inference, the app rejects more than 128 tokens rather than silently truncating input.

The returned archive was verified against its export manifest, model hashes and evaluation bindings. Saved encoder tensor changes match training evidence; the model loads locally. Training evidence and metrics are under `outputs/finetuning/`.

**Prosus reference:** `ProsusAI/finbert` remains an optional pretrained sentiment classifier. This is a different release from the HKUST starting model. We do not retrain Prosus. Its model card records earlier Financial PhraseBank sentiment fine-tuning, so its PhraseBank scores remain a diagnostic with known prior exposure. The pinned Prosus revision is in `outputs/model_manifest.json`.

**Trained baseline:** a scikit-learn pipeline combines TF-IDF features with logistic regression. It is trained by this project's training script. The fitted pipeline stores the vocabulary and classifier together. The generated training report records the candidate settings and validation choice.

**Majority baseline:** always predicts the most common training label. It provides a transparent reference for an imbalanced dataset.

## Evaluation interpretation

Report accuracy, macro F1, per-class precision/recall/F1, class support, and a confusion matrix. Always identify the model, evaluated split, number of examples, and prior training exposure.

For each class, precision is the fraction of that class's predictions that are correct. Recall is the fraction of true examples recovered. F1 is their harmonic mean. Macro F1 averages F1 across labels equally. Accuracy can look relatively strong when a model mainly predicts the dominant neutral class, so both accuracy and macro F1 matter.

**The Prosus PhraseBank score is a diagnostic with prior training exposure.** A new local split does not undo that earlier supervised exposure. It is not a fair unseen-data ranking against the student-trained models. The new HKUST classifier and TF-IDF baseline share the same saved partitions; their held-out comparison is meaningful within that experiment. Absence of sentence overlap with HKUST's large pretraining corpus has not been established. These results do not prove future-news performance or statistical significance.

## Independent evaluation status

An independent test needs financial sentences outside PhraseBank, traceable sources, a documented label policy, and human labels frozen before model predictions are revealed. Check exact and close matches against the existing corpus. Flag uncertain or mixed examples rather than forcing apparent agreement. Evaluate once and report the small sample size and review limits.

The project includes 30 source-linked paraphrased candidates in `data/review/candidates.csv`, with blank `human_label` fields. Follow `data/review/README.md` to review them before predictions. `python -m scripts.evaluate_review` refuses incomplete labels, checks normalized exact overlap against PhraseBank, and freezes the reviewed-file fingerprint. The curated candidates span four company sources, so they have narrow coverage and correlated examples. Source dates and paraphrasing do not alone prove absence from every possible training corpus.

The current project has no completed independently reviewed future-news test. Demo inputs are self-authored behavior checks, not accuracy evidence. The reported 85.12% accuracy applies to the saved PhraseBank partition; it is not a claim about a new human-reviewed news benchmark.

## Unit 4 connection

The application demonstrates matching BERT tokenization, contextual transformer representations, full supervised fine-tuning from a financial language model, validation-based checkpoint selection, and three-class text classification. The trained conventional baseline and recorded leakage controls support interpretation of the deep learning experiment. It is transfer learning, not pretraining a transformer from scratch.

