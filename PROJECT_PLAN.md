**Financial News Sentiment Analysis Using FinBERT — implementation plan**

This project lives in `D:\Desktop\CA2_DL`. Implementation has completed the local environment, data audit, baseline training, FinBERT inference, text/CSV app, saved evaluation, README, and editable presentation. The complete local test suite passed 25 tests. Public repository creation is authorized and the remote is https://github.com/akshatog/financial-sentiment-finbert. Publication and deployment status are tracked in `docs/STATUS.md`. Human review of the separate source-linked evaluation set remains pending.

The submission deadline is the morning of **30 October 2026, Asia/Kolkata**; the exact cutoff has not been supplied. Required deliverables are a working project, a proper GitHub repository with a strong README, and a presentation. A live deployment is desirable after the required deliverables work.

**Recommended scope.** Use pretrained `ProsusAI/finbert` as the main transformer model. Train a small TF-IDF plus logistic-regression classifier as a separate baseline, and include a majority-class baseline. This gives us an understandable experiment and our own training work while keeping the build manageable. The README and presentation will explicitly identify which model we trained and which model we downloaded. This does not establish compliance with a hypothetical requirement to fine-tune a transformer; that requirement is currently unspecified. Full transformer fine-tuning is an optional extension after the submission essentials are complete.

The application will classify short English financial text as **positive, negative, or neutral**. Labels represent the financial sentiment of the statement toward the described company or event. Mixed, insufficient-context, and nonfinancial examples will be discussed in the label policy and error analysis. The project will not claim to forecast actual stock-price movements. Its syllabus connection is Unit 4: BERT, transformer-based text classification, tokenization, and pretrained-model applications.

**Planning checks already completed.** The workspace was empty and contained no Git repository or local AGENTS.md. The installed Python is 3.13.2; the machine has an NVIDIA GeForce RTX 3050 Laptop GPU with 4,096 MiB of GPU memory. Approximately 67 GiB was free on the project drive at inspection. Package compatibility and usable CUDA support still need to be tested during implementation; the presence of a GPU does not establish either.

| Priority | Deliverable | Definition of done |
| --- | --- | --- |
| Required | Working local app | Text and CSV analysis run end to end with clear validation and downloadable results. |
| Required | Reproducible analysis | Dataset audit, saved split, trained baseline, actual metrics, and explicit FinBERT training-overlap disclosure. |
| Required | GitHub repository | Source, setup instructions, examples, screenshots, evaluation summaries, and presentation are published to the selected repository. |
| Required | Presentation | An editable PPTX explains the problem, dataset, architecture, results, limitations, and live demonstration. |
| Desired | Deployment | A tested live URL works on the hosting environment, with the local demo retained as a fallback. |
| Stretch | Transformer fine-tuning | Attempted only after required deliverables are secure and a small feasibility run succeeds. |

**Phase 0 — Freeze scope and organize the submission.** I will record the decisions above, establish the label policy, and make the requirements traceable to the deliverables. While the rubric is unspecified, the main route remains a pretrained FinBERT application with an independently trained conventional baseline. Environment setup and data work can proceed without waiting for a teacher response. Your contributions are the exact cutoff if known, any subsequently available rubric, GitHub repository visibility/ownership, and the student/team details needed for the slides. Completion gate: we have a shared scope, publication preferences are recorded before publication, and optional work is separated from submission essentials.

**Phase 1 — Set up the local project and verify dependencies.** I will create a project-local virtual environment, source/data/tests/docs folders, dependency files, launch instructions, and Git ignore rules. I will verify compatible package wheels for the installed Python before choosing versions, then run a PyTorch CPU check and test CUDA availability. Python replacement or a system driver installation will only become a user task if a demonstrated incompatibility makes it necessary. Model, package, and training caches will be configured inside the project directory. Completion gate: the isolated environment imports the required libraries and can run a small inference check. No user coding is needed.

**Phase 2 — Acquire and audit Financial PhraseBank.** I will download the source archive, record its source and checksum, retain the raw data unchanged, and parse its ISO-8859-1 files using the final `@` as the sentence/label separator. We will use the 50Agree version as the main dataset: the previous inspection counted 4,846 records, comprising 2,879 neutral, 1,363 positive, and 604 negative. It also found eight repeated sentence occurrences beyond the unique sentence count and no blank sentences. I will check exact and normalized duplicates, conflicting labels, unexpected classes, and sentence lengths. Conflicting duplicates will be isolated with a recorded decision rather than silently given a label.

The 66Agree, 75Agree, and AllAgree versions overlap with 50Agree; we will keep them separate. After duplicate handling, I will create and save a reproducible class-stratified 70/15/15 train/validation/test partition for our trained baseline. Duplicate groups must stay in one partition. Class totals and row membership will be checked. Completion gate: a saved audit, clean canonical data, documented exclusions, and split files with no sentence overlap. Split sizes will be calculated from the cleaned data, not promised from the raw row count.

**Phase 3 — Implement the prediction engine and train the baseline.** I will implement one reusable FinBERT inference module that loads the model once, uses evaluation/inference mode, reads label names from the model configuration, and returns all three model scores. FinBERT and the dataset loader use different numeric label orders, so evaluation will compare named labels or explicitly remapped IDs. The first version will validate short inputs against the tokenizer's actual input limit and explain oversized input instead of silently discarding text. CSV inference will use small batches suitable for the verified device.

I will fit TF-IDF and logistic regression using only the training split, select any settings using validation data, and save the fitted pipeline. No preprocessing or vocabulary fitting will use the held-out test split. A majority-class predictor supplies the simplest reference. Completion gate: individual and batch predictions have the same output schema; label mapping, save/load behavior, blank-input handling, and input-length handling pass focused checks. This phase is our actual baseline training, not FinBERT fine-tuning.

**Phase 4 — Build and verify the Streamlit app.** The interface will include single-text analysis, CSV analysis, and dataset/results views. Text analysis will display the selected class and three model scores. CSV analysis will accept a documented text column, retain row identifiers, flag blank or invalid rows, display summary counts, and export row-level results. The dataset/results view will show class distribution and clearly identify the source of each metric. A simple model selection can expose the trained baseline alongside FinBERT if it remains easy to understand. There will be a configurable small upload/batch limit for the first release. Completion gate: a user can perform the complete text and CSV demo without editing code; scores are described as model scores rather than guaranteed correctness probabilities.

**Phase 5 — Produce honest evaluation and error analysis.** FinBERT was already fine-tuned on Financial PhraseBank. We will therefore identify any PhraseBank scores from that checkpoint as an overlapping-data diagnostic. Our newly trained baseline can have a held-out PhraseBank evaluation, but the two results will not be presented as a fair unseen-data comparison. Accuracy, macro F1, per-class precision/recall/F1, class support, and a confusion matrix will be generated from actual saved predictions. The label imbalance makes a majority baseline and macro F1 useful.

For an additional independent check, I will prepare a small candidate set of approximately 30–60 recent, source-linked financial sentences and a label-review sheet. You will review the labels before seeing model predictions, with uncertain cases flagged. We will check exact and close overlaps against PhraseBank, record sources and the annotation policy, freeze the reviewed set, and evaluate it once. This remains a small student-reviewed test with its limitations stated. If label review cannot be completed, we will retain it as unfinished evaluation work and make no unsupported independent-accuracy claim. The application, README, and slides can still be completed. Completion gate: all reported values match saved outputs, prior training exposure is disclosed, and errors include illustrative neutral, negative, positive, negation, and mixed-information cases.

**Phase 6 — Finish the submission package and local demo.** I will prepare a strong README covering the problem, screenshots, architecture, dataset provenance, setup, commands, training/evaluation procedure, results, limitations, and references. I will add a concise methodology note and runnable scripts; a notebook will be added only if required or if it helps explain the experiment. The PowerPoint will contain roughly 10–12 slides: title/team, problem, objective and Unit 4 connection, data, preprocessing/splits, FinBERT, baseline training, app architecture, actual results, demo, limitations, and references/future work. Charts and screenshots will come from the completed project. I will render and inspect the deck before delivery and include speaker notes or a short viva guide. Your contributions are accurate team/college details and a final rehearsal. Completion gate: the local app launches from the documented command, the demo CSV works, the slides are readable, and the presentation and README agree with the implementation.

**Phase 7 — Publish the GitHub repository.** I will initialize the local repository, review what will be committed, create or use the selected remote, and push the project once the account and visibility are resolved. A suggested repository name is `financial-sentiment-finbert`; the final name will be recorded before creation. The repository will contain source, dependency specifications, scripts, tests, small examples, screenshots, evaluation summaries, documentation, and the final presentation. Ignore rules will keep the local virtual environment, package/model caches, large model weights, and credentials out of Git. Dataset download scripts and provenance will make the project reproducible without committing the whole local environment. Attribution and separate source-code/model/dataset licensing will be documented.

The connected GitHub tools do not currently expose repository creation, while GitHub CLI is installed; the publication step may therefore use the CLI or the browser when needed. You may need to complete GitHub sign-in, an OTP, or a device authorization if no usable account connection exists. You will not need to create files or copy the repository manually if the available authenticated tools can do it. Completion gate: the remote contains the intended project files, repository visibility and reviewer access match the submission plan, and the README commands match the published revision.

**Phase 8 — Deploy if the required deliverables are complete.** The first hosting candidate is Streamlit Community Cloud because it supports deploying a Streamlit app from a GitHub repository. I will prepare deployment-compatible dependencies and model caching, select CPU inference for the hosting environment, inspect build/runtime logs, and test a text request and a small CSV on the live app. Hosting resource limits and first-model-download latency need to be checked rather than assumed. You may need to sign in to the hosting service and authorize GitHub access. Deployment is complete only after a working live URL has been tested. If hosting consumes the remaining submission time, we will deliver the tested local demo and document deployment as pending. No paid hosting or GPU service is part of the default plan.

**Optional extension — Train a compact transformer.** If a later rubric requires transformer fine-tuning, or the required deliverables are complete with enough time remaining, we can train a separately named BERT-family classifier from a base language-model checkpoint on the saved training partition. Its settings will be selected on validation data and its final metrics calculated once on the held-out test set. We will first run a tiny training batch on the 4 GB GPU; sequence length, batch size, and gradient accumulation will be set from measured memory use. A compact DistilBERT model is an alternative if appropriate to the course requirements. We will not call this newly trained FinBERT. External compute would require a separate decision and any account sign-in from you. This extension will not delay the required README, PPTX, or repository publication.

| When | Your part | Why your involvement matters |
| --- | --- | --- |
| Before publication | Confirm repository visibility and ownership; supply an existing repo link if you have one. | These are your publication choices. Local development can proceed while they are unresolved. |
| Before slide finalization | Supply team names, roll numbers, course/section, faculty name, and any required template. | I should not invent academic identity or submission details. |
| During Phase 5 | Review the proposed independent test labels before predictions are revealed. | I can draft and organize them, but my own generated labels do not substitute for independent human review. |
| Only if prompted by the platform | Complete GitHub/hosting sign-in, OTP/device authorization, or an installation approval. | Account authentication and interactive security checks may require you. |
| Before submission | Run through the demo and understand the model/data limitations using the prepared notes. | You will present the work and answer the viva. |
| If subsequently available | Share the actual rubric and exact deadline. | This can change training, slide, and deployment requirements. It does not block the known core work. |

**Execution order and deadline protection.** Complete environment/data/inference first, then the app and baseline evaluation. Human label review can run in parallel with app development. Once outputs are stable, README and presentation preparation can proceed in parallel. Finish and verify the required repository and presentation before spending the remaining time on hosting or extra transformer training. I will report meaningful phase results and blockers; a phase is complete only when its completion gate is satisfied.

**Anticipated project layout.** The actual layout may be simplified during implementation; this is a plan, not a set of folders already created.

```text
CA2_DL/
  PROJECT_PLAN.md
  README.md
  app.py
  requirements.txt
  .gitignore
  src/                 # shared data, inference, and evaluation functions
  scripts/             # download, audit, train baseline, evaluate
  data/                # raw/processed/splits and small demo inputs
  models/              # locally saved baseline; large files ignored as needed
  outputs/             # audit, predictions, metrics, figures
  docs/                # methodology, screenshots, demo/viva notes
  presentation/        # final PPTX and supporting slide assets
  tests/               # focused data/inference/CSV checks
  .venv/               # local only
  .cache/              # local model and package caches
```

**Source references for implementation.**

- [FinBERT model and its training data](https://huggingface.co/ProsusAI/finbert)
- [FinBERT label configuration](https://huggingface.co/ProsusAI/finbert/blob/main/config.json)
- [Financial PhraseBank dataset documentation](https://huggingface.co/datasets/takala/financial_phrasebank)
- [Raw dataset archive](https://huggingface.co/datasets/takala/financial_phrasebank/blob/main/data/FinancialPhraseBank-v1.0.zip)
- [Dataset parser and label mapping](https://huggingface.co/datasets/takala/financial_phrasebank/blob/main/financial_phrasebank.py)
- [PyTorch installation and verification](https://pytorch.org/get-started/locally/)
- [Streamlit installation](https://docs.streamlit.io/get-started/installation)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
