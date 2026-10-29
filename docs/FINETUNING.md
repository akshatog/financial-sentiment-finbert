# Fine-tuning handoff

The training workflow is prepared. The actual HKUST FinBERT training and final
evaluation remain pending until the GPU notebook finishes. The deployed app
continues to use the existing Prosus reference and trained TF-IDF baseline.

## Run in Colab

1. Open https://colab.research.google.com/ and sign in to your Google account.
2. Choose **File → Upload notebook** and select
   `notebooks/finetune_finbert_colab.ipynb` from this project.
3. Choose **Runtime → Change runtime type → GPU**. Select T4 if offered.
4. Choose **Runtime → Run all**. When the file picker appears, select
   `.cache/colab/finbert_colab_training.zip` from this project. Upload only that ZIP.
5. Allow the GPU check, short training check, full training, and final evaluation
   to finish. Follow progress in the notebook. Do not change settings based on
   test accuracy. If a cell fails, preserve its error message and ask for help.
6. Save the downloaded `finbert_finetuned_results.zip` in `D:\Desktop\CA2_DL`.
   Tell Codex its path so it can verify the evidence and integrate the model.

No API key, paid Colab plan, or GitHub token is needed. Free GPU availability is
not guaranteed, and Colab VM files can disappear; save the result before
disconnecting. Runtime and GPU details are recorded in the training manifest.
See [Colab's resource and lifetime information](https://research.google.com/colaboratory/faq.html).

The notebook and ZIP are a matched pair with checksum verification. To rebuild
them after changing any training source, run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_colab_bundle.py
```

The private ZIP contains the existing licensed dataset splits. Keep it private;
the notebook and training source can be included in the public GitHub repo.
No credentials are included. All authoritative project files stay in this
folder; Colab receives a temporary copy for computation.

## Training and evidence

| Item | Decision |
| --- | --- |
| Starting checkpoint | `yiyanghkust/finbert-pretrain` |
| Pinned revision | `88ab954a39ea6d3ce2b62cff086dd5ad1172c664` |
| Method | Full encoder fine-tuning plus a newly initialized three-class head |
| Label IDs | negative=0, neutral=1, positive=2 |
| Dataset | Existing cleaned Financial PhraseBank 50Agree |
| Training / validation / test | 3,385 / 725 / 726; original saved membership |
| Selection criterion | Best validation macro-F1 |
| Initial settings | 3 epochs, learning rate 2e-5, maximum 128 tokens |
| GPU batch settings | Microbatch 2, accumulation 8; effective batch 16 on one GPU |
| Memory controls | Dynamic padding, FP16, gradient checkpointing |
| Final evaluation | Separate command after model selection; reuse frozen matching results |

Training verifies saved split fingerprints and reads only training/validation
CSV files. The short smoke run performs real optimizer updates, validation, and
checkpoint saving on small training/validation subsets. Its artifacts are marked
as smoke results and cannot be evaluated/exported as a completed classifier.

The final output includes safe model weights, the matching tokenizer, explicit
label mapping, a training manifest, training history, validation scores, file
hashes, test accuracy and macro-F1, per-class scores, and a confusion matrix.
Truncation counts are recorded separately for each used split. Private test
sentences/predictions and resumable optimizer checkpoints are excluded from the
final export ZIP.

HKUST's finance language-model checkpoint is a different release from the
existing `ProsusAI/finbert`, which was already sentiment-fine-tuned on Financial
PhraseBank. The original Prosus score stays a diagnostic with known dataset
exposure. The new classifier can be compared with our TF-IDF baseline on the
same held-out partition. This does not establish zero overlap with HKUST's
large pretraining corpus. Fine-tuning may improve results or overfit; report
the measured results without guaranteeing an improvement.

Sources: [HKUST checkpoint](https://huggingface.co/yiyanghkust/finbert-pretrain),
[authors' fine-tuning notebook](https://github.com/yya518/FinBERT/blob/master/finetune.ipynb),
[Transformers Trainer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).

## Responsibilities

| Who | Work |
| --- | --- |
| Akshat | Google sign-in, GPU selection, upload notebook/bundle, start run, save output ZIP; approve any account prompts personally |
| Codex | Prepare and check code; inspect returned evidence; load model locally; update app, README, presentation, GitHub, and deployment with verified results |

No further dataset preparation is required. Independent fresh-news labels would
provide additional evaluation evidence, but are separate from this training run.

The first model load may report unused pretraining-head weights and newly
initialized classifier/pooler weights. These warnings are expected when adding
the sentiment classifier. A traceback or failed cell should be checked before
continuing. Actual CUDA execution is verified by the notebook's smoke test;
the local tiny CPU test verifies workflow code, not large-model GPU fit or speed.

The current notebook streams the child process's standard output and errors into
each cell. Older saved copies used inherited subprocess output, which may appear
silent in some notebook environments. A running cell without visible progress
does not establish success or failure. Preserve its session and files; inspect
the final result or saved training manifest before deciding whether to retry.

## Local alternative and interruptions

The current app environment has CPU-only PyTorch and must remain separate from
any CUDA training environment. The local RTX 3050 Laptop GPU has 4 GB; full
fine-tuning there requires a short feasibility test and smaller microbatches.
Colab is the selected route because a GPU is available now.

The trainer saves resumable epoch checkpoints under the model's `checkpoints/`
directory. Colab files are ephemeral. For a longer run you may optionally mount
Drive and provide `--checkpoint-dir` pointing to a new empty Drive folder before
starting. Drive access is optional and requires your own account authorization.
If interrupted, preserve existing checkpoint files and ask for help resuming;
do not overwrite a nonempty training output directory. The compact final ZIP
contains inference weights and evidence, not resumable optimizer state.
