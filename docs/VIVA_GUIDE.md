# Viva guide

**What problem does this project solve?**
It classifies short English financial statements into positive, negative and neutral sentiment. It does not predict stock prices or returns.

**What did you train yourself?**
We fine-tuned all 109,754,115 parameters of `yiyanghkust/finbert-pretrain` with a new three-class sentiment head. We also trained TF-IDF + logistic regression. We did not pretrain BERT from scratch.

**How was FinBERT trained?**
The fixed training split has 3,385 sentences, validation 725, and test 726. On a Tesla T4, training used 3 epochs, learning rate 2e-5, maximum length 128, microbatch 2, accumulation 8, FP16 and gradient checkpointing. The run completed 636 optimizer updates in about 7 minutes 50 seconds.

**Which optimizer and loss did you use?**
PyTorch AdamW, selected by Hugging Face Trainer with `optim="adamw_torch"`, with learning rate 0.00002 and weight decay 0.01. AdamW uses moving averages of gradients and squared gradients to adapt updates, with weight decay applied separately. Multi-class cross-entropy compares the three raw logits with the correct class ID. Backpropagation computes gradients; AdamW updates parameters. Softmax turns logits into displayed class scores during inference. See the [AdamW documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html).

**What scheduler did you use?**
The installed Trainer's default linear schedule, with `warmup_ratio=0.1`: the learning rate rises during the first 10% of scheduled updates, then decreases linearly. The verified inherited defaults are beta1=0.9, beta2=0.999, epsilon=1e-8 and gradient clipping at norm 1.0. These settings are defaults, not separately tuned choices.

**What does gradient accumulation mean here?**
Each microbatch contains two sentences. Gradients accumulate across eight microbatches before an optimizer update, giving an effective batch of 16 on one GPU for full batches. It reduces the memory needed compared with processing all 16 simultaneously. The last accumulation group in an epoch can be smaller.

**Why FP16, dynamic padding, and gradient checkpointing?**
Mixed precision uses lower precision for suitable operations while retaining training safeguards. Dynamic padding pads only to each batch's required length. Gradient checkpointing recomputes some intermediate activations during backpropagation instead of retaining all of them, trading computation for lower memory use. These controls support the training run; they do not establish greater accuracy.

**Why is the chosen model from epoch 2 if you trained 3 epochs?**
Validation macro-F1 selected checkpoint 424 from epoch 2 at 81.92%. Epoch 3 was slightly weaker and its validation loss increased. The training code reloaded the best validation checkpoint before export; the test did not choose the epoch.

**What results can you claim?**
On the saved held-out PhraseBank test split, fine-tuned FinBERT achieved 85.12% accuracy and 84.46% macro-F1, correctly classifying 618/726 examples. The baseline achieved 75.76% and 71.94%, or 550/726. The gains are 9.37 and 12.52 percentage points on that split, not guarantees about future news.

**Is this the same FinBERT as the original app model?**
No. The new experiment starts from HKUST's financial language-model checkpoint. The original `ProsusAI/finbert` is a separate release already sentiment-fine-tuned on Financial PhraseBank; it remains an optional pretrained reference.

**Why not call the Prosus 88.15% score better than our 85.12%?**
Prosus already used PhraseBank for supervised sentiment training. A new local split cannot undo that prior exposure. Its 88.15% accuracy and 87.51% macro-F1 are a diagnostic, not a fair unseen-data ranking against our models. Unknown overlap with HKUST's language-model pretraining corpus also remains a limitation.

**What proves actual encoder fine-tuning happened?**
The manifest records all parameters trainable, optimizer steps and finite training/validation metrics. The saved encoder query tensor has a different before/after checksum and matches the returned weight file. Training history, selected checkpoint, model hashes and evaluation bindings were verified; the exported model loads locally.

**What is transfer learning?**
We start from a model that has already learned language representations, then update its parameters on labelled financial sentiment examples. This is full fine-tuning with a new classifier, not training a large language model from scratch.

**What does self-attention do?**
It combines information across tokens to form contextual representations. BERT has multiple attention heads and encoder layers; the classification head maps the sentence representation to three logits.

**What is tokenization and why have a limit?**
The matching tokenizer converts text into vocabulary IDs and special tokens, including subword pieces. The fine-tuned workflow used a 128-token limit. No training or test sentence was truncated; one validation sentence was. The app rejects inputs beyond the selected model's supported policy instead of silently cutting them.

**Why read label names from configuration?**
Our IDs are negative=0, neutral=1, positive=2, while Prosus uses a different order. Explicit mappings prevent silently swapping class meanings.

**Why include a conventional baseline?**
TF-IDF + logistic regression shows what a smaller model trained on the same split achieves. Its vocabulary fits on training rows only, and validation selects C=2. That gives the transformer result a meaningful comparison.

**Why report macro-F1 as well as accuracy?**
Neutral is the largest class. Accuracy counts all examples equally, while macro-F1 averages the three class F1 scores equally. This helps expose poor performance on smaller classes.

**How did you avoid data leakage?**
We normalized and grouped duplicates before splitting, quarantined conflicting groups, and saved disjoint train/validation/test membership. Both student models train only on training rows and select settings on validation. Fingerprints bind the final test to the original split. This does not establish time or entity independence or exclude all paraphrases.

**What are the weakest results?**
Positive sentiment has the lowest fine-tuned class F1 at 76.59%. Positive/neutral confusions account for 87 of 108 errors. Those counts identify an error pattern without proving its cause.

**Are the displayed scores confidence guarantees?**
No. They are softmax or classifier scores. We have not calibrated them against independent data.

**What would strengthen the evidence?**
Freeze human labels for new source-linked financial text before revealing model predictions, check overlap, and evaluate once. The provided 30-example review sheet is still unlabelled and is not a completed independent benchmark.

**Can this be used commercially under MIT?**
MIT covers our source code, not third-party data or model weights. Prosus, HKUST and PhraseBank are separate assets with separate terms. Consult the attribution document rather than assuming one license covers everything.
