# Viva guide

**What problem does this project solve?**  
It classifies the financial sentiment expressed by short English statements. It helps inspect a collection of financial text, but it does not predict stock returns.

**What did you train?**  
The project trains TF-IDF + logistic regression on the saved training split. It uses the pretrained ProsusAI FinBERT checkpoint for transformer inference without further fine-tuning.

**Why is this connected to deep learning?**  
FinBERT is a BERT transformer with a sequence classification head. The app applies its tokenizer, contextual encoder, and three-class output to financial language. The baseline gives an interpretable supervised reference.

**What is tokenization?**  
The tokenizer converts text into model vocabulary IDs and special tokens. BERT may split an unfamiliar word into subword pieces. The tokenizer's length limit counts tokens, not characters or words.

**Why use FinBERT instead of generic sentiment?**  
Financial language uses specialized meanings. An increase in costs and an increase in revenue can imply different financial sentiment. FinBERT's financial adaptation aims to represent that domain language. We must measure performance rather than assume every example will work.

**What does attention do?**  
Self-attention combines information across the tokens in a sequence to construct contextual representations. BERT uses multiple attention heads and encoder layers. The sentence-classification head maps its representation to class logits.

**What are the three displayed scores?**  
FinBERT applies softmax to logits. The baseline uses its classifier's probability outputs. Their largest score selects a label, but these scores are not a measured guarantee of correctness. Calibration is outside the current experiment.

**Why read the model label configuration?**  
Numeric label IDs can use different orders in a model and a dataset. Matching names avoids silently scoring positive predictions as another class.

**Why a conventional baseline?**  
It shows what a smaller model trained by this project can achieve. It also makes the training contribution and held-out evaluation explicit.

**Why is macro F1 useful?**  
Neutral is the largest class. Accuracy can hide poor performance on negative examples. Macro F1 weights each class equally rather than weighting by its size.

**How did you avoid leakage?**  
We audit and handle duplicate groups before partitioning. Saved train, validation, and test membership has no normalized duplicate text across splits. TF-IDF fits only on training rows. Validation selects settings, and the final test supplies the held-out baseline result.

**Is FinBERT's test result independent?**  
No. The published FinBERT checkpoint already uses Financial PhraseBank for fine-tuning. A new local partition cannot remove that earlier exposure. Its PhraseBank metric is an overlapping-data diagnostic, so it is not a fair unseen-data comparison with our baseline.

**Is the baseline test result fully independent of all financial news?**  
It is held out from our baseline training process. It still comes from the same corpus and random sampling. Temporal shifts, repeated entities, and close paraphrases can limit generalization.

**How would you improve the evidence?**  
Collect a new source-linked set of financial sentences, obtain human labels before running the models, check overlap, freeze it, and evaluate once. A larger test over later dates and more markets would support stronger conclusions.

**What are the main failure cases?**  
Mixed statements, negation, missing entity context, nonfinancial language, sarcasm, and inputs beyond the supported token limit. Review a misclassified example and explain its ambiguity instead of assuming the model is always right.

**What would fine-tuning add?**  
A separate experiment could start from a base language-model checkpoint and train a classifier on our saved training rows. Hyperparameters would use validation data. It needs additional compute and a clean distinction from the already fine-tuned FinBERT checkpoint.

**Can this be used commercially under the source code license?**  
The original source uses MIT. That does not cover third-party weights or data. The FinBERT weights and PhraseBank dataset have their own noncommercial Creative Commons terms, recorded in the attribution document.

