# Presentation and viva rehearsal guide

Use the current [six-slide presentation](../presentation/financial_sentiment_finetuned.pptx). Keep the [live app](https://financial-sentiment-akshatog.streamlit.app/), [GitHub repository](https://github.com/akshatog/financial-sentiment-finbert), and saved evidence ready. Aim for approximately three to five minutes of slides, followed by a two-minute demonstration. This guide can be read offline; the app link needs a connection.

## Your 30-second opening

> Good morning. I am Akshat Agarwal from CS-5A. My project is Financial News Sentiment Analysis Using FinBERT. It classifies short English financial statements as positive, neutral, or negative. I fine-tuned a financial BERT model, compared it with a conventional machine-learning baseline, and built a Streamlit application. Our fine-tuned model achieved 85.12% accuracy and 84.46% macro-F1 on the saved held-out test split.

## Slide 1 — Financial News Sentiment Analysis

**Say, in about 30–45 seconds:**

> Financial statements contain information about earnings, demand, costs, and forecasts. The same word can have different implications depending on context. Our task is to identify the financial sentiment of a statement. Higher profit may support positive sentiment, weaker margins may support negative sentiment, and a scheduled announcement may be neutral. The application accepts one statement or a CSV and displays the predicted class and all three scores.

Mention Python, PyTorch, Hugging Face Transformers, scikit-learn, and Streamlit. State the scope clearly: this is sentiment classification, not stock-price prediction or a trading recommendation.

## Slide 2 — How Transformer Works

**Say, in about 30–45 seconds:**

> First, the tokenizer converts the sentence into subword tokens and numerical IDs. An attention mask distinguishes actual tokens from padding. BERT's encoder layers use self-attention to build contextual representations. This helps the model relate words such as “profit,” “fell,” and “costs” within the sentence. A representation associated with the special CLS token is pooled and passed through a classification head. The head produces three logits. Softmax converts those logits into the displayed class scores, and the largest score determines the label.

BERT has **12 encoder layers, 12 attention heads per layer, and hidden size 768**. Attention heads learn different relationships; they are not twelve separate sentiment models. The model is an encoder classifier and does not generate a written answer.

## Slide 3 — Training Pipeline: Three Model Routes

**Say, in about 30–45 seconds:**

> We used Financial PhraseBank's 50Agree subset, which contains human-labeled financial sentences. From 4,846 raw rows, we removed six matching duplicate occurrences and quarantined four rows with conflicting labels. That left 4,836 clean sentences. We preserved a seeded split of 3,385 training, 725 validation, and 726 test examples, with no normalized exact-sentence overlap. We compared majority prediction, a trained TF-IDF plus logistic-regression baseline, and our fine-tuned FinBERT.

Explain the roles: **training learns weights, validation selects the model, and test measures the selected frozen model**. TF-IDF learns word and word-pair features from training text; logistic regression learns classification weights. Validation selected its regularization setting, **C=2**.

The transformer starts from **`yiyanghkust/finbert-pretrain`**, which already learned financial language. We added a new three-class head and fine-tuned the encoder and head. This is transfer learning, not training BERT from scratch. “50Agree” describes annotator agreement, not a 50% training split.

## Slide 4 — Evaluation Results Comparison

**Say, in about 30–45 seconds:**

> On the same 726 test examples, the majority predictor achieved 59.37% accuracy. Our TF-IDF baseline achieved 75.76% accuracy and 71.94% macro-F1. Fine-tuned FinBERT achieved 85.12% accuracy and 84.46% macro-F1, correctly classifying 618 examples. That is 68 more correct predictions than the baseline, with gains of 9.37 percentage points in accuracy and 12.52 points in macro-F1.

Accuracy is the fraction correctly classified. Macro-F1 averages the three class F1 scores equally, which matters because neutral examples are most common. Precision asks how often a predicted class is correct; recall asks how many actual examples of that class were recovered. Positive sentiment has the weakest F1 here, **76.59%**, showing room for improvement.

## Slide 5 — Fine-Tuning: Checkpoint Selection

**Say, in about 30–45 seconds:**

> We fully fine-tuned all 109,754,115 trainable parameters on a Tesla T4. Training ran for three epochs and 636 optimizer steps, taking about seven minutes fifty seconds. Validation macro-F1 was 81.40%, 81.92%, and 81.87%. We therefore selected epoch two, checkpoint 424. The final test evaluates that selected checkpoint. We did not automatically choose the last epoch or use test accuracy to select it.

Show the [training curves](../outputs/finetuning/training_curves.png). Training loss decreased, but epoch-three validation loss increased. This is a reason to retain the validation-selected checkpoint rather than assume that longer training helps. The final repeated validation entry in [training history](../outputs/finetuning/training_history.json) evaluates the restored best checkpoint; it is not a fourth epoch.

## Slide 6 — Conclusion: Unit Mapping

**Say, in about 30–45 seconds:**

> This project connects Unit 4 topics through BERT tokenization, self-attention, transfer learning, supervised fine-tuning, and a deployed text application. Our contribution includes dataset cleaning, reproducible splits, actual transformer training, baseline comparison, evaluation, and the application. Results support an improvement over our conventional baseline on this fixed split. Limitations include class imbalance, positive-neutral confusion, and the absence of a completed independent future-news evaluation.

Finish with: “I will now demonstrate single-statement analysis and batch CSV processing.”

## Explain training without mixing up the terms

Use this sequence: **forward pass → loss → backpropagation → optimizer update**.

The forward pass produces logits. **Cross-entropy loss** compares those raw logits with the correct class ID. **Backpropagation** calculates how parameters contributed to the loss. **AdamW**, the optimizer, uses gradients and moving averages to update the parameters. The learning rate controls the update scale. AdamW's decoupled weight decay regularizes weights; it is different from dropout.

Softmax is used to display scores. Cross-entropy already handles the required log-softmax calculation internally, so we do not apply softmax before passing logits into that loss. A score of 0.95 is not a guarantee of 95% real-world correctness.

| Actual setting | Meaning |
| --- | --- |
| PyTorch AdamW; learning rate `2e-5` | Optimizer and configured peak learning rate |
| Weight decay `0.01`; dropout `0.1` | Different regularization mechanisms |
| Linear scheduler; 10% warmup | Increase learning rate initially, then decrease it |
| Microbatch 2; accumulation 8 | Typically 16 examples per optimizer update; final batches can be smaller |
| FP16 and gradient checkpointing | Mixed precision and recomputation reduce memory demands |
| Dynamic padding; maximum 128 tokens | Pad each batch as needed; enforce a documented token limit |
| Seed 42; three epochs | Record randomness and passes through the training data |

Training and test required no truncation; one validation sentence exceeded 128 tokens and was right-truncated. The app rejects inputs beyond its 128-token limit. Tokens are subword pieces, not necessarily whole words. Gradient checkpointing saves memory by recomputing activations; a saved training checkpoint stores a model state. They mean different things.

## Exact application demonstration

Before presenting, open the app and load the fine-tuned model while online. Keep the local PPT and evidence available. You do not need Colab or retraining during the presentation.

1. In **Analysis model**, select **Fine-tuned FinBERT**. Open **Analyze text**.
2. In **Load a demo statement**, choose **Positive · stronger earnings**. Click **Analyze sentiment**. Point out the label and three scores.
3. Repeat with **Neutral · scheduled announcement** and **Negative · weaker demand**. Explain the financial meaning of each statement.
4. If time permits, show **Mixed · a useful edge case**. Explain that opposing facts still require one of three labels, so human review may be useful. Demo sample names are illustrations, not evaluation ground truth.
5. Open **Batch CSV**, check **Use the bundled demo CSV**, keep the `text` column selected, and click **Analyze CSV**. Show all eight source rows: seven valid statements and one flagged blank row. Click **Download results CSV**.
6. Open **Dataset & Results**. Show the saved split and measured test results. These are recorded evaluation results, not accuracy calculated from the demo sentences.
7. If asked for evidence, open [training manifest](../outputs/finetuning/training_manifest.json), [training history](../outputs/finetuning/training_history.json), [evaluation JSON](../outputs/finetuning/evaluation.json), and the curves PNG. Show GitHub if time allows.

An optional switch to **TF-IDF baseline** demonstrates a different trained approach. If showing **Prosus FinBERT reference**, explain its prior PhraseBank sentiment-training exposure. Its 88.15% diagnostic is not a fair unseen-data ranking against our student-trained models. If a live operation fails, explain the failure and show saved evidence; do not describe a saved result as a fresh prediction.

## Likely viva questions

**Why FinBERT rather than ordinary BERT?** Its prior financial-language learning gives a relevant starting point; our supervised training adapts it to three sentiment classes.

**Did you train the model yourself?** We fine-tuned the HKUST pretrained encoder and a new classifier using the project's training script. All parameters were trainable; recorded encoder changes provide training evidence.

**Why not continue training Prosus?** It already received supervised PhraseBank training. Our separate random split cannot undo that exposure. The HKUST financial language-model checkpoint is a clearer starting point for our training experiment, although zero overlap with its unpublished pretraining corpus is not established.

**Why epoch two?** It had the best validation macro-F1. Selection did not depend on the final test.

**Is this generative AI?** It uses transformer technology studied in Unit 4, but this particular BERT application classifies text rather than generating text.

**Can it predict share prices?** No. Financial sentiment and subsequent market returns are different targets.

**What would you improve?** Collect independently reviewed recent examples, examine positive-neutral errors, assess score calibration, and compare across more carefully designed evaluations.

## Last-minute memory card

**Title:** Financial News Sentiment Analysis Using FinBERT. **Labels:** negative, neutral, positive. **Data:** 4,846 raw → 4,836 clean; split 3,385 / 725 / 726. **Training:** HKUST pretrained BERT, full fine-tuning, AdamW, three epochs, best epoch two. **Test:** 85.12% accuracy, 84.46% macro-F1, 618 correct. **Baseline:** 75.76% / 71.94%. **Main limitation:** one corpus split, not proven future-news performance. Say **percentage-point gain**, not “9.37% more accurate.”
