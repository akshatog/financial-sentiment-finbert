# Attribution and third-party assets

Checked against the linked project cards on 29 October 2026. Original project code uses the repository's MIT License. Third-party model weights and datasets retain their own terms.

## FinBERT

- Checkpoint: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
- Purpose: pretrained financial sentiment inference, without new transformer training in this project
- Weight license shown by the model card: **CC BY-NC-SA 4.0**
- Attribution: Dogu Araci, *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*, 2019, [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)
- The model card credits Financial PhraseBank for sentiment fine-tuning and a subset of Reuters TRC2 for financial language adaptation. Access terms for underlying corpora remain separate.

## Financial PhraseBank

- Dataset: [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank)
- Configuration: `Sentences_50Agree.txt` in `FinancialPhraseBank-v1.0.zip`
- License shown by the dataset card: **CC BY-NC-SA 3.0**
- Attribution: Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology, 65(4), 782–796, 2014.
- The download/preparation pipeline records source details and checksum. The data is not relicensed under MIT.

## BERT

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).

## Software

The application uses Streamlit, PyTorch, Hugging Face Transformers, scikit-learn, pandas, and their dependencies. Their source packages retain their respective licenses. The installed versions are specified by the project's dependency files.

## Demo examples and presentation

Short demonstration sentences in this project are self-authored examples. They have no independent human annotation and do not form a benchmark. Screenshots show this project's own application. The PowerPoint uses editable text, tables, and charts based on generated project evidence.

