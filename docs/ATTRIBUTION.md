# Attribution and third-party assets

Checked against the linked project cards on 29 October 2026. Original project code uses the repository's MIT License. Third-party model weights and datasets retain their own terms.

## Prosus FinBERT reference

- Checkpoint: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
- Purpose: optional pretrained financial sentiment reference; no additional training of this Prosus checkpoint in this project
- Weight license shown by the model card: **CC BY-NC-SA 4.0**
- Attribution: Dogu Araci, *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*, 2019, [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)
- The model card credits Financial PhraseBank for sentiment fine-tuning and a subset of Reuters TRC2 for financial language adaptation. Access terms for underlying corpora remain separate.

## HKUST FinBERT and our fine-tuned classifier

- Starting checkpoint: [`yiyanghkust/finbert-pretrain`](https://huggingface.co/yiyanghkust/finbert-pretrain), pinned revision `88ab954a39ea6d3ce2b62cff086dd5ad1172c664`.
- Purpose: a financial language model used as the starting point for our full supervised encoder and three-class sentiment-head fine-tuning.
- Attribution: Allen H. Huang, Hui Wang, and Yi Yang. *FinBERT: A Large Language Model for Extracting Information from Financial Text*, Contemporary Accounting Research (2022); Yi Yang, Mark Christopher Siy Uy, and Allen Huang. *FinBERT: A Pretrained Language Model for Financial Communications* (2020), [arXiv:2006.08097](https://arxiv.org/abs/2006.08097).
- [Authors' repository](https://github.com/yya518/FinBERT) and [fine-tuning example](https://github.com/yya518/FinBERT/blob/master/finetune.ipynb).
- This checkpoint is separate from Prosus FinBERT. Do not assign the Prosus card's license to HKUST weights or infer a model-weight license from a source-code license. Our MIT license covers original code only; third-party weights, underlying pretraining corpora and PhraseBank data retain their separate terms.
- The fine-tuned artifact records its source revision, training settings, label mapping, selected checkpoint and hashes. It excludes the licensed dataset sentences and private prediction files.

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

