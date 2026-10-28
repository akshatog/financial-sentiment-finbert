"""Named-label inference with explicit input limits and no silent truncation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config import BASELINE_PATH, LABELS, MODEL_ID, MODEL_REVISION, ROOT, setup_cache


class ModelInputError(ValueError):
    """An input cannot be classified under the documented input policy."""


class ModelLoadError(RuntimeError):
    """A model dependency or checkpoint is unavailable."""


class FinBERTPredictor:
    """Sentence-level English financial sentiment with pretrained FinBERT."""

    def __init__(self, device: str = 'cpu', model_path: str | Path | None = None):
        setup_cache()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError('Device must be cpu, cuda, or auto.')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise ModelLoadError('CUDA is unavailable in this environment. Select CPU.')
        self.device = torch.device(device)
        torch.set_num_threads(max(1, min(8, int(os.environ.get('FINBERT_CPU_THREADS', '4')))))
        local = Path(model_path) if model_path else ROOT / 'models' / 'finbert'
        source = str(local) if (local / 'config.json').exists() else MODEL_ID
        kwargs: dict[str, Any] = {} if source != MODEL_ID else {'revision': MODEL_REVISION}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                source, use_safetensors=True, **kwargs
            ).to(self.device).eval()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelLoadError(
                'FinBERT could not load. Run python -m scripts.download_model '
                'with internet access, then try again.'
            ) from exc
        self.id2label = {int(i): label.lower() for i, label in self.model.config.id2label.items()}
        if set(self.id2label.values()) != set(LABELS):
            raise ModelLoadError(f'Unexpected checkpoint labels: {self.id2label}')
        self.max_tokens = min(int(self.tokenizer.model_max_length), int(self.model.config.max_position_embeddings))
        self.name = 'FinBERT'

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        import torch

        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ModelInputError('Enter a non-empty text string for every prediction.')
        lengths = self.token_lengths(texts)
        if any(length > self.max_tokens for length in lengths):
            raise ModelInputError(
                f'FinBERT accepts at most {self.max_tokens} tokens including special tokens. '
                f'This batch contains an input of {max(lengths)} tokens. Use shorter sentences.'
            )
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(texts), 8):
            batch = self.tokenizer(texts[start:start + 8], padding=True, truncation=False, return_tensors='pt')
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                probabilities = self.model(**batch).logits.softmax(dim=-1).cpu().tolist()
            for row in probabilities:
                scores = {self.id2label[i]: float(score) for i, score in enumerate(row)}
                label = max(scores, key=scores.get)
                outputs.append({'label': label, 'score': scores[label], **scores})
        return outputs


class BaselineAdapter:
    """Expose the same interface for the student-trained conventional model."""

    max_tokens = None
    name = 'TF-IDF baseline'
    device = 'cpu'

    def __init__(self, model_path: Path = BASELINE_PATH):
        from src.baseline import BaselinePredictor

        if not model_path.exists():
            raise ModelLoadError('Train the baseline first: python -m scripts.train_baseline')
        self.predictor = BaselinePredictor(model_path)

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ModelInputError('Enter a non-empty text string for every prediction.')
        return self.predictor.predict(texts) if texts else []


def get_predictor(model_name: str = 'FinBERT', device: str = 'cpu'):
    if model_name == 'FinBERT':
        return FinBERTPredictor(device=device)
    if model_name in {'TF-IDF baseline', 'Baseline'}:
        return BaselineAdapter()
    raise ValueError(f'Unknown model: {model_name}')
