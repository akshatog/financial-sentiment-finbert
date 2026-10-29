"""Shared paths and project-local caches."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
BASELINE_PATH = ROOT / 'models' / 'tfidf_logreg.joblib'
MODEL_ID = 'ProsusAI/finbert'
MODEL_REVISION = '0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031'
FINETUNED_MODEL_PATH = ROOT / 'models' / 'finbert_finetuned'
FINETUNED_MODEL_NAME = 'Fine-tuned FinBERT'
PROSUS_MODEL_NAME = 'Prosus FinBERT reference'
FINETUNED_BASE_ID = 'yiyanghkust/finbert-pretrain'
FINETUNED_BASE_REVISION = '88ab954a39ea6d3ce2b62cff086dd5ad1172c664'
FINETUNED_ARCHIVE_URL = 'https://github.com/akshatog/financial-sentiment-finbert/releases/download/finbert-finetuned-v1/finbert_finetuned_results.zip'
FINETUNED_PARTS_URL = 'https://github.com/akshatog/financial-sentiment-finbert/releases/download/finbert-finetuned-v1/finbert_finetuned_parts.json'
FINETUNED_PARTS_SHA256 = '72b8f34a155db424b3fffe51816156b2f31f180032e5f28c8451beebaaf95585'
FINETUNED_ARCHIVE_SHA256 = '155ca67e272fbaef5413b4cad7f44237a52a006b9ffee623ddd5013e5a862592'
FINETUNED_MANIFEST_SHA256 = '1ea309e26939975f5e220be8115e299cc9fca9f52695500f3cb0bb07bb8b4710'
FINETUNED_WEIGHTS_SHA256 = '39eb5674d0adaf82a9a69dcfdec6736a2698bd2a9c0c7e3e90cd41c2afbe68f3'
LABELS = ('negative', 'neutral', 'positive')
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_CSV_ROWS = 500


def setup_cache() -> Path:
    """Keep project downloads and runtime caches under the project root."""
    cache = ROOT / '.cache'
    locations = {
        'HF_HOME': cache / 'huggingface',
        'TORCH_HOME': cache / 'torch',
        'MPLCONFIGDIR': cache / 'matplotlib',
        'PIP_CACHE_DIR': cache / 'pip',
        'XDG_CACHE_HOME': cache,
    }
    for name, location in locations.items():
        location.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(location)
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return cache
