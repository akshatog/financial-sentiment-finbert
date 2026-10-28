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
