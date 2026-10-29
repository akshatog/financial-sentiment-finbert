"""Evaluate the student models on a completed, frozen human review file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import FINETUNED_MODEL_NAME, LABELS, OUTPUT_DIR, ROOT
from src.dataset import normalize_text
from src.inference import get_predictor
from src.metrics import classification_metrics


def validate_review(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'id', 'text', 'source_url', 'human_label', 'review_notes'}
    if not required.issubset(frame.columns):
        raise ValueError('Review file is missing required columns.')
    if frame['id'].duplicated().any():
        raise ValueError('Review IDs must be unique.')
    if not frame['text'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Review sentences must be non-empty strings.')
    if not frame['human_label'].isin([*LABELS, 'exclude']).all():
        raise ValueError('Human review is incomplete. Fill every human_label before running predictions.')
    excluded = frame['human_label'] == 'exclude'
    if frame.loc[excluded, 'review_notes'].str.strip().eq('').any():
        raise ValueError('Every excluded example needs a review note.')
    reviewed = frame.loc[~excluded].copy()
    if reviewed.empty:
        raise ValueError('No reviewed examples remain.')
    keys = reviewed['text'].map(normalize_text)
    if keys.duplicated().any():
        raise ValueError('The reviewed set contains duplicate sentences.')
    return reviewed


def main() -> None:
    path = ROOT / 'data' / 'review' / 'candidates.csv'
    raw = pd.read_csv(path, keep_default_na=False)
    reviewed = validate_review(raw)
    fingerprint = hashlib.sha256(raw.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    destination = OUTPUT_DIR / 'independent_evaluation.json'
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved['review_sha256'] == fingerprint:
            print('Reusing the saved evaluation of this frozen reviewed set.')
            return
        raise ValueError('The review file changed after evaluation. Preserve the original test and record a separate study.')
    canonical_path = ROOT / 'data' / 'processed' / 'financial_phrasebank.csv'
    if not canonical_path.exists():
        raise ValueError('Prepare PhraseBank first so known training overlap can be checked.')
    canonical = pd.read_csv(canonical_path, keep_default_na=False)
    corpus = set(canonical['text'].map(normalize_text))
    if any(key in corpus for key in reviewed['text'].map(normalize_text)):
        raise ValueError('A review sentence also appears in PhraseBank.')
    report = {
        'review_sha256': fingerprint, 'n_reviewed': len(reviewed), 'n_excluded': len(raw) - len(reviewed),
        'label_support': reviewed['human_label'].value_counts().to_dict(),
        'known_exact_phrasebank_overlap': 0,
        'interpretation': 'Small source-linked, paraphrased, human-reviewed check across four company sources. Correlated curated examples; not a random benchmark or market prediction test.',
        'models': {},
    }
    private = OUTPUT_DIR / 'predictions'
    private.mkdir(parents=True, exist_ok=True)
    for name in [FINETUNED_MODEL_NAME, 'TF-IDF baseline']:
        rows = get_predictor(name).predict(reviewed['text'].tolist())
        report['models'][name] = classification_metrics(reviewed['human_label'].tolist(), [row['label'] for row in rows])
        output = reviewed.reset_index(drop=True).join(pd.DataFrame(rows).add_prefix('model_'))
        output.to_csv(private / f'review_{name.split()[0].lower()}.csv', index=False)
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
