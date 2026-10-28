"""Download only FinBERT inference files, and record their resolved revision."""
from __future__ import annotations

import json

from src.config import MODEL_ID, MODEL_REVISION, OUTPUT_DIR, ROOT, setup_cache


def main() -> None:
    setup_cache()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
    destination = ROOT / 'models' / 'finbert'
    snapshot_download(
        MODEL_ID, revision=info.sha, local_dir=destination,
        allow_patterns=['config.json', 'model.safetensors', 'tokenizer_config.json',
                        'special_tokens_map.json', 'vocab.txt', 'tokenizer.json'],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'model_id': MODEL_ID, 'revision': info.sha, 'format': 'safetensors',
                'source': f'https://huggingface.co/{MODEL_ID}',
                'training_overlap': 'Checkpoint fine-tuned on Financial PhraseBank.'}
    (OUTPUT_DIR / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
