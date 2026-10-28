"""CSV handling that preserves invalid rows and protects exported text cells."""
from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd

from src.config import MAX_CSV_BYTES, MAX_CSV_ROWS

RESULT_COLUMNS = ('prediction', 'model_score', 'score_positive', 'score_neutral', 'score_negative', 'status', 'error')


def read_csv_file(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_CSV_BYTES:
        raise ValueError('The CSV must be no larger than 2 MB.')
    try:
        header = next(csv.reader(io.StringIO(content.decode('utf-8-sig'))), [])
        if len(header) != len(set(header)):
            raise ValueError('CSV column names must be unique.')
        frame = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig', keep_default_na=False, dtype=str)
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Upload a valid UTF-8 CSV with a header row.') from exc
    if frame.empty:
        raise ValueError('The CSV contains no data rows.')
    if len(frame) > MAX_CSV_ROWS:
        raise ValueError(f'The CSV contains {len(frame):,} rows. The limit is {MAX_CSV_ROWS}.')
    if frame.columns.duplicated().any():
        raise ValueError('CSV column names must be unique.')
    return frame


def analyze_dataframe(frame: pd.DataFrame, text_column: str, predictor: Any, batch_size: int = 8) -> pd.DataFrame:
    if text_column not in frame.columns:
        raise ValueError(f'Text column {text_column!r} was not found.')
    if len(frame) > MAX_CSV_ROWS:
        raise ValueError(f'Analyze at most {MAX_CSV_ROWS} rows at a time.')
    if any(column in frame.columns for column in RESULT_COLUMNS):
        raise ValueError('The CSV already contains result columns. Rename them before uploading.')
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    result = frame.copy().reset_index(drop=True)
    for column in RESULT_COLUMNS:
        result[column] = '' if column in {'prediction', 'status', 'error'} else float('nan')
    valid: list[tuple[int, str]] = []
    for index, value in enumerate(result[text_column]):
        if not isinstance(value, str) or not value.strip():
            result.loc[index, ['status', 'error']] = ['invalid', 'Text must be a non-empty string.']
            continue
        valid.append((index, value))
    maximum = getattr(predictor, 'max_tokens', None)
    if maximum is not None and valid:
        lengths = predictor.token_lengths([text for _, text in valid])
        remaining = []
        for (index, text), length in zip(valid, lengths, strict=True):
            if length > maximum:
                result.loc[index, ['status', 'error']] = ['invalid', f'Input has {length} tokens; limit is {maximum}.']
            else:
                remaining.append((index, text))
        valid = remaining
    for start in range(0, len(valid), batch_size):
        items = valid[start:start + batch_size]
        predictions = predictor.predict([text for _, text in items])
        if len(predictions) != len(items):
            raise RuntimeError('The model returned an unexpected number of predictions.')
        for (index, _), prediction in zip(items, predictions, strict=True):
            result.loc[index, list(RESULT_COLUMNS)] = [
                prediction['label'], prediction['score'], prediction['positive'],
                prediction['neutral'], prediction['negative'], 'ok', '',
            ]
    return result


def export_csv(frame: pd.DataFrame) -> bytes:
    """Escape formula-like text when users open results in a spreadsheet."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=['object', 'string']).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r'))
            else value
        )
    return safe.to_csv(index=False).encode('utf-8-sig')
