import io

import pandas as pd
import pytest

from src.batch import analyze_dataframe, export_csv, read_csv_file


class FakePredictor:
    max_tokens = 8

    def token_lengths(self, texts):
        return [len(text.split()) + 2 for text in texts]

    def predict(self, texts):
        return [{'label': 'positive', 'score': .7, 'positive': .7, 'neutral': .2, 'negative': .1} for text in texts]


def test_batch_retains_blanks_and_oversized_rows():
    frame = pd.DataFrame({'text': ['Profit increased.', '', 'word ' * 10], 'id': [5, 6, 7]})
    result = analyze_dataframe(frame, 'text', FakePredictor())
    assert result['id'].tolist() == [5, 6, 7]
    assert result['status'].tolist() == ['ok', 'invalid', 'invalid']
    assert result.loc[0, 'prediction'] == 'positive'
    assert result.loc[1, 'error']
    assert 'limit' in result.loc[2, 'error']
    assert frame.columns.tolist() == ['text', 'id']


def test_csv_keeps_literal_na_and_rejects_bad_files():
    assert read_csv_file(b'text\nNA\n').iloc[0, 0] == 'NA'
    with pytest.raises(ValueError, match='valid UTF-8'):
        read_csv_file(b'')
    with pytest.raises(ValueError, match='no data'):
        read_csv_file(b'text\n')


def test_exports_escape_formulas_but_keep_numbers():
    encoded = export_csv(pd.DataFrame({'text': ['=SUM(1,2)', 'normal'], 'score': [.2, .8]}))
    decoded = pd.read_csv(io.BytesIO(encoded))
    assert decoded.loc[0, 'text'].startswith("'=")
    assert decoded['score'].tolist() == [.2, .8]


def test_rejects_result_column_overwrite():
    with pytest.raises(ValueError, match='already contains'):
        analyze_dataframe(pd.DataFrame({'text': ['Profit increased.'], 'status': ['old']}), 'text', FakePredictor())


def test_csv_preserves_leading_zero_identifiers_and_rejects_duplicate_headers():
    frame = read_csv_file(b'id,text\n0007,Profit increased.\n')
    assert frame.loc[0, 'id'] == '0007'
    with pytest.raises(ValueError, match='unique'):
        read_csv_file(b'text,text\nprofit,loss\n')
