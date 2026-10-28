import pandas as pd
import pytest

from scripts.evaluate_review import validate_review


def example(label='', notes=''):
    return pd.DataFrame([{'id': 'R1', 'text': 'Revenue increased.', 'source_url': 'https://example.com', 'human_label': label, 'review_notes': notes}])


def test_incomplete_human_review_cannot_be_evaluated():
    with pytest.raises(ValueError, match='incomplete'):
        validate_review(example())


def test_exclusion_requires_a_documented_reason():
    with pytest.raises(ValueError, match='review note'):
        validate_review(example('exclude'))


def test_reviewed_label_is_retained():
    assert validate_review(example('positive')).iloc[0]['human_label'] == 'positive'
