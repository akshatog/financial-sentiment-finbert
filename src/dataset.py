"""Financial PhraseBank parsing, duplicate handling, and reproducible splits.

The agreement subsets overlap. Only the 50Agree subset is used for training;
the other files are independently audited and are never concatenated.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("negative", "neutral", "positive")
DATASET_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
DATASET_SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
EXPECTED_VERSION_ROWS = {"50Agree": 4846, "66Agree": 4217, "75Agree": 3453, "AllAgree": 2264}
EXPECTED_MAIN_COUNTS = {"negative": 604, "neutral": 2879, "positive": 1363}


def normalize_text(text: str) -> str:
    """Group case and whitespace variants without removing meaningful words."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint ordered, canonical rows, including labels and text."""
    payload = frame.loc[:, ["id", "text", "label"]].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_phrasebank_bytes(content: bytes, source: str = "PhraseBank") -> pd.DataFrame:
    records = []
    for line_number, line in enumerate(content.decode("latin-1").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{source}:{line_number}: blank record")
        if "@" not in line:
            raise ValueError(f"{source}:{line_number}: missing final @ label separator")
        text, label = line.rsplit("@", 1)
        text, label = text.strip(), label.strip()
        if not text:
            raise ValueError(f"{source}:{line_number}: blank sentence")
        if label not in LABELS:
            raise ValueError(f"{source}:{line_number}: unexpected label {label!r}")
        records.append({"text": text, "label": label})
    if not records:
        raise ValueError(f"{source}: no records")
    return pd.DataFrame.from_records(records, columns=["text", "label"])


def read_archive(archive_path: str | Path) -> dict[str, pd.DataFrame]:
    """Read expected text members in memory; never extract archive members."""
    versions = {}
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or re.match(r"^[A-Za-z]:", str(name)):
                raise ValueError(f"Unsafe ZIP member: {member.filename!r}")
        for version in EXPECTED_VERSION_ROWS:
            expected_name = f"Sentences_{version}.txt"
            matches = [member for member in members if PurePosixPath(member.filename).name == expected_name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {expected_name}, found {len(matches)}")
            member = matches[0]
            if member.file_size > 20 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large dataset member: {member.filename}")
            versions[version] = parse_phrasebank_bytes(archive.read(member), member.filename)
    return versions


def validate_records(frame: pd.DataFrame) -> None:
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must have text and label columns")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if not frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every sentence must be a nonblank string")
    unexpected = set(frame["label"]) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels: {sorted(map(str, unexpected))}")


def audit_records(frame: pd.DataFrame) -> dict:
    validate_records(frame)
    keys = frame["text"].map(normalize_text)
    key_labels = pd.DataFrame({"key": keys, "label": frame["label"]})
    conflicting = key_labels.groupby("key")["label"].nunique()
    conflicting_keys = set(conflicting[conflicting > 1].index)
    char_lengths = frame["text"].str.len()
    word_lengths = frame["text"].str.split().str.len()
    return {
        "rows": int(len(frame)),
        "class_counts": {label: int((frame["label"] == label).sum()) for label in LABELS},
        "blank_sentences": int(frame["text"].str.strip().eq("").sum()),
        "exact_duplicate_occurrences": int(frame["text"].duplicated().sum()),
        "normalized_duplicate_occurrences": int(keys.duplicated().sum()),
        "normalized_conflicting_groups": len(conflicting_keys),
        "normalized_conflicting_rows": int(keys.isin(conflicting_keys).sum()),
        "sentence_characters": {"min": int(char_lengths.min()), "median": float(char_lengths.median()), "max": int(char_lengths.max())},
        "sentence_words": {"min": int(word_lengths.min()), "median": float(word_lengths.median()), "max": int(word_lengths.max())},
    }


def clean_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Quarantine every contradictory group, then keep one matching duplicate.

    IDs derive from normalized text, so repeated preparation preserves membership.
    Full quarantined sentences must be kept in private local data, not published.
    """
    audit = audit_records(frame)
    working = frame.loc[:, ["text", "label"]].copy().reset_index(drop=True)
    working["_key"] = working["text"].map(normalize_text)
    distinct_labels = working.groupby("_key")["label"].nunique()
    conflicting_keys = set(distinct_labels[distinct_labels > 1].index)
    conflict_mask = working["_key"].isin(conflicting_keys)
    quarantine = working.loc[conflict_mask, ["text", "label"]].copy()
    quarantine["reason"] = "conflicting_labels_for_normalized_sentence"
    clean = working.loc[~conflict_mask].drop_duplicates("_key", keep="first").copy()
    clean["id"] = clean["_key"].map(lambda text: "fpb_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20])
    clean = clean.loc[:, ["id", "text", "label"]].sort_values("id").reset_index(drop=True)
    if not clean["id"].is_unique:
        raise ValueError("Unexpected canonical ID collision")
    audit.update({
        "conflicting_rows_quarantined": int(conflict_mask.sum()),
        "matching_duplicate_occurrences_removed": int((~conflict_mask).sum() - len(clean)),
        "clean_rows": int(len(clean)),
        "clean_class_counts": {label: int((clean["label"] == label).sum()) for label in LABELS},
        "duplicate_policy": "NFKC Unicode normalization, casefold, collapse whitespace; quarantine every conflicting group; retain first matching sentence",
        "quarantined_group_ids": sorted("fpb_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] for key in conflicting_keys),
    })
    return clean, quarantine.reset_index(drop=True), audit


def create_splits(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 split after grouping and duplicate removal."""
    validate_records(frame)
    if "id" not in frame or not frame["id"].is_unique:
        raise ValueError("Canonical dataset must contain unique IDs")
    if frame["text"].map(normalize_text).duplicated().any():
        raise ValueError("Normalize and deduplicate the dataset before splitting")
    try:
        train, held_out = train_test_split(frame, test_size=0.30, random_state=seed, stratify=frame["label"])
        validation, test = train_test_split(held_out, test_size=0.50, random_state=seed, stratify=held_out["label"])
    except ValueError as exc:
        raise ValueError("Not enough examples per class for stratified 70/15/15 splitting") from exc
    splits = {name: part.loc[:, ["id", "text", "label"]].sort_values("id").reset_index(drop=True) for name, part in {"train": train, "validation": validation, "test": test}.items()}
    assert_disjoint_splits(splits)
    return splits


def assert_disjoint_splits(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for index, name in enumerate(names):
        frame = splits[name]
        validate_records(frame)
        if "id" not in frame or not frame["id"].is_unique:
            raise ValueError(f"{name} must contain unique IDs")
        if frame["text"].map(normalize_text).duplicated().any():
            raise ValueError(f"{name} contains normalized duplicate sentences")
        for other_name in names[index + 1:]:
            other = splits[other_name]
            if set(frame["id"]) & set(other["id"]):
                raise ValueError(f"ID overlap between {name} and {other_name}")
            if set(frame["text"].map(normalize_text)) & set(other["text"].map(normalize_text)):
                raise ValueError(f"Normalized sentence overlap between {name} and {other_name}")
