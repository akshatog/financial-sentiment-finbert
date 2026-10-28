"""Synthetic fixtures only; no licensed PhraseBank sentences are embedded."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd

from src.dataset import (assert_disjoint_splits, clean_records, create_splits,
                         normalize_text, parse_phrasebank_bytes, read_archive, ROOT)


class DatasetTests(unittest.TestCase):
    def test_final_separator_and_latin1_are_preserved(self):
        parsed = parse_phrasebank_bytes("Café reports @ its annual meeting @positive\nNo change is expected@neutral\n".encode("latin-1"))
        self.assertEqual(parsed.iloc[0]["text"], "Café reports @ its annual meeting")
        self.assertEqual(parsed.iloc[0]["label"], "positive")

    def test_conflicting_group_is_entirely_quarantined(self):
        frame = pd.DataFrame([
            {"text": "  Sales   improved ", "label": "positive"},
            {"text": "SALES improved", "label": "positive"},
            {"text": "Costs fell", "label": "positive"},
            {"text": "COSTS   FELL", "label": "negative"},
            {"text": "Statement unchanged", "label": "neutral"},
        ])
        clean, quarantine, audit = clean_records(frame)
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(quarantine), 2)
        self.assertEqual(audit["matching_duplicate_occurrences_removed"], 1)
        self.assertEqual(audit["conflicting_rows_quarantined"], 2)
        self.assertEqual(set(quarantine["label"]), {"positive", "negative"})
        self.assertTrue(quarantine["reason"].str.contains("conflicting").all())
        self.assertEqual(set(clean["text"].map(normalize_text)), {"sales improved", "statement unchanged"})

    def test_seeded_splits_keep_every_canonical_row_once(self):
        rows = [{"text": f"Synthetic {label} statement number {number}", "label": label} for label in ("positive", "negative", "neutral") for number in range(40)]
        canonical, _, _ = clean_records(pd.DataFrame(rows))
        first = create_splits(canonical)
        second = create_splits(canonical.sample(frac=1, random_state=8).sort_values("id"))
        assert_disjoint_splits(first)
        self.assertEqual(sum(len(part) for part in first.values()), len(canonical))
        for name, part in first.items():
            self.assertEqual(set(part["label"]), {"positive", "negative", "neutral"})
            self.assertEqual(part["id"].tolist(), second[name]["id"].tolist())
        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [84, 18, 18])

    def test_overlap_detection_uses_normalized_text_not_only_ids(self):
        first = pd.DataFrame([{"id": "one", "text": "Revenue stable", "label": "neutral"}])
        second = pd.DataFrame([{"id": "two", "text": " REVENUE   stable ", "label": "neutral"}])
        with self.assertRaisesRegex(ValueError, "Normalized sentence overlap"):
            assert_disjoint_splits({"train": first, "test": second})

    def test_split_refuses_unresolved_duplicate_groups(self):
        rows = [{"id": f"{label}_{number}", "text": f"Synthetic {label} statement {number}", "label": label} for label in ("positive", "negative", "neutral") for number in range(20)]
        rows.append({"id": "different_id", "text": "  SYNTHETIC positive statement 0 ", "label": "positive"})
        with self.assertRaisesRegex(ValueError, "deduplicate"):
            create_splits(pd.DataFrame(rows))

    def test_rejects_invalid_record_and_archive_traversal(self):
        with self.assertRaises(ValueError):
            parse_phrasebank_bytes(b"sentence@unknown")
        with self.assertRaises(ValueError):
            parse_phrasebank_bytes(b"  @neutral")
        temp_root = ROOT / ".cache" / "tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            archive = Path(directory) / "unsafe.zip"
            with ZipFile(archive, "w") as handle:
                handle.writestr("../untrusted.txt", "do not extract")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                read_archive(archive)


if __name__ == "__main__":
    unittest.main()
