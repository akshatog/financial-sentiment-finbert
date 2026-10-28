"""Download, verify, audit, deduplicate, and split Financial PhraseBank."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (DATASET_SHA256, DATASET_URL, EXPECTED_MAIN_COUNTS,
                         EXPECTED_VERSION_ROWS, LABELS, ROOT, audit_records,
                         clean_records, create_splits, file_sha256,
                         frame_fingerprint, read_archive)
from src.metrics import plot_distribution
import matplotlib.pyplot as plt


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if file_sha256(destination) != DATASET_SHA256:
            raise ValueError(f"Existing archive checksum mismatch: {destination}. Remove or replace it explicitly.")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = Request(DATASET_URL, headers={"User-Agent": "FinancialSentimentStudentProject/1.0"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    raise ValueError("Archive exceeded the expected download size limit")
                output.write(chunk)
        if file_sha256(temporary) != DATASET_SHA256:
            raise ValueError("Downloaded archive SHA-256 does not match the pinned source archive")
        temporary.replace(destination)
    finally:
        if temporary.is_file():
            temporary.unlink()


def prepare(archive_path: Path, seed: int = 42) -> dict:
    checksum = file_sha256(archive_path)
    if checksum != DATASET_SHA256:
        raise ValueError(f"Archive checksum mismatch: expected {DATASET_SHA256}, got {checksum}")
    versions = read_archive(archive_path)
    version_audits = {name: audit_records(frame) for name, frame in versions.items()}
    for name, expected_rows in EXPECTED_VERSION_ROWS.items():
        if len(versions[name]) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows; got {len(versions[name])}")
    if version_audits["50Agree"]["class_counts"] != EXPECTED_MAIN_COUNTS:
        raise ValueError("Pinned 50Agree class counts do not match the expected source")
    clean, quarantined, main_audit = clean_records(versions["50Agree"])
    splits = create_splits(clean, seed=seed)
    processed = ROOT / "data" / "processed"
    split_dir = ROOT / "data" / "splits"
    output_dir = ROOT / "outputs"
    figures = output_dir / "figures"
    for directory in (processed, split_dir, output_dir, figures):
        directory.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed / "financial_phrasebank.csv", index=False, encoding="utf-8")
    quarantined.to_csv(processed / "quarantined.csv", index=False, encoding="utf-8")
    for name, frame in splits.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False, encoding="utf-8")
    main_audit["raw_rows"] = main_audit["rows"]
    audit = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"name": "Financial PhraseBank v1.0", "main_subset": "50Agree", "url": DATASET_URL, "sha256": checksum, "archive": str(archive_path.relative_to(ROOT)) if archive_path.is_relative_to(ROOT) else archive_path.name, "encoding": "ISO-8859-1 / latin-1", "parser": "Split each source line at its final @", "license_note": "Dataset terms are separate from code terms; consult the source/archive README. Full dataset sentences and private error examples are not published."},
        "versions": version_audits,
        "version_policy": "Audit agreement subsets separately; never concatenate these overlapping versions",
        "main_dataset": main_audit,
        "splits": {
            "seed": seed,
            "method": "Two stratified random splits after canonical duplicate removal; integer rounding may shift proportions slightly",
            "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "partitions": {name: {"rows": len(frame), "class_counts": {label: int((frame["label"] == label).sum()) for label in LABELS}, "fingerprint": frame_fingerprint(frame)} for name, frame in splits.items()},
            "normalized_sentence_overlap": 0,
            "id_overlap": 0,
        },
    }
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_distribution(main_audit["clean_class_counts"], figures / "dataset_distribution.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    ax.hist(clean["text"].str.split().str.len(), bins=35, color="#387ba6", edgecolor="white")
    ax.set(title="Cleaned PhraseBank sentence lengths", xlabel="Whitespace-separated words", ylabel="Sentences")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "sentence_lengths.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "raw" / "FinancialPhraseBank-v1.0.zip")
    parser.add_argument("--skip-download", action="store_true", help="Use an already downloaded archive; never access the network")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.skip_download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")
    audit = prepare(args.archive.resolve(), seed=args.seed)
    print(json.dumps({"raw_rows": audit["main_dataset"]["raw_rows"], "clean_rows": audit["main_dataset"]["clean_rows"], "quarantined_rows": audit["main_dataset"]["conflicting_rows_quarantined"], "split_rows": {name: info["rows"] for name, info in audit["splits"]["partitions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
