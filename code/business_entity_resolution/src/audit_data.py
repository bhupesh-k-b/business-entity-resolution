"""Stream the supplied TSVs and print data-contract statistics as JSON.

Usage: python -m src.audit_data --data-dir student_resource/dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .data_contract import parse_ids
except ImportError:
    from data_contract import parse_ids


SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]


def source_audit(path: Path, prefix: str) -> tuple[dict, np.ndarray]:
    ids = array("I")
    hashes = array("Q")
    countries = Counter()
    blank = Counter()
    bad_prefix = bad_width = bad_id = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) != 4:
                bad_width += 1
                continue
            entity_id, name, address, country = row
            if not entity_id.startswith(prefix + "-"):
                bad_prefix += 1
            try:
                ids.append(int(entity_id.split("-", 1)[1]))
            except (IndexError, ValueError, OverflowError):
                bad_id += 1
            for field, value in zip(SOURCE_HEADER, row):
                if not value.strip():
                    blank[field] += 1
            countries[country] += 1
            raw = "\x1f".join(row[1:]).encode("utf-8")
            hashes.append(int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big"))
    id_array = np.frombuffer(ids, dtype=np.uint32).copy()
    hash_array = np.frombuffer(hashes, dtype=np.uint64).copy()
    unique_ids = np.unique(id_array)
    unique_records = np.unique(hash_array)
    return {
        "header": header,
        "rows": int(len(id_array)),
        "countries": dict(countries),
        "blank": dict(blank),
        "bad_width": bad_width,
        "bad_prefix": bad_prefix,
        "bad_numeric_id": bad_id,
        "duplicate_ids": int(len(id_array) - len(unique_ids)),
        "duplicate_exact_records": int(len(hash_array) - len(unique_records)),
    }, unique_ids


def truth_audit(path: Path, source1: np.ndarray, source2: np.ndarray, source3: np.ndarray) -> dict:
    s1_ids = array("I")
    target2 = array("I")
    target3 = array("I")
    cardinality = Counter()
    duplicate_in_row = bad_prefix = bad_width = bad_s1 = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) != 2:
                bad_width += 1
                continue
            s1, raw = row
            try:
                if not s1.startswith("S1-"):
                    bad_s1 += 1
                s1_ids.append(int(s1.split("-", 1)[1]))
            except (IndexError, ValueError, OverflowError):
                bad_s1 += 1
            labels = parse_ids(raw)
            cardinality[len(labels)] += 1
            duplicate_in_row += len(labels) - len(set(labels))
            for label in labels:
                try:
                    p, number = label.split("-", 1)
                    target = int(number)
                    if p == "S2":
                        target2.append(target)
                    elif p == "S3":
                        target3.append(target)
                    else:
                        bad_prefix += 1
                except (IndexError, ValueError, OverflowError):
                    bad_prefix += 1

    s1_array = np.frombuffer(s1_ids, dtype=np.uint32).copy()
    t2_array = np.frombuffer(target2, dtype=np.uint32).copy()
    t3_array = np.frombuffer(target3, dtype=np.uint32).copy()
    unique_s1 = np.unique(s1_array)
    unique_t2 = np.unique(t2_array)
    unique_t3 = np.unique(t3_array)
    s1_missing = np.setdiff1d(source1, unique_s1, assume_unique=True)
    truth_s1_unknown = np.setdiff1d(unique_s1, source1, assume_unique=True)
    t2_unknown = np.setdiff1d(unique_t2, source2, assume_unique=True)
    t3_unknown = np.setdiff1d(unique_t3, source3, assume_unique=True)
    return {
        "header": header,
        "rows": int(len(s1_array)),
        "bad_width": bad_width,
        "bad_s1_id": bad_s1,
        "bad_target_prefix_or_id": bad_prefix,
        "duplicate_s1_rows": int(len(s1_array) - len(unique_s1)),
        "duplicate_labels_within_row": duplicate_in_row,
        "source1_missing_truth": int(len(s1_missing)),
        "truth_s1_not_in_source1": int(len(truth_s1_unknown)),
        "unknown_s2_labels": int(len(t2_unknown)),
        "unknown_s3_labels": int(len(t3_unknown)),
        "label_count_s2": int(len(t2_array)),
        "label_count_s3": int(len(t3_array)),
        "distinct_labeled_s2": int(len(unique_t2)),
        "distinct_labeled_s3": int(len(unique_t3)),
        "reused_s2_labels": int(len(t2_array) - len(unique_t2)),
        "reused_s3_labels": int(len(t3_array) - len(unique_t3)),
        "cardinality": {str(k): v for k, v in sorted(cardinality.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    sources = {}
    for split in ("train", "test"):
        for number in (1, 2, 3):
            key = f"{split}_source{number}"
            result[key], sources[key] = source_audit(
                args.data_dir / split / f"{key}.tsv", f"S{number}"
            )
    for number in (1, 2, 3):
        a = sources[f"train_source{number}"]
        b = sources[f"test_source{number}"]
        result[f"source{number}_train_test_id_overlap"] = int(
            len(np.intersect1d(a, b, assume_unique=True))
        )
    result["train_ground_truth"] = truth_audit(
        args.data_dir / "train" / "train_ground_truth.tsv",
        sources["train_source1"],
        sources["train_source2"],
        sources["train_source3"],
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
