"""Data-contract checks, deterministic grouped split, and macro F0.5 scoring.

The split groups Source 1 rows that share a positive Source 2/3 entity. This
prevents the same labeled target from appearing in both model training and
validation. Raw records may still be used for unsupervised blocking on either
side, as they are available at inference time.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def parse_ids(value: str) -> tuple[str, ...]:
    """Parse a comma-separated label/prediction cell; retain order and duplicates."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def f05_per_entity(truth: Iterable[str], prediction: Iterable[str]) -> float:
    """Challenge F0.5, including the stated singleton convention.

    Duplicated predictions are invalid submission data and raise ValueError.
    """
    actual, guessed = set(truth), tuple(prediction)
    if len(guessed) != len(set(guessed)):
        raise ValueError("duplicate predicted entity ID")
    predicted = set(guessed)
    if not actual:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(actual & predicted)
    if not tp:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(actual)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
) -> float:
    """Average per-Source-1 F0.5; require exactly one prediction per S1."""
    if ground_truth.keys() != predictions.keys():
        raise ValueError("ground truth and prediction Source 1 IDs differ")
    if not ground_truth:
        raise ValueError("empty ground truth")
    return sum(
        f05_per_entity(actual, predictions[s1])
        for s1, actual in ground_truth.items()
    ) / len(ground_truth)


def assert_audit_contract(audit_json: Path) -> None:
    """Fail early on malformed IDs, labels, headers, or missing truth rows."""
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    expected_source = ["entity_id", "business_name", "business_address", "country"]
    for split in ("train", "test"):
        for number in (1, 2, 3):
            item = audit[f"{split}_source{number}"]
            assert item["header"] == expected_source
            assert item["bad_width"] == 0
            assert item["bad_prefix"] == 0
            assert item["bad_numeric_id"] == 0
            assert item["duplicate_ids"] == 0
            assert item["blank"].get("entity_id", 0) == 0
            assert item["blank"].get("business_name", 0) == 0
            assert item["blank"].get("country", 0) == 0
    truth = audit["train_ground_truth"]
    assert truth["header"] == ["source1_entity_id", "matched_entity_ids"]
    for key in (
        "bad_width", "bad_s1_id", "bad_target_prefix_or_id",
        "duplicate_s1_rows", "duplicate_labels_within_row",
        "source1_missing_truth", "truth_s1_not_in_source1",
        "unknown_s2_labels", "unknown_s3_labels",
    ):
        assert truth[key] == 0, key
    assert truth["rows"] == audit["train_source1"]["rows"]
    for number in (1, 2, 3):
        assert audit[f"source{number}_train_test_id_overlap"] == 0


def _hash_fraction(value: str, seed: str) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / (1 << 64)


def grouped_validation_ids(
    truth_path: Path,
    fraction: float = 0.10,
    seed: str = "amazon-er-v1",
) -> set[str]:
    """Hash positive-link components into validation; return held-out S1 IDs.

    Grouping uses only link topology to prevent a target label crossing the
    boundary. A singleton forms its own component. All countries are eligible.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    parent: dict[str, str] = {}
    target_owner: dict[str, str] = {}
    s1_ids: list[str] = []

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    with truth_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("unexpected ground truth header")
        for row in reader:
            s1 = row["source1_entity_id"]
            if s1 in parent:
                raise ValueError(f"duplicate ground truth S1 ID: {s1}")
            parent[s1] = s1
            s1_ids.append(s1)
            ids = parse_ids(row["matched_entity_ids"])
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate target ID for {s1}")
            for target in ids:
                owner = target_owner.setdefault(target, s1)
                union(s1, owner)

    # Root can depend on union order; the lexicographically smallest member is
    # stable for this fixed ground truth, independent of process hash randomization.
    selected_roots = {
        find(s1) for s1 in s1_ids if _hash_fraction(find(s1), seed) < fraction
    }
    return {s1 for s1 in s1_ids if find(s1) in selected_roots}


def write_split(
    truth_path: Path,
    output_path: Path,
    fraction: float = 0.10,
    seed: str = "amazon-er-v1",
) -> tuple[int, int]:
    """Write a two-column deterministic split manifest in truth-file order."""
    validation = grouped_validation_ids(truth_path, fraction, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_validation = 0
    with truth_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("source1_entity_id", "split"))
        for row in reader:
            is_validation = row["source1_entity_id"] in validation
            writer.writerow((row["source1_entity_id"], "validation" if is_validation else "train"))
            n_validation += is_validation
            n_train += not is_validation
    return n_train, n_validation
