"""Stream the complete test split through the frozen blocker and pair matcher.

The candidate file records exactly the candidate IDs passed to ``pair_features``.
Output files are replaced only after the complete inference run succeeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import sklearn

from blocking import build_index, open_index
from matching import FEATURE_NAMES, pair_features
from models import predict_probabilities
from normalization import NORMALIZATION_VERSION, normalize_record


EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _source_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_SOURCE_COLUMNS:
            raise ValueError(f"Unexpected source columns in {path}: {reader.fieldnames}")
        for row in reader:
            if None in row or any(row[field] is None for field in EXPECTED_SOURCE_COLUMNS):
                raise ValueError(f"Malformed source row in {path}")
            if not row["entity_id"].startswith("S1-"):
                raise ValueError(f"Unexpected Source 1 ID: {row['entity_id']}")
            yield row


def _load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if tuple(policy["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Decision policy feature order differs from matcher feature order")
    threshold = float(policy["threshold"])
    if not 0 <= threshold <= 1 or not np.isfinite(threshold):
        raise ValueError("Decision threshold must be finite and between 0 and 1")
    policy["threshold"] = threshold
    return policy


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    policy = _load_policy(args.policy)
    model = joblib.load(args.model)
    if not (args.index / "manifest.json").is_file():
        if not args.build_index_if_missing:
            raise FileNotFoundError(f"Missing blocking index: {args.index}")
        build_index(args.source2, args.source3, args.index)
    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    expected_paths = [str(args.source2.resolve()), str(args.source3.resolve())]
    if manifest["source_paths"] != expected_paths:
        raise ValueError("Blocking index source paths do not match the requested test targets")
    if args.batch_pairs < 1 or args.target_cache_size < 0:
        raise ValueError("batch-pairs must be positive and target-cache-size nonnegative")

    args.output.mkdir(parents=True, exist_ok=True)
    matching_path = args.output / "matching_results.tsv"
    candidate_path = args.output / "candidate_pairs.tsv"
    matching_tmp = args.output / "matching_results.tsv.partial"
    candidate_tmp = args.output / "candidate_pairs.tsv.partial"
    target_cache: OrderedDict[int, object] = OrderedDict()
    counters = {"source1_rows": 0, "candidate_pairs": 0, "matched_pairs": 0,
                "empty_candidate_rows": 0, "empty_match_rows": 0, "france_rows": 0,
                "model_batches": 0}

    def get_target(index, row_id: int):
        if args.target_cache_size and row_id in target_cache:
            target_cache.move_to_end(row_id)
            return target_cache[row_id]
        entity_id, name, address, country = index.get_record(row_id)
        if not entity_id.startswith(("S2-", "S3-")):
            raise ValueError(f"Invalid indexed target ID: {entity_id}")
        normalized = normalize_record(name, address, country)
        if args.target_cache_size:
            target_cache[row_id] = normalized
            if len(target_cache) > args.target_cache_size:
                target_cache.popitem(last=False)
        return normalized

    pending_rows: list[tuple[str, list[str], int]] = []
    pending_features: list[tuple[float, ...]] = []

    def flush(matching_writer, candidate_writer):
        if not pending_rows:
            return
        if pending_features:
            matrix = np.asarray(pending_features, dtype=np.float32)
            scores = predict_probabilities(model, matrix)
            if len(scores) != len(pending_features) or not np.isfinite(scores).all():
                raise ValueError("Matcher returned invalid candidate probabilities")
            counters["model_batches"] += 1
        else:
            scores = np.empty(0, dtype=np.float32)
        offset = 0
        for source_id, ids, count in pending_rows:
            selected = [entity_id for entity_id, score in
                        zip(ids, scores[offset:offset + count])
                        if score >= policy["threshold"]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate candidate ID for {source_id}")
            matching_writer.writerow((source_id, ",".join(selected)))
            candidate_writer.writerow((source_id, ",".join(ids)))
            counters["matched_pairs"] += len(selected)
            counters["empty_match_rows"] += not selected
            offset += count
        if offset != len(scores):
            raise ValueError("Candidate and score count mismatch")
        pending_rows.clear()
        pending_features.clear()

    try:
        with open_index(args.index) as index, \
             matching_tmp.open("w", newline="", encoding="utf-8") as matching_file, \
             candidate_tmp.open("w", newline="", encoding="utf-8") as candidate_file:
            matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            matching_writer.writerow(("source1_entity_id", "matched_entity_ids"))
            candidate_writer.writerow(("source1_entity_id", "candidate_entity_ids"))
            for row in _source_rows(args.source1):
                source_id = row["entity_id"]
                source = normalize_record(row["business_name"],
                                          row["business_address"], row["country"])
                candidates = index.candidate_details(source)
                if len(pending_features) + len(candidates) > args.batch_pairs:
                    flush(matching_writer, candidate_writer)
                ids = []
                for candidate in candidates:
                    ids.append(candidate.entity_id)
                    pending_features.append(pair_features(
                        source, get_target(index, candidate.row_id), candidate.entity_id))
                pending_rows.append((source_id, ids, len(ids)))
                counters["source1_rows"] += 1
                counters["candidate_pairs"] += len(ids)
                counters["empty_candidate_rows"] += not ids
                counters["france_rows"] += source.country == "france"
                if args.progress_every and counters["source1_rows"] % args.progress_every == 0:
                    flush(matching_writer, candidate_writer)
                    print(json.dumps({"progress_rows": counters["source1_rows"],
                                      "candidate_pairs": counters["candidate_pairs"],
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}),
                          file=sys.stderr, flush=True)
            flush(matching_writer, candidate_writer)
        os.replace(matching_tmp, matching_path)
        os.replace(candidate_tmp, candidate_path)
    except BaseException:
        matching_tmp.unlink(missing_ok=True)
        candidate_tmp.unlink(missing_ok=True)
        raise

    metadata = {
        "inputs": {"source1": str(args.source1.resolve()),
                   "source2": expected_paths[0], "source3": expected_paths[1],
                   "index": str(args.index.resolve()),
                   "policy": str(args.policy.resolve()), "model": str(args.model.resolve())},
        "output": {"matching": str(matching_path.resolve()),
                   "candidate": str(candidate_path.resolve())},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
                     "normalization": NORMALIZATION_VERSION},
        "parameters": {"model": policy.get("model"), "threshold": policy["threshold"],
                       "feature_names": list(FEATURE_NAMES),
                       "blocking": manifest["config"], "batch_pairs": args.batch_pairs,
                       "target_cache_size": args.target_cache_size,
                       "model_random_seed": 17,
                       "model_sample_seed": "model-sample-v1",
                       "validation_split_seed": "amazon-er-v1"},
        "index": {"target_count": manifest["target_count"],
                  "posting_count": manifest["posting_count"]},
        "counts": counters,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }
    (args.output / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-pairs", type=int, default=8192)
    parser.add_argument("--target-cache-size", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--build-index-if-missing", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
