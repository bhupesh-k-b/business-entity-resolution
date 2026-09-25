"""Train and validate a local pair matcher on blocked candidates.

The split manifest is created from the full ground truth by data_contract.py.
This program selects the lowest hash-valued IDs in each split, avoiding a
first-N geographic/order bias. Validation scores every final blocked candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from blocking import open_index
from data_contract import parse_ids
from matching import FEATURE_NAMES, pair_features
from models import fit_pair_models, predict_probabilities
from normalization import NORMALIZATION_VERSION, normalize_record


def _hash_id(value: str, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b((salt + value).encode(), digest_size=8).digest(), "big")


def select_ids(path: Path, train_rows: int, validation_rows: int,
               evaluation_rows: int = 0) -> tuple[set[str], set[str], set[str]]:
    """Select disjoint model, tuning, and final-evaluation IDs with O(K) memory."""
    heaps: dict[str, list[tuple[int, str]]] = {"train": [], "validation": []}
    limits = {"train": train_rows, "validation": validation_rows + evaluation_rows}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "split"]:
            raise ValueError("unexpected split manifest columns")
        for row in reader:
            group, entity_id = row["split"], row["source1_entity_id"]
            if group not in heaps:
                raise ValueError(f"unknown split: {group}")
            heap, limit = heaps[group], limits[group]
            if not limit:
                continue
            item = (-_hash_id(entity_id, "model-sample-v1:"), entity_id)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    held_out = sorted(((-score, entity_id) for score, entity_id in heaps["validation"]))
    return ({entity_id for _, entity_id in heaps["train"]},
            {entity_id for _, entity_id in held_out[:validation_rows]},
            {entity_id for _, entity_id in held_out[validation_rows:]})


def load_selected_truth(path: Path, selected: set[str]) -> dict[str, set[str]]:
    truth = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            entity_id = row["source1_entity_id"]
            if entity_id in selected:
                truth[entity_id] = set(parse_ids(row["matched_entity_ids"]))
    if truth.keys() != selected:
        raise ValueError(f"truth missing {len(selected - truth.keys())} sampled IDs")
    return truth


def collect_pairs(source1_path: Path, index_dir: Path, truth: dict[str, set[str]],
                  *, train: bool, negatives_per_row: int = 25) -> dict:
    """Collect labeled pairs and row topology; retain all validation candidates."""
    x, y, groups, candidate_ids, source_ids, true_counts = [], [], [], [], [], []
    total_links = found_links = zero_candidates = 0
    target_cache = {}
    with open_index(index_dir) as idx, source1_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            source_id = row["entity_id"]
            if source_id not in truth:
                continue
            source = normalize_record(row["business_name"], row["business_address"], row["country"])
            candidates = idx.candidate_details(source)
            actual = truth[source_id]
            total_links += len(actual)
            found_links += len(actual & {c.entity_id for c in candidates})
            zero_candidates += not candidates
            group = len(source_ids)
            source_ids.append(source_id)
            true_counts.append(len(actual))
            if train:
                positives = [c for c in candidates if c.entity_id in actual]
                negatives = [c for c in candidates if c.entity_id not in actual][:negatives_per_row]
                candidates = positives + negatives
            for c in candidates:
                target = target_cache.get(c.row_id)
                if target is None:
                    _, name, address, country = idx.get_record(c.row_id)
                    target = normalize_record(name, address, country)
                    if len(target_cache) >= 60_000:
                        target_cache.clear()
                    target_cache[c.row_id] = target
                x.append(pair_features(source, target, c.entity_id))
                y.append(c.entity_id in actual)
                groups.append(group)
                candidate_ids.append(c.entity_id)
    if len(source_ids) != len(truth):
        raise ValueError(f"source1 missing {len(truth) - len(source_ids)} sampled IDs")
    return {
        "x": np.asarray(x, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
        "y": np.asarray(y, dtype=np.uint8),
        "groups": np.asarray(groups, dtype=np.int32),
        "candidate_ids": candidate_ids,
        "source_ids": source_ids,
        "true_counts": np.asarray(true_counts, dtype=np.int32),
        "total_links": total_links,
        "found_links": found_links,
        "zero_candidates": zero_candidates,
    }


def evaluate(data: dict, probabilities: np.ndarray, threshold: float) -> dict:
    """Macro entity F0.5, singleton correctness, and micro link diagnostics."""
    chosen = probabilities >= threshold
    n = len(data["source_ids"])
    pred_counts = np.bincount(data["groups"], weights=chosen, minlength=n)
    tp = np.bincount(data["groups"], weights=chosen & data["y"].astype(bool), minlength=n)
    actual = data["true_counts"]
    positive_actual = actual > 0
    denom = 0.25 * actual + pred_counts
    scores = np.divide(1.25 * tp, denom, out=np.zeros(n), where=denom > 0)
    scores[~positive_actual] = (pred_counts[~positive_actual] == 0).astype(float)
    singleton = ~positive_actual
    total_tp = int(tp.sum())
    total_pred = int(pred_counts.sum())
    return {
        "macro_f05": float(scores.mean()),
        "link_precision": total_tp / total_pred if total_pred else 0.0,
        "link_recall": total_tp / int(actual.sum()) if actual.sum() else 0.0,
        "singleton_accuracy": float((pred_counts[singleton] == 0).mean()) if singleton.any() else None,
        "predicted_counts": {str(k): int(v) for k, v in sorted(Counter(pred_counts.astype(int)).items())},
        "false_merge_rows": int(((pred_counts > tp) & (pred_counts > 0)).sum()),
        "missed_link_rows": int((tp < actual).sum()),
        "singleton_false_merges": int(((pred_counts > 0) & singleton).sum()),
    }


def tune(data: dict, probabilities: np.ndarray) -> tuple[float, dict, list[dict]]:
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99),
                                  np.quantile(probabilities, np.linspace(0.01, 0.99, 99)),
                                  0.999, 1.0])
    rows = []
    for t in thresholds:
        result = evaluate(data, probabilities, float(t))
        rows.append({"threshold": float(t), **result})
    best = max(rows, key=lambda row: (row["macro_f05"], row["threshold"]))
    return best["threshold"], best, rows


def write_diagnostics(out: Path, data: dict, scores: np.ndarray, threshold: float,
                      truth: dict[str, set[str]], prefix: str) -> None:
    with (out / f"{prefix}_scores.tsv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(("source1_entity_id", "candidate_entity_id", "label", "score"))
        for group, target, label, score in zip(data["groups"], data["candidate_ids"], data["y"], scores):
            writer.writerow((data["source_ids"][group], target, int(label), f"{score:.8f}"))
    examples = []
    offset = 0
    for group, source_id in enumerate(data["source_ids"]):
        # Candidate rows are contiguous by source. Advance via precomputed group boundaries.
        next_offset = int(np.searchsorted(data["groups"], group + 1, side="left"))
        ranked = sorted(((data["candidate_ids"][i], float(scores[i]))
                         for i in range(offset, next_offset)), key=lambda item: -item[1])
        selected = [(target, score) for target, score in ranked if score >= threshold]
        actual = truth[source_id]
        predicted = {c for c, _ in selected}
        kind = ("singleton_false_merge" if not actual and predicted else
                "false_merge" if predicted - actual else
                "missed_link" if actual - predicted else None)
        if kind and len(examples) < 100:
            examples.append({"source1_entity_id": source_id, "kind": kind,
                             "truth": sorted(actual), "predicted": sorted(predicted),
                             "top_scores": ranked[:8]})
        offset = next_offset
    (out / f"{prefix}_errors.json").write_text(json.dumps(examples, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=15_000)
    parser.add_argument("--validation-rows", type=int, default=5_000)
    parser.add_argument("--evaluation-rows", type=int, default=5_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, eval_ids = select_ids(
        args.split, args.train_rows, args.validation_rows, args.evaluation_rows
    )
    selected_truth = load_selected_truth(args.truth, train_ids | val_ids | eval_ids)
    train = collect_pairs(args.source1, args.index, {k: selected_truth[k] for k in train_ids}, train=True)
    val_truth = {k: selected_truth[k] for k in val_ids}
    validation = collect_pairs(args.source1, args.index, val_truth, train=False)
    eval_truth = {k: selected_truth[k] for k in eval_ids}
    final_evaluation = collect_pairs(args.source1, args.index, eval_truth, train=False)
    models = fit_pair_models(train["x"], train["y"])
    comparison = {}
    winner = None
    for name, model in models.items():
        scores = predict_probabilities(model, validation["x"])
        threshold, best, sweep = tune(validation, scores)
        comparison[name] = best
        (args.output / f"threshold_sweep_{name}.json").write_text(json.dumps(sweep, indent=2) + "\n")
        if winner is None or (best["macro_f05"], threshold) > (winner[2]["macro_f05"], winner[1]):
            winner = (name, threshold, best, scores)
    assert winner is not None
    winner_name, threshold, best, scores = winner
    joblib.dump(models[winner_name], args.output / "matcher.joblib")
    (args.output / "MODEL_LICENSE.txt").write_text(
        "MIT License\n\n"
        "Copyright (c) 2026 Amazon ML Challenge participant\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this model artifact and associated documentation files (the \"Software\"),\n"
        "to deal in the Software without restriction, including without limitation\n"
        "the rights to use, copy, modify, merge, publish, distribute, sublicense,\n"
        "and/or sell copies of the Software, and to permit persons to whom the\n"
        "Software is furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included\n"
        "in all copies or substantial portions of the Software.\n\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS\n"
        "OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\n"
        "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n"
        "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY\n"
        "CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,\n"
        "TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE\n"
        "SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
    )
    index_manifest = json.loads((args.index / "manifest.json").read_text())
    policy = {"model": winner_name, "threshold": threshold, "feature_names": FEATURE_NAMES,
              "normalization_version": NORMALIZATION_VERSION,
              "blocking_config": index_manifest["config"],
              "model_license": "MIT", "pretrained_model_used": False,
              "decision": "score each candidate independently; include score >= threshold"}
    (args.output / "decision_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    write_diagnostics(args.output, validation, scores, threshold, val_truth, "validation")
    final_scores = predict_probabilities(models[winner_name], final_evaluation["x"])
    final_metrics = evaluate(final_evaluation, final_scores, threshold)
    write_diagnostics(args.output, final_evaluation, final_scores, threshold,
                      eval_truth, "evaluation")
    calibration = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        calibration.append({"score_interval": [float(lo), float(hi)],
                            "pairs": int(mask.sum()),
                            "observed_link_rate": float(validation["y"][mask].mean()) if mask.any() else None})
    report = {
        "sample_seed": "model-sample-v1", "train_rows": len(train_ids),
        "validation_rows": len(val_ids), "evaluation_rows": len(eval_ids),
        "train_pairs": len(train["y"]),
        "validation_pairs": len(validation["y"]),
        "evaluation_pairs": len(final_evaluation["y"]),
        "candidate_recall_train": train["found_links"] / train["total_links"],
        "candidate_recall_validation": validation["found_links"] / validation["total_links"],
        "candidate_recall_evaluation": final_evaluation["found_links"] / final_evaluation["total_links"],
        "zero_candidate_validation_rows": validation["zero_candidates"],
        "zero_candidate_evaluation_rows": final_evaluation["zero_candidates"],
        "score_calibration": calibration,
        "model_comparison": comparison, "selected_model": winner_name,
        "selected_threshold": threshold,
        "final_evaluation": final_metrics,
        "logistic_standardized_coefficients": {
            name: float(weight) for name, weight in zip(
                FEATURE_NAMES, models["logistic"].named_steps["logisticregression"].coef_[0]
            )
        },
    }
    (args.output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
