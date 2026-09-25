# Business entity resolution pipeline

This package uses only the seven supplied TSV files. It normalizes name and
address text, builds an on-disk blocking index, trains a pair classifier on a
deterministic training sample, selects a threshold on a disjoint validation
sample, and streams every test Source 1 row to the two required output TSVs.
No external business lookup, geocoding, API, AWS service, or pretrained model
is used. France is processed as an ordinary open-set country label.

## Environment

Python 3.13 is used for the included version pins. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

If the supplied Conda interpreter has these exact versions, it can be used as
`PY` below. The development run used `/opt/miniconda3/bin/python`.

## Reproduce from the supplied dataset

Run from the repository root. Set `DATA` to the supplied `dataset` directory,
`PKG` to this package, and `PY` to the prepared Python interpreter:

```bash
DATA=student_resource/dataset
PKG=code/business_entity_resolution
PY=.venv/bin/python
```

Audit the seven input tables and produce a deterministic validation split:

```bash
$PY "$PKG/src/audit_data.py" --data-dir "$DATA" --output "$PKG/reports/data_audit.json"
$PY -c 'import sys; from pathlib import Path; sys.path.insert(0,"code/business_entity_resolution/src"); from data_contract import write_split; write_split(Path("student_resource/dataset/train/train_ground_truth.tsv"), Path("code/business_entity_resolution/reports/validation_split.tsv"))'
```

Build the training target index. This is an unsupervised index over the raw
training Source 2 and Source 3 tables. Its target offsets refer to these files,
so keep them at the same paths after indexing:

```bash
$PY -c 'import sys; sys.path.insert(0,"code/business_entity_resolution/src"); from blocking import build_index; build_index("student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv", "code/business_entity_resolution/cache/train_index")'
```

Fit candidate pair models on sampled training Source 1 IDs, select the model
and F0.5 decision threshold on sampled validation IDs, and save the frozen
`matcher.joblib` and `decision_policy.json`:

```bash
$PY "$PKG/src/run_validation.py" \
  --source1 "$DATA/train/train_source1.tsv" \
  --truth "$DATA/train/train_ground_truth.tsv" \
  --split "$PKG/reports/validation_split.tsv" \
  --index "$PKG/cache/train_index" \
  --output "$PKG/reports"
```

Stream test inference. The flag builds the test target index only if absent.
The index may take substantial disk space and time to construct; inference
uses a fixed pair batch and a bounded normalized target cache.

```bash
$PY "$PKG/src/run_inference.py" \
  --source1 "$DATA/test/test_source1.tsv" \
  --source2 "$DATA/test/test_source2.tsv" \
  --source3 "$DATA/test/test_source3.tsv" \
  --index "$PKG/cache/test_index" \
  --build-index-if-missing \
  --model "$PKG/reports/matcher.joblib" \
  --policy "$PKG/reports/decision_policy.json" \
  --output output
```

Validate the exact output files, including target ID existence:

```bash
$PY student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "$DATA/test" --check-ids
```

`output/inference_metadata.json` records package versions, normalization
version, blocking configuration, threshold, row and pair counts, elapsed time,
and peak resident memory. The Source 1 input order and candidate ranking are
deterministic. Both TSVs are written to temporary files and replaced after a
successful complete run.

## Files

- `src/normalization.py`: deterministic local text views.
- `src/blocking.py`: on-disk target index and final capped candidate set.
- `src/matching.py`: fixed-order pair features.
- `src/models.py`: locally trained classifiers and probability inference.
- `src/run_validation.py`: training, validation, model selection, threshold.
- `src/run_inference.py`: bounded-memory full test scoring and exact TSV output.
- `src/data_contract.py`, `src/audit_data.py`: split, metric, input audit.

The serialized matcher contains weights fitted from the supplied training
labels. No downloaded model weights are included. The scikit-learn runtime is
BSD-3-Clause licensed; the locally produced model artifact and our pipeline
code can be released under the accompanying MIT license, subject to the team's
final license declaration.
