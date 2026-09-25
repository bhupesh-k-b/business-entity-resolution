# Business Entity Resolution

Deterministic entity-resolution pipeline for matching Source 1 business records
to matching records in Source 2 and Source 3. The pipeline normalizes text,
builds an auditable on-disk blocking index, trains a local pair classifier, and
writes the required matching and candidate TSV files.

## Repository Contents

- `code/business_entity_resolution/src/`: normalization, blocking, matching,
  validation, and inference code.
- `code/business_entity_resolution/tests/`: focused blocking and normalization
  tests.
- `code/business_entity_resolution/requirements.txt`: pinned Python
  dependencies.
- `student_resource/utils/validate_submission.py`: stdlib-only submission
  validator.

The supplied datasets, generated indexes, trained model, reports, and inference
outputs are intentionally excluded from Git. Place the challenge dataset under
`student_resource/dataset/` before running the pipeline.

## Environment

Python 3.13 is recommended. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

Set convenient paths:

```bash
DATA=student_resource/dataset
PKG=code/business_entity_resolution
PY=.venv/bin/python
```

## Run

Audit the input tables and create the deterministic validation split:

```bash
$PY "$PKG/src/audit_data.py" \
  --data-dir "$DATA" \
  --output "$PKG/reports/data_audit.json"

$PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "code/business_entity_resolution/src"); from data_contract import write_split; write_split(Path("student_resource/dataset/train/train_ground_truth.tsv"), Path("code/business_entity_resolution/reports/validation_split.tsv"))'
```

Build the training index and fit the matcher:

```bash
$PY -c 'import sys; sys.path.insert(0, "code/business_entity_resolution/src"); from blocking import build_index; build_index("student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv", "code/business_entity_resolution/cache/train_index")'

$PY "$PKG/src/run_validation.py" \
  --source1 "$DATA/train/train_source1.tsv" \
  --truth "$DATA/train/train_ground_truth.tsv" \
  --split "$PKG/reports/validation_split.tsv" \
  --index "$PKG/cache/train_index" \
  --output "$PKG/reports"
```

Run test inference:

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

Validate the final files:

```bash
$PY student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "$DATA/test" \
  --check-ids
```

## Design Notes

- Blocking is bounded and produces the exact candidate set sent to the matcher.
- Validation uses a deterministic grouped split to avoid target-label leakage.
- The model is trained locally from the supplied labels; no pretrained model or
  external business data is used.
- The inference writer replaces final files only after the complete run succeeds.