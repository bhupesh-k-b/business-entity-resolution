# Pair matcher and set decision

`src/matching.py` computes 21 local features for each candidate returned by the
final blocking index. Name features cover exact normalized, legal-suffix-stripped,
and sorted-token equality; token Jaccard/containment; character trigram Dice;
prefix and length agreement. Address features cover exact, token and trigram
agreement, numeric overlap/conflict, and postal overlap. Three explicit
indicators describe country agreement and missing name/address, and one
indicates a Source 3 target. Missing fields never create exact agreement.

`src/run_validation.py` consumes the grouped `validation_split.tsv` and
deterministically samples the lowest hashed Source 1 IDs in each split. It fits
logistic regression and histogram gradient boosting on labeled blocked pairs.
Every retrieved positive and at most 25 top-ranked retrieved negatives per
training Source 1 are used. Validation retains **every** final capped candidate,
and includes Source 1 rows with zero candidates. Target records and labels come
only from the supplied training files.

The selected model is the one with highest validation macro F0.5 after a
deterministic global threshold sweep. Ties favor the higher threshold. A match
is emitted for each candidate whose score meets that threshold; no other ID can
be predicted. The score and decision files are `matcher.joblib` and
`decision_policy.json`. `validation_report.json`, score TSV, threshold sweeps,
and error examples document the choice.

The sampled validation score is a model-selection estimate and may be
optimistic because the same sample chooses the model and threshold. Blocking
recall is a hard upper bound on link recall and is reported separately.

No pretrained model or external business data is used. The exported, locally
trained model artifact is released under MIT in `MODEL_LICENSE.txt`; scikit-learn
is used as a BSD-3-Clause implementation dependency.
