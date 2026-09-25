"""Local pair features for business entity resolution.

The input records use the versioned normalizer in ``normalization.py``.  Every
feature is computed from the two supplied records; no external data is used.
"""

from __future__ import annotations

from functools import lru_cache

from normalization import NormalizedRecord


FEATURE_NAMES = (
    "name_exact",
    "name_core_exact",
    "name_sorted_exact",
    "name_token_jaccard",
    "name_token_containment",
    "name_char3_dice",
    "name_core_char3_dice",
    "name_prefix3",
    "name_length_ratio",
    "address_exact",
    "address_token_jaccard",
    "address_token_containment",
    "address_char3_dice",
    "address_numeric_jaccard",
    "address_numeric_overlap",
    "address_numeric_conflict",
    "address_postal_overlap",
    "country_agreement",
    "name_missing_either",
    "address_missing_either",
    "target_is_source3",
)


@lru_cache(maxsize=200_000)
def _ngrams(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    if len(value) < 3:
        return frozenset((value,))
    return frozenset(value[i : i + 3] for i in range(len(value) - 2))


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def _containment(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    a, b = set(left), set(right)
    return len(a & b) / min(len(a), len(b))


def _char_dice(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    a, b = _ngrams(left), _ngrams(right)
    return 2.0 * len(a & b) / (len(a) + len(b))


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def pair_features(
    source: NormalizedRecord, target: NormalizedRecord, target_id: str
) -> tuple[float, ...]:
    """Return fixed-order numeric features for a candidate pair.

    Missing values deliberately score zero for all equality features; two
    missing fields never become an artificial exact match.
    """
    n1, n2 = source.name, target.name
    a1, a2 = source.address, target.address
    num1, num2 = set(a1.numeric_tokens), set(a2.numeric_tokens)
    shared_num = bool(num1 & num2)
    both_numeric = bool(num1 and num2)
    postal1, postal2 = set(a1.postal_tokens), set(a2.postal_tokens)
    return (
        float(bool(n1.norm and n2.norm and n1.norm == n2.norm)),
        float(bool(n1.core and n2.core and n1.core == n2.core)),
        float(bool(n1.sorted_tokens and n2.sorted_tokens and n1.sorted_tokens == n2.sorted_tokens)),
        _jaccard(n1.tokens, n2.tokens),
        _containment(n1.tokens, n2.tokens),
        _char_dice(n1.compact, n2.compact),
        _char_dice(n1.core.replace(" ", ""), n2.core.replace(" ", "")),
        float(bool(n1.compact and n2.compact and n1.compact[:3] == n2.compact[:3])),
        _length_ratio(n1.compact, n2.compact),
        float(bool(a1.norm and a2.norm and a1.norm == a2.norm)),
        _jaccard(a1.tokens, a2.tokens),
        _containment(a1.tokens, a2.tokens),
        _char_dice(a1.compact, a2.compact),
        _jaccard(a1.numeric_tokens, a2.numeric_tokens),
        float(shared_num),
        float(both_numeric and not shared_num),
        float(bool(postal1 and postal2 and postal1 & postal2)),
        float(bool(source.country and target.country and source.country == target.country)),
        float(not (n1.present and n2.present)),
        float(not (a1.present and a2.present)),
        float(target_id.startswith("S3-")),
    )
