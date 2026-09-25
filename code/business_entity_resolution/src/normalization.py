"""Deterministic, local-only text views for business entity resolution.

The original values should be kept by the caller. These views are intentionally
lossy and must never be treated as proof that two records are the same entity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata


NORMALIZATION_VERSION = "1.0.0"

# These are intentionally short, explicit lists. A change requires a version bump.
LEGAL_SUFFIXES = frozenset(
    {"inc", "incorporated", "corp", "corporation", "llc", "llp", "ltd",
     "limited", "gmbh", "sarl", "sas"}
)
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "hwy": "highway",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
}

_SPACE = re.compile(r"\s+")
_POSTAL = re.compile(r"(?:[0-9]{5}|[0-9]{6}|[0-9]{9})\Z")


@dataclass(frozen=True, slots=True)
class NameViews:
    present: bool
    norm: str
    core: str
    tokens: tuple[str, ...]
    sorted_tokens: str
    compact: str


@dataclass(frozen=True, slots=True)
class AddressViews:
    present: bool
    norm: str
    tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    postal_tokens: tuple[str, ...]
    compact: str


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    name: NameViews
    address: AddressViews
    country: str


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    # pandas.NA is not a string, but its representation is not useful evidence.
    if type(value).__name__ == "NAType":
        return ""
    return str(value)


def _normalize_text(value: object, *, name: bool) -> str:
    text = unicodedata.normalize("NFKC", _as_text(value)).casefold()
    out: list[str] = []
    for char in text:
        if char == "&":
            out.append(" and ")
        elif char == "+" and name:
            out.append(" plus ")
        elif char in {"'", "’", "ʼ", "`"}:
            # Dropping internal apostrophes keeps O'Reilly and OReilly together.
            continue
        elif char.isspace() or unicodedata.category(char)[0] in {"P", "S", "C"}:
            out.append(" ")
        else:
            out.append(char)
    return _SPACE.sub(" ", "".join(out)).strip()


def _compact(value: str) -> str:
    return "".join(char for char in value if char.isalnum())


def _strip_legal_suffix(tokens: tuple[str, ...]) -> tuple[str, ...]:
    end = len(tokens)
    while end > 1:
        token = tokens[end - 1]
        if token in LEGAL_SUFFIXES:
            end -= 1
        elif token in {"private", "pvt"} and end < len(tokens):
            # Only strip private/pvt when it precedes a removed suffix.
            end -= 1
        else:
            break
    return tokens[:end]


def normalize_name(value: object) -> NameViews:
    """Return raw-normalized and additional name views without transliteration."""
    norm = _normalize_text(value, name=True)
    tokens = tuple(norm.split())
    core = " ".join(_strip_legal_suffix(tokens))
    return NameViews(
        present=bool(norm),
        norm=norm,
        core=core,
        tokens=tokens,
        sorted_tokens=" ".join(sorted(tokens)),
        compact=_compact(norm),
    )


def normalize_address(value: object) -> AddressViews:
    """Return address views and possible numeric/postal components."""
    raw_norm = _normalize_text(value, name=False)
    tokens = tuple(ADDRESS_ABBREVIATIONS.get(t, t) for t in raw_norm.split())
    norm = " ".join(tokens)
    numeric = tuple(t for t in tokens if any(c.isdigit() for c in t))
    postal = tuple(t for t in numeric if _POSTAL.fullmatch(t))
    return AddressViews(
        present=bool(norm),
        norm=norm,
        tokens=tokens,
        numeric_tokens=numeric,
        postal_tokens=postal,
        compact=_compact(norm),
    )


def normalize_country(value: object) -> str:
    """Normalize any country label; no fixed vocabulary is assumed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", _as_text(value)).casefold()).strip()


def normalize_record(name: object, address: object, country: object) -> NormalizedRecord:
    return NormalizedRecord(normalize_name(name), normalize_address(address), normalize_country(country))


def char_ngrams(value: str, n: int = 3) -> tuple[str, ...]:
    """Return contiguous character n-grams from a normalized/compact view."""
    if n < 1:
        raise ValueError("n must be positive")
    if not value:
        return ()
    if len(value) < n:
        return (value,)
    return tuple(value[i : i + n] for i in range(len(value) - n + 1))
