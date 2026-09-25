# Normalization specification

Version: `1.0.0` (`NORMALIZATION_VERSION` in `src/normalization.py`). All transformations run locally with Python's standard library and use no labels or external data.

## API and stable fields

`normalize_name(value)` returns `NameViews(present, norm, core, tokens, sorted_tokens, compact)`.

`normalize_address(value)` returns `AddressViews(present, norm, tokens, numeric_tokens, postal_tokens, compact)`.

`normalize_country(value)` returns a normalized string. `normalize_record(name, address, country)` combines these views. `char_ngrams(value, n=3)` returns contiguous character n-grams, usually from a `compact` view. Missing values yield `present=False`, empty strings and empty tuples. Downstream indexing and exact-match features must check `present` before using an empty key. The caller must retain the original columns for audit and comparison.

## Name transformations

1. Unicode NFKC normalization and case folding. This handles full-width Latin characters and casing while preserving Devanagari and other scripts. It does not transliterate or remove accents; `café` and `cafe` remain distinct.
2. Replace `&` with `and` and `+` with `plus`. Remove apostrophes, including curly apostrophes, so `O’Reilly` becomes `oreilly`. Other Unicode punctuation, symbols and control characters separate tokens. Collapse whitespace.
3. `norm` is the resulting text, `tokens` is its ordered token tuple, `sorted_tokens` sorts the original tokens lexically, and `compact` removes spaces and keeps Unicode alphanumeric characters.
4. `core` removes only terminal legal suffixes: `inc`, `incorporated`, `corp`, `corporation`, `llc`, `llp`, `ltd`, `limited`, `gmbh`, `sarl`, `sas`. `private` and `pvt` are removed only immediately before a removed suffix. At least one token remains. The full `norm` remains available, so this view does not destroy legal-form evidence. Ambiguous `co`, `company`, `sa`, and generic words are retained.

Examples: `B+ Retail Inc` → `norm="b plus retail inc"`, `core="b plus retail"`; `Red Ventures Private Limited` → `core="red ventures"`; `रेड वेंचर्स` remains in Devanagari. Script differences must be handled by address or learned similarity features, not an invented transliteration.

## Address transformations

The same Unicode/case/punctuation/whitespace rules apply. A short, explicit token map expands `rd→road`, `ave/av→avenue`, `blvd→boulevard`, `ln→lane`, `hwy→highway`, `apt→apartment`, `ste→suite`, and `bldg→building`. `st` and `dr` are deliberately retained because they may mean Saint/Street and Doctor/Drive. The expanded sequence is `norm` and `tokens`; `compact` removes spaces. `numeric_tokens` contains any token with a digit. `postal_tokens` contains all standalone ASCII 5, 6, or 9 digit tokens. They are *candidates*, not validated postal codes; house numbers can have the same length, and other countries use different patterns.

Example: `12 Rd, Apt 3` → `norm="12 road apartment 3"`, `numeric_tokens=("12", "3")`. An address missing its postal code still has the rest of its evidence.

## Country and missing values

Country labels receive NFKC, case folding, whitespace collapsing and trimming. There is no fixed country enum; unseen values such as `France` remain usable. `None`, NaN, pandas `NA`, empty and whitespace-only values are missing. The literal string `nan` remains present, avoiding a false missing-value interpretation of a legitimate raw string.

## Collision risk

The name `core` is a retrieval/similarity view, never an identity key. In the first 200,000 training Source 1 rows, 1,039 of 171,648 nonempty `norm` keys represented multiple distinct raw name strings. Removing legal suffixes increased this to 8,629 of 158,788 `core` keys containing multiple distinct `norm` strings. These counts were measured without inspecting ground-truth matches. Examples with the same `core` but widely separated addresses include `Helios` (Connecticut), `Helios LLC` (Kentucky), and `Helios Corporation` (Texas); `Dream Construction Limited` (Hyderabad) and `Dream Construction Private Limited` (Delhi); `Apex Inc` (West Virginia) and `Apex` (Oregon). Their common core alone is insufficient evidence of a match. Sorted tokens and compact strings create further collisions, especially for short or generic names. Address and country evidence, plus calibrated pair scoring, must decide matches.

Changing these mappings or semantics requires a version bump and re-evaluation of downstream blocking and model features.
