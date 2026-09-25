"""Bounded, auditable candidate generation for the challenge TSV files.

The index stores fixed-width target IDs, byte offsets into the original TSVs,
and sorted (key hash, target row) postings. It does not materialize millions of
Python target objects. The target TSVs must remain at their indexed paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import mmap
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from .normalization import normalize_record
except ImportError:
    from normalization import normalize_record


POSTING_DTYPE = np.dtype([("key", "<u8"), ("row", "<u4")])
ID_DTYPE = "S24"
PASS_WEIGHTS = {
    "name_exact": 5.0,
    "name_core": 4.5,
    "name_sorted": 4.0,
    "name_fragment": 1.8,
    "name_token": 2.2,
    "address_exact": 5.0,
    "address_number": 2.5,
    "address_numeric_only": 1.4,
    "address_tokens": 1.5,
}
_ADDRESS_STOP = frozenset({
    "street", "road", "avenue", "lane", "drive", "boulevard", "suite",
    "floor", "building", "near", "opposite", "main", "north", "south",
    "east", "west", "city", "district", "state", "india", "usa",
    "the", "and", "new", "old", "sector", "block", "area", "plot",
    "st", "rd", "ave", "ln", "dr", "nc", "ca", "tx", "ny", "fl",
})


@dataclass(frozen=True)
class BlockingConfig:
    max_postings_per_key: int = 500
    max_candidates: int = 160
    min_fragment_chars: int = 7
    chunk_rows: int = 50_000
    use_name_tokens: bool = False
    expanded_address: bool = False


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    row_id: int
    score: float
    passes: tuple[str, ...]


def _hash_key(pass_name: str, country: str, value: str) -> int:
    raw = (pass_name + "\x1f" + country + "\x1f" + value).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def _tokens(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(value.split())
    return tuple(value or ())


def _numeric_variants(token: str) -> tuple[str, ...]:
    """Treat 001/1 and 120C/120 as nearby house-number variants."""
    if not any(c.isdigit() for c in token):
        return ()
    variants = {token}
    canonical = re.sub(r"^0+(?=\d)", "", token)
    variants.add(canonical)
    if len(canonical) >= 3 and canonical[-1].isalpha() and canonical[:-1].isdigit():
        variants.add(canonical[:-1])
    return tuple(sorted(variants))


def record_keys(rec, config: BlockingConfig = BlockingConfig()) -> dict[str, tuple[int, ...]]:
    """Independent key families. All keys are country conditioned.

    Missing fields emit no keys. A country label is treated as an open string,
    rather than a fixed US/India/France enumeration.
    """
    country = rec.country or ""
    keys: dict[str, set[int]] = defaultdict(set)
    name = rec.name
    addr = rec.address
    if name.present:
        if name.norm:
            keys["name_exact"].add(_hash_key("ne", country, name.norm))
        if name.core:
            keys["name_core"].add(_hash_key("nc", country, name.core))
        if name.sorted_tokens:
            keys["name_sorted"].add(_hash_key("ns", country, name.sorted_tokens))
        if config.use_name_tokens:
            useful_name = {t for t in _tokens(name.tokens)
                           if len(t) >= 4 and t not in {"private", "limited", "pvt", "ltd", "inc", "corp", "llc", "llp"}}
            for token in sorted(useful_name, key=lambda t: (-len(t), t))[:5]:
                keys["name_token"].add(_hash_key("nt", country, token))
        compact = name.compact
        if len(compact) >= config.min_fragment_chars:
            # End fragments survive many middle-character edits and transpositions.
            keys["name_fragment"].add(_hash_key("nf", country, compact[:5]))
            keys["name_fragment"].add(_hash_key("nb", country, compact[-5:]))
    if addr.present:
        if addr.norm:
            keys["address_exact"].add(_hash_key("ae", country, addr.norm))
        useful = sorted({t for t in _tokens(addr.tokens)
                         if len(t) >= 4 and t not in _ADDRESS_STOP and not t.isdigit()},
                        key=lambda t: (-len(t), t))
        if config.expanded_address:
            # Include trailing locality terms as well as distinctive long words.
            trailing = [t for t in _tokens(addr.tokens) if t in useful][-3:]
            useful = list(dict.fromkeys(useful[:5] + trailing))[:8]
            raw_nums = sorted(set(_tokens(addr.numeric_tokens)))[:4]
            nums = tuple(sorted({v for token in raw_nums for v in _numeric_variants(token)}))
            for number in nums:
                if len(number) >= 3:
                    keys["address_numeric_only"].add(_hash_key("ao", country, number))
        else:
            useful = useful[:3]
            nums = sorted(set(_tokens(addr.numeric_tokens)))[:2]
        for number in nums:
            for token in useful:
                keys["address_number"].add(_hash_key("an", country, number + "|" + token))
        # Two location words can recover a pair with missing numeric addresses.
        if len(useful) >= 2:
            for i in range(min(len(useful), 3)):
                for j in range(i + 1, min(len(useful), 3)):
                    pair = "|".join(sorted((useful[i], useful[j])))
                    keys["address_tokens"].add(_hash_key("at", country, pair))
    return {p: tuple(sorted(v)) for p, v in keys.items()}


def _iter_tsv_with_offsets(path: Path):
    with path.open("rb") as fh:
        header = fh.readline().rstrip(b"\r\n").split(b"\t")
        expected = [b"entity_id", b"business_name", b"business_address", b"country"]
        if header != expected:
            raise ValueError(f"Unexpected TSV columns in {path}: {header!r}")
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            fields = line.rstrip(b"\r\n").decode("utf-8").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed TSV at {path}:{offset}")
            yield offset, fields


def build_index(source2_path: str | Path, source3_path: str | Path,
                index_dir: str | Path, config: BlockingConfig = BlockingConfig()) -> dict:
    """Build sorted on-disk postings; intended for one dataset split at a time."""
    paths = [Path(source2_path).resolve(), Path(source3_path).resolve()]
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    postings_path = out / "postings.bin"
    ids_path = out / "target_ids.bin"
    offsets_path = out / "target_offsets.bin"
    sources_path = out / "target_sources.bin"
    posting_buffer: list[tuple[int, int]] = []
    id_buffer: list[bytes] = []
    offset_buffer: list[int] = []
    source_buffer: list[int] = []
    row_count = 0
    posting_count = 0
    pass_counts = Counter()

    def flush(pf, idf, of, sf):
        nonlocal posting_count
        if not id_buffer:
            return
        np.asarray(id_buffer, dtype=ID_DTYPE).tofile(idf)
        np.asarray(offset_buffer, dtype="<u8").tofile(of)
        np.asarray(source_buffer, dtype="u1").tofile(sf)
        if posting_buffer:
            p = np.empty(len(posting_buffer), dtype=POSTING_DTYPE)
            p["key"] = [x[0] for x in posting_buffer]
            p["row"] = [x[1] for x in posting_buffer]
            p.tofile(pf)
            posting_count += len(p)
        posting_buffer.clear()
        id_buffer.clear()
        offset_buffer.clear()
        source_buffer.clear()

    with postings_path.open("wb") as pf, ids_path.open("wb") as idf, \
         offsets_path.open("wb") as of, sources_path.open("wb") as sf:
        for source_num, path in enumerate(paths):
            for offset, (entity_id, name, address, country) in _iter_tsv_with_offsets(path):
                if row_count >= 2**32:
                    raise OverflowError("Target count exceeds uint32 row IDs")
                expected_prefix = f"S{source_num + 2}-"
                if not entity_id.startswith(expected_prefix):
                    raise ValueError(f"Unexpected target ID {entity_id!r} in {path}")
                encoded = entity_id.encode("ascii")
                if len(encoded) > np.dtype(ID_DTYPE).itemsize:
                    raise ValueError(f"Target ID too long: {entity_id}")
                id_buffer.append(encoded)
                offset_buffer.append(offset)
                source_buffer.append(source_num)
                for pass_name, hashes in record_keys(
                        normalize_record(name, address, country), config).items():
                    pass_counts[pass_name] += len(hashes)
                    posting_buffer.extend((h, row_count) for h in hashes)
                row_count += 1
                if len(id_buffer) >= config.chunk_rows:
                    flush(pf, idf, of, sf)
        flush(pf, idf, of, sf)
    postings = np.memmap(postings_path, dtype=POSTING_DTYPE, mode="r+", shape=(posting_count,))
    postings.sort(order=["key", "row"], kind="quicksort")
    postings.flush()
    # Contiguous key storage makes batched binary searches much faster than a
    # strided structured-array field, especially for random memory-mapped IO.
    keys_path = out / "keys.bin"
    rows_path = out / "rows.bin"
    with keys_path.open("wb") as kf, rows_path.open("wb") as rf:
        for start in range(0, posting_count, 1_000_000):
            chunk = postings[start:start + 1_000_000]
            np.asarray(chunk["key"], dtype="<u8").tofile(kf)
            np.asarray(chunk["row"], dtype="<u4").tofile(rf)
    del postings
    postings_path.unlink()
    manifest = {
        "source_paths": [str(p) for p in paths],
        "target_count": row_count,
        "posting_count": posting_count,
        "pass_postings": dict(pass_counts),
        "config": asdict(config),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class BlockingIndex:
    def __init__(self, index_dir: str | Path):
        self.path = Path(index_dir)
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.config = BlockingConfig(**manifest["config"])
        self.target_count = manifest["target_count"]
        self.keys = np.memmap(self.path / "keys.bin", dtype="<u8",
                              mode="r", shape=(manifest["posting_count"],))
        self.rows = np.memmap(self.path / "rows.bin", dtype="<u4",
                              mode="r", shape=(manifest["posting_count"],))
        self.ids = np.memmap(self.path / "target_ids.bin", dtype=ID_DTYPE,
                             mode="r", shape=(self.target_count,))
        self.offsets = np.memmap(self.path / "target_offsets.bin", dtype="<u8",
                                 mode="r", shape=(self.target_count,))
        self.sources = np.memmap(self.path / "target_sources.bin", dtype="u1",
                                 mode="r", shape=(self.target_count,))
        self._files = [Path(p).open("rb") for p in manifest["source_paths"]]
        self._maps = [mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for f in self._files]

    def close(self):
        for m in self._maps:
            m.close()
        for f in self._files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_record(self, row_id: int) -> tuple[str, str, str, str]:
        """Return original entity_id, name, address, country for a target ordinal."""
        source = int(self.sources[row_id])
        mm = self._maps[source]
        start = int(self.offsets[row_id])
        end = mm.find(b"\n", start)
        if end < 0:
            end = len(mm)
        parts = mm[start:end].rstrip(b"\r").decode("utf-8").split("\t")
        if len(parts) != 4:
            raise ValueError(f"Malformed indexed target row {row_id}")
        return tuple(parts)

    def candidate_details(self, rec) -> list[Candidate]:
        """Exact final candidate set, sorted by blocking strength and target ID."""
        evidence: dict[int, float] = defaultdict(float)
        passes: dict[int, set[str]] = defaultdict(set)
        all_keys = record_keys(rec, self.config)
        probes = [(p, h) for p, hashes in all_keys.items() for h in hashes]
        if probes:
            lookup = np.fromiter((h for _, h in probes), dtype="<u8", count=len(probes))
            lows = np.searchsorted(self.keys, lookup, side="left")
            highs = np.searchsorted(self.keys, lookup, side="right")
        else:
            lows = highs = ()
        for (pass_name, _), lo, hi in zip(probes, lows, highs):
                lo, hi = int(lo), int(hi)
                count = hi - lo
                if count == 0 or count > self.config.max_postings_per_key:
                    continue
                weight = PASS_WEIGHTS[pass_name] / (1.0 + math.log1p(count) / 5.0)
                for row in self.rows[lo:hi]:
                    rid = int(row)
                    evidence[rid] += weight
                    passes[rid].add(pass_name)
        ranked = sorted(evidence, key=lambda rid: (-evidence[rid], bytes(self.ids[rid])))
        return [Candidate(bytes(self.ids[rid]).decode("ascii").rstrip("\x00"), rid,
                          evidence[rid], tuple(sorted(passes[rid])))
                for rid in ranked[:self.config.max_candidates]]

    def candidates(self, rec) -> list[str]:
        return [c.entity_id for c in self.candidate_details(rec)]

    def iter_candidates(self, source1_path: str | Path) -> Iterator[tuple[str, list[Candidate]]]:
        for _, (entity_id, name, address, country) in _iter_tsv_with_offsets(Path(source1_path)):
            if not entity_id.startswith("S1-"):
                raise ValueError(f"Unexpected source1 ID: {entity_id}")
            yield entity_id, self.candidate_details(normalize_record(name, address, country))


def open_index(index_dir: str | Path) -> BlockingIndex:
    return BlockingIndex(index_dir)


def evaluate_recall(index: BlockingIndex, source1_path: str | Path,
                    truth_path: str | Path, max_rows: int | None = None,
                    missed_examples: int = 20,
                    measure_raw_passes: bool = False) -> dict:
    """Evaluate the exact capped candidate set against known links.

    Use an index containing *all* target records for the evaluated split. The
    report includes per-pass retrieval, final recall, candidate counts, and
    concrete misses. `max_rows` takes the first N S1 rows for quick checks.
    """
    truth: dict[str, set[str]] = {}
    with Path(truth_path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    total_links = kept_links = rows = zero_candidates = 0
    pass_hits = Counter()
    raw_pass_hits = Counter()
    counts = []
    misses = []
    for _, (source_id, name, address, country) in _iter_tsv_with_offsets(Path(source1_path)):
        if max_rows is not None and rows >= max_rows:
            break
        rec = normalize_record(name, address, country)
        candidates = index.candidate_details(rec)
        actual = truth[source_id]
        found = {c.entity_id for c in candidates}
        rows += 1
        counts.append(len(candidates))
        zero_candidates += not candidates
        total_links += len(actual)
        kept_links += len(actual & found)
        for c in candidates:
            if c.entity_id in actual:
                pass_hits.update(c.passes)
        if measure_raw_passes and actual:
            for pass_name, hashes in record_keys(rec, index.config).items():
                pass_found = set()
                for h in hashes:
                    lo = int(np.searchsorted(index.keys, h, side="left"))
                    hi = int(np.searchsorted(index.keys, h, side="right"))
                    if hi - lo > index.config.max_postings_per_key:
                        continue
                    for rid in index.rows[lo:hi]:
                        candidate_id = bytes(index.ids[int(rid)]).decode("ascii").rstrip("\x00")
                        if candidate_id in actual:
                            pass_found.add(candidate_id)
                raw_pass_hits[pass_name] += len(pass_found)
        if len(misses) < missed_examples:
            misses.extend((source_id, target_id) for target_id in sorted(actual - found)
                          if len(misses) < missed_examples)
    counts_arr = np.asarray(counts, dtype=np.int32)
    return {
        "rows": rows,
        "true_links": total_links,
        "retained_links": kept_links,
        "candidate_recall": kept_links / total_links if total_links else None,
        "raw_pass_true_link_hits": dict(raw_pass_hits) if measure_raw_passes else None,
        "raw_pass_recall": ({p: raw_pass_hits[p] / total_links if total_links else None
                             for p in PASS_WEIGHTS} if measure_raw_passes else None),
        "capped_pass_true_link_hits": dict(pass_hits),
        "mean_candidates": float(counts_arr.mean()) if rows else 0,
        "p50_candidates": float(np.quantile(counts_arr, .5)) if rows else 0,
        "p95_candidates": float(np.quantile(counts_arr, .95)) if rows else 0,
        "p99_candidates": float(np.quantile(counts_arr, .99)) if rows else 0,
        "empty_candidate_rows": int(zero_candidates),
        "reduction_ratio": float(1 - counts_arr.sum() / (rows * index.target_count)) if rows else None,
        "missed_examples": misses[:missed_examples],
    }
