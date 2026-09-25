import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normalization import (  # noqa: E402
    NORMALIZATION_VERSION,
    char_ngrams,
    normalize_address,
    normalize_country,
    normalize_name,
    normalize_record,
)


class NormalizationTests(unittest.TestCase):
    def test_name_views_keep_legal_suffix_evidence(self):
        value = normalize_name("  B+ Retail, Inc.  ")
        self.assertEqual(value.norm, "b plus retail inc")
        self.assertEqual(value.core, "b plus retail")
        self.assertEqual(value.sorted_tokens, "b inc plus retail")
        self.assertEqual(value.compact, "bplusretailinc")

    def test_suffix_is_only_removed_from_end(self):
        self.assertEqual(normalize_name("Limited Edition LLC").core, "limited edition")
        self.assertEqual(normalize_name("Private Eye").core, "private eye")
        self.assertEqual(normalize_name("Red Ventures Private Limited").core, "red ventures")

    def test_unicode_preserved_and_width_normalized(self):
        self.assertEqual(normalize_name("Ｒｅｄ ＆ वेंचर्स").norm, "red and वेंचर्स")
        self.assertEqual(normalize_name("CAFÉ").norm, "café")
        self.assertNotEqual(normalize_name("रेड वेंचर्स").norm, normalize_name("red ventures").norm)

    def test_apostrophe_and_whitespace(self):
        self.assertEqual(normalize_name("O’Reilly\u00a0\tFoods").norm, "oreilly foods")

    def test_address_tokens_and_postal_candidates(self):
        value = normalize_address("1795 Westchester Dr., High Point, NC 27265")
        self.assertEqual(value.norm, "1795 westchester dr high point nc 27265")
        self.assertEqual(value.numeric_tokens, ("1795", "27265"))
        self.assertEqual(value.postal_tokens, ("27265",))
        self.assertEqual(normalize_address("12 Rd, Apt 3").norm, "12 road apartment 3")

    def test_ambiguous_st_is_not_expanded(self):
        self.assertEqual(normalize_address("St Martin St").norm, "st martin st")

    def test_missing_does_not_create_present_key(self):
        for value in (None, "", "  ", float("nan")):
            self.assertFalse(normalize_name(value).present)
            self.assertFalse(normalize_address(value).present)
            self.assertEqual(normalize_name(value).norm, "")
        self.assertTrue(normalize_name("nan").present)

    def test_open_country_and_record(self):
        record = normalize_record("Chez Léo", "10 Rue de Paris", "  France ")
        self.assertEqual(record.country, "france")
        self.assertEqual(record.name.norm, "chez léo")
        self.assertEqual(normalize_country("नेपाल"), "नेपाल")

    def test_char_ngrams(self):
        self.assertEqual(char_ngrams("café", 3), ("caf", "afé"))
        self.assertEqual(char_ngrams("ab", 3), ("ab",))
        self.assertEqual(char_ngrams(""), ())
        with self.assertRaises(ValueError):
            char_ngrams("abc", 0)

    def test_version_declared(self):
        self.assertEqual(NORMALIZATION_VERSION, "1.0.0")


if __name__ == "__main__":
    unittest.main()
