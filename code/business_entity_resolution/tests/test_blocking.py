import csv
import tempfile
import unittest
from pathlib import Path

from src.blocking import BlockingConfig, build_index, evaluate_recall, open_index


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


class BlockingContractTest(unittest.TestCase):
    def test_candidates_are_targets_and_retain_known_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "s2.tsv").write_text(HEADER +
                "S2-1\tAcme Inc\t100 Main Road, Delhi\tIndia\n" +
                "S2-2\tOther Shop\t200 River Road\tIndia\n")
            (root / "s3.tsv").write_text(HEADER +
                "S3-1\tAcme\t100 Main Rd, Delhi\tIndia\n")
            (root / "s1.tsv").write_text(HEADER +
                "S1-1\tAcme Ltd\t100 Main Road Delhi\tIndia\n" +
                "S1-2\tUnseen Name\t\tFrance\n")
            (root / "truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\n" +
                "S1-1\tS2-1,S3-1\nS1-2\t\n")
            build_index(root / "s2.tsv", root / "s3.tsv", root / "idx",
                        BlockingConfig(max_candidates=10))
            with open_index(root / "idx") as index:
                outputs = dict(index.iter_candidates(root / "s1.tsv"))
                self.assertEqual(set(outputs), {"S1-1", "S1-2"})
                self.assertEqual({c.entity_id for c in outputs["S1-1"]},
                                 {"S2-1", "S3-1"})
                self.assertEqual(outputs["S1-2"], [])
                self.assertEqual(index.get_record(outputs["S1-1"][0].row_id)[0],
                                 outputs["S1-1"][0].entity_id)
                self.assertEqual(evaluate_recall(index, root / "s1.tsv",
                                                 root / "truth.tsv")["candidate_recall"], 1.0)
                for candidates in outputs.values():
                    ids = [c.entity_id for c in candidates]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertTrue(all(x.startswith(("S2-", "S3-")) for x in ids))


if __name__ == "__main__":
    unittest.main()
