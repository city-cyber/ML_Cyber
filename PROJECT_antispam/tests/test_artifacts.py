"""Проверки фактических подготовленных выборок и отчётов."""

import json
import unittest

import pandas as pd

from src.constants import PROJECT_ROOT
from src.data import assert_disjoint


@unittest.skipUnless((PROJECT_ROOT / "reports/metrics.json").exists(), "Сначала выполните build.py")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts = {
            name: pd.read_csv(PROJECT_ROOT / f"data/processed/{name}.csv.gz", keep_default_na=False)
            for name in ["train", "validation", "test", "external"]
        }
        cls.metrics = json.loads((PROJECT_ROOT / "reports/metrics.json").read_text(encoding="utf-8"))

    def test_all_prepared_splits_are_disjoint(self):
        parts = list(self.parts.values())
        for index, left in enumerate(parts):
            for right in parts[index + 1:]:
                assert_disjoint(left, right)

    def test_all_sources_are_represented_in_each_internal_split(self):
        for name in ["train", "validation", "test"]:
            self.assertEqual(set(self.parts[name].source), {"easy_ham", "hard_ham", "spam_2"})
            self.assertEqual(set(self.parts[name].label), {0, 1})

    def test_saved_predictions_reproduce_reported_counts(self):
        for name, metrics in self.metrics["evaluations"].items():
            predictions = pd.read_csv(PROJECT_ROOT / f"reports/predictions_{name}.csv")
            self.assertEqual(len(predictions), metrics["n"])
            fp = ((predictions.label == 0) & (predictions.prediction == 1)).sum()
            fn = ((predictions.label == 1) & (predictions.prediction == 0)).sum()
            self.assertEqual(fp, metrics["fp"])
            self.assertEqual(fn, metrics["fn"])
            self.assertTrue((predictions.score.ge(self.metrics["threshold"]["threshold"]).astype(int) == predictions.prediction).all())

    def test_full_control_has_all_1600_messages(self):
        control = self.metrics["evaluations"]["external_all"]
        self.assertEqual(control["n"], 1600)
        self.assertEqual(control["ham"], 600)
        self.assertEqual(control["spam"], 1000)


if __name__ == "__main__":
    unittest.main()
