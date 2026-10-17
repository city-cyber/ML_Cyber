"""Проверки инвариантов, от которых зависит достоверность результата."""

import unittest
import io
import tarfile
import tempfile
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import assert_disjoint, clean_corpus, fingerprint, load_external_archive, parse_email
from src.modeling import binary_metrics, tune_threshold


class EmailParsingTests(unittest.TestCase):
    def test_mime_subject_and_plain_preferred_over_html(self):
        message = EmailMessage()
        message["Subject"] = "Встреча завтра"
        message["X-Spam-Status"] = "Yes SECRET_CLASS_MARKER"
        message.set_content("План встречи")
        message.add_alternative("<html><script>secret_script</script>Другой вариант</html>", subtype="html")
        row = parse_email(message.as_bytes())
        self.assertIn("Встреча завтра", row["text"])
        self.assertIn("План встречи", row["text"])
        self.assertNotIn("Другой вариант", row["text"])
        self.assertNotIn("SECRET_CLASS_MARKER", row["text"])
        self.assertTrue(row["html"])

    def test_html_and_attachments(self):
        message = EmailMessage()
        message["Subject"] = "Report"
        message.set_content("<html><head><title>hidden</title></head><body>Visible<script>secret</script></body></html>", subtype="html")
        message.add_attachment(b"attachment_secret", maintype="application", subtype="octet-stream", filename="data.bin")
        row = parse_email(message.as_bytes())
        self.assertIn("Visible", row["text"])
        self.assertNotIn("secret", row["text"])
        self.assertNotIn("hidden", row["text"])
        self.assertEqual(row["attachments"], 1)

    def test_normalized_fingerprint(self):
        self.assertEqual(fingerprint("Hello\n WORLD"), fingerprint("hello world"))

    def test_external_archive_labels_and_mime(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "test.tar.xz"
            with tarfile.open(archive, "w:xz") as collection:
                for name in ["ham/000.eml", "spam/000.eml"]:
                    payload = b"Subject: Test\n\nExample mail body\n"
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    collection.addfile(member, io.BytesIO(payload))
            data, audit = load_external_archive(archive)
            self.assertEqual(data.label.tolist(), [0, 1])
            self.assertEqual(data.class_name.tolist(), ["ham", "spam"])
            self.assertTrue(data.text.str.contains("Example mail body").all())
            self.assertEqual(len(audit), 0)


class LeakageAndThresholdTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            {"path": "a", "source": "ham", "label": 0, "text": "A", "raw_hash": "a", "text_hash": "same", "body_hash": "body", "message_id": "a"},
            {"path": "b", "source": "ham", "label": 0, "text": "A", "raw_hash": "b", "text_hash": "same", "body_hash": "body", "message_id": "b"},
            {"path": "c", "source": "spam", "label": 1, "text": "C", "raw_hash": "c", "text_hash": "other", "body_hash": "other", "message_id": "c"},
        ])

    def test_dedup_and_conflicting_labels_quarantined(self):
        data = self.sample()
        clean, audit = clean_corpus(data)
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(audit), 1)
        data.loc[1, "label"] = 1
        clean, audit = clean_corpus(data)
        self.assertEqual(clean.path.tolist(), ["c"])
        self.assertTrue(audit.reason.str.startswith("conflicting_").all())

    def test_cross_split_overlap_fails(self):
        data = self.sample()
        with self.assertRaises(AssertionError):
            assert_disjoint(data.iloc[:1], data.iloc[1:2])

    def test_threshold_minimizes_fp_then_maximizes_recall(self):
        y = [0, 0, 1, 1, 1]
        scores = [0.1, 0.6, 0.5, 0.8, 0.9]
        chosen, _ = tune_threshold(y, scores)
        self.assertEqual(chosen["fp"], 0)
        self.assertAlmostEqual(chosen["recall"], 2 / 3)
        self.assertTrue(chosen["constraints_met"])

    def test_no_feasible_threshold_is_explicit(self):
        chosen, _ = tune_threshold([0, 1], [0.9, 0.1])
        self.assertFalse(chosen["constraints_met"])
        self.assertIn("fallback", chosen["selection_status"])
        self.assertGreater(chosen["threshold"], 0.9)

    def test_zero_observed_fp_has_positive_uncertainty(self):
        metrics = binary_metrics([0] * 100 + [1], [0] * 100 + [1])
        self.assertEqual(metrics["fpr"], 0)
        self.assertGreater(metrics["fpr_upper_95"], 0)
        self.assertTrue(np.isclose(metrics["fpr_upper_95"], 1 - 0.05 ** (1 / 100)))


if __name__ == "__main__":
    unittest.main()
