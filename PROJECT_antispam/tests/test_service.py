"""Контракт API и сохранение результатов исходного эксперимента."""

import json
import shutil
import tempfile
import unittest
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from microservice.app.main import MAX_EML_BYTES, create_app
from src.artifacts import DEFAULT_MODEL_DIR, load_artifact
from src.constants import PROJECT_ROOT
from src.inference import spam_scores
from src.preprocessing import parse_email


@unittest.skipUnless((DEFAULT_MODEL_DIR / "model.joblib").exists(), "Сначала выполните export_model.py")
class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.metadata = load_artifact()
        cls.client_context = TestClient(create_app())
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health_model_info_and_openapi(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])
        info = self.client.get("/model").json()
        self.assertEqual(info["threshold"], self.metadata["threshold"])
        self.assertEqual(info["model_sha256"], self.metadata["model_sha256"])
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertIn("/predict/eml", self.client.get("/openapi.json").json()["paths"])

    def test_json_prediction_matches_pipeline(self):
        text = "Please review our meeting agenda and project report."
        response = self.client.post("/predict", json={"text": text})
        self.assertEqual(response.status_code, 200)
        expected = float(spam_scores(self.bundle["pipeline"], [text])[0])
        self.assertAlmostEqual(response.json()["score"], expected, places=12)
        self.assertEqual(response.json()["is_spam"], expected >= self.bundle["threshold"])

    def test_html_json_and_eml_have_same_preprocessing(self):
        message = EmailMessage()
        message["Subject"] = "Встреча завтра"
        body = "<html><body>Report for Monday<script>ignored</script></body></html>"
        message.set_content(body, subtype="html", charset="utf-8")
        raw = message.as_bytes()
        json_result = self.client.post("/predict", json={
            "subject": "Встреча завтра", "body": body, "body_format": "html",
        })
        eml_result = self.client.post("/predict/eml", files={"file": ("sample.eml", raw, "message/rfc822")})
        self.assertEqual(json_result.status_code, 200)
        self.assertEqual(eml_result.status_code, 200)
        self.assertAlmostEqual(json_result.json()["score"], eml_result.json()["score"], places=12)
        direct = float(spam_scores(self.bundle["pipeline"], [parse_email(raw)["text"]])[0])
        self.assertAlmostEqual(eml_result.json()["score"], direct, places=12)

    def test_invalid_requests_are_rejected(self):
        payloads = [
            {}, {"text": "   "}, {"text": "x", "body": "y"},
            {"text": "x", "threshold": 0.1}, {"body": "x", "subject": "one\ntwo"},
            {"text": "x" * 200_001},
            {"body": "<script>hidden</script>", "body_format": "html"},
        ]
        for payload in payloads:
            with self.subTest(payload=list(payload)):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)
        self.assertEqual(self.client.post("/predict/eml", files={"file": ("empty.eml", b"")}).status_code, 422)
        self.assertEqual(self.client.post("/predict/eml", files={"file": ("big.eml", b"x" * (MAX_EML_BYTES + 1))}).status_code, 413)

    def test_saved_test_examples_match_api(self):
        test = pd.read_csv(PROJECT_ROOT / "data/processed/test.csv.gz", keep_default_na=False)
        reference = pd.read_csv(PROJECT_ROOT / "reports/predictions_test.csv")
        indices = [test.loc[test.label.eq(label)].index[0] for label in [0, 1]]
        for index in indices:
            response = self.client.post("/predict", json={"text": test.loc[index, "text"]})
            self.assertEqual(response.status_code, 200)
            self.assertAlmostEqual(response.json()["score"], reference.loc[index, "score"], places=12)
            self.assertEqual(int(response.json()["is_spam"]), reference.loc[index, "prediction"])

    def test_tampered_model_and_versions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            for name in ["model.joblib", "metadata.json"]:
                shutil.copyfile(DEFAULT_MODEL_DIR / name, target / name)
            with (target / "model.joblib").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Контрольная сумма"):
                load_artifact(target)
            metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            metadata["packages"]["scikit-learn"] = "0.0.invalid"
            (target / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scikit-learn"):
                load_artifact(target)

    def test_missing_artifact_prevents_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                with TestClient(create_app(Path(temporary))):
                    pass


if __name__ == "__main__":
    unittest.main()
