"""Проверить работающий сервис и сравнить HTTP-предсказания с отчётами."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_api(base_url="http://127.0.0.1:8000"):
    base_url = base_url.rstrip("/")
    session = requests.Session()
    session.trust_env = False
    health = session.get(base_url + "/health", timeout=15)
    health.raise_for_status()
    info = session.get(base_url + "/model", timeout=15)
    info.raise_for_status()
    metadata = json.loads((ROOT / "microservice/app/model/metadata.json").read_text(encoding="utf-8"))
    assert info.json()["model_sha256"] == metadata["model_sha256"]
    assert info.json()["threshold"] == metadata["threshold"]
    checks = []
    for split, report_name in [("test", "test"), ("external", "external_unique")]:
        frame = pd.read_csv(ROOT / f"data/processed/{split}.csv.gz", keep_default_na=False)
        reference = pd.read_csv(ROOT / f"reports/predictions_{report_name}.csv")
        # Проверяем ham, найденный spam и пропущенный spam, если они есть.
        chosen = []
        for label, prediction in [(0, 0), (1, 1), (1, 0)]:
            subset = reference.loc[reference.label.eq(label) & reference.prediction.eq(prediction)]
            if len(subset):
                chosen.append(subset.index[0])
        for index in chosen:
            response = session.post(base_url + "/predict", json={"text": frame.loc[index, "text"]}, timeout=20)
            response.raise_for_status()
            actual = response.json()
            assert abs(actual["score"] - reference.loc[index, "score"]) < 1e-10
            assert int(actual["is_spam"]) == reference.loc[index, "prediction"]
            checks.append({"dataset": split, "path": frame.loc[index, "path"],
                           "score_matches": True, "prediction_matches": True})
    example = ROOT / "examples/ham.eml"
    with example.open("rb") as stream:
        eml = session.post(base_url + "/predict/eml", files={"file": (example.name, stream, "message/rfc822")}, timeout=20)
    eml.raise_for_status()
    assert session.post(base_url + "/predict", json={"text": " "}, timeout=10).status_code == 422
    result = {"base_url": base_url, "health": health.json(), "model": info.json(),
              "prediction_checks": checks, "eml_upload": eml.json(), "blank_text_rejected": True}
    (ROOT / "reports/docker_api_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    check_api(parser.parse_args().url)
