"""Восстановить выбранную модель на сохранённом train и проверить её предсказания.

Повторного выбора модели, порога и использования тестовых меток при fit нет.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import DEFAULT_MODEL_DIR, load_artifact, save_artifact
from src.constants import PROJECT_ROOT
from src.inference import spam_scores
from src.modeling import make_candidates


def export_model(project_root=PROJECT_ROOT, model_dir=DEFAULT_MODEL_DIR):
    root = Path(project_root)
    summary = json.loads((root / "reports/metrics.json").read_text(encoding="utf-8"))
    train = pd.read_csv(root / "data/processed/train.csv.gz", keep_default_na=False)
    model = make_candidates()[summary["selected_model"]].fit(train.text, train.label)
    metadata = save_artifact(model, summary, model_dir, train=train)
    bundle, _ = load_artifact(model_dir)
    checks = {}
    for split, report_name in [
        ("validation", "validation"), ("test", "test"), ("external", "external_unique"),
    ]:
        frame = pd.read_csv(root / f"data/processed/{split}.csv.gz", keep_default_na=False)
        reference = pd.read_csv(root / f"reports/predictions_{report_name}.csv")
        if frame.path.tolist() != reference.path.tolist():
            raise AssertionError(f"Изменился порядок данных {split}")
        scores = spam_scores(bundle["pipeline"], frame.text)
        if not np.allclose(scores, reference.score.to_numpy(), rtol=1e-12, atol=1e-12):
            raise AssertionError(f"Экспорт изменил оценки {split}")
        prediction = (scores >= bundle["threshold"]).astype(int)
        if not np.array_equal(prediction, reference.prediction):
            raise AssertionError(f"Экспорт изменил решения {split}")
        checks[split] = {"rows": len(frame), "scores_match": True, "predictions_match": True}
    report = {"model_dir": str(Path(model_dir).resolve()), "metadata": metadata, "checks": checks}
    (root / "reports/export_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"model": metadata["model_name"], "threshold": metadata["threshold"],
                      "sha256": metadata["model_sha256"], "checks": checks}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    export_model(args.project_root, args.model_dir)


if __name__ == "__main__":
    main()
