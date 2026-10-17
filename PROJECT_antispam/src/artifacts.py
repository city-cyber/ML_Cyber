"""Сохранение и проверка доверенного артефакта модели для сервиса."""

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from .preprocessing import PREPROCESSING_VERSION

SCHEMA_VERSION = 1
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "microservice/app/model"
MODEL_PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib")


def runtime_versions():
    return {name: version(name) for name in MODEL_PACKAGES}


def training_fingerprint(frame):
    digest = hashlib.sha256()
    for text, label in zip(frame.text, frame.label):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes([int(label)]))
    return digest.hexdigest()


def save_artifact(model, summary, model_dir=DEFAULT_MODEL_DIR, train=None):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(summary["threshold"]["threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Порог должен быть конечным числом")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_name": summary["selected_model"],
        "threshold": threshold, "score_kind": summary["score_kind"],
        "threshold_selection": summary["threshold"]["selection_status"],
        "python": platform.python_version(), "packages": runtime_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train) if train is not None else None,
        "training_sha256": training_fingerprint(train) if train is not None else None,
        "evaluation_metrics": summary.get("evaluations", {}),
    }
    bundle = {"schema_version": SCHEMA_VERSION, "pipeline": model,
              "threshold": threshold, "model_name": metadata["model_name"],
              "score_kind": metadata["score_kind"]}
    model_file = model_dir / MODEL_FILENAME
    temporary = model_dir / (MODEL_FILENAME + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(model_file)
    metadata["model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
    metadata_file = model_dir / METADATA_FILENAME
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def load_artifact(model_dir=DEFAULT_MODEL_DIR):
    """Joblib загружается только из локального каталога модели, никогда из HTTP."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Несовместимый формат артефакта")
    if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Версия предобработки отличается от обучения")
    for name, actual in runtime_versions().items():
        expected = metadata["packages"].get(name)
        if actual != expected:
            raise ValueError(f"Несовместимая версия {name}: {actual}, требуется {expected}")
    model_file = model_dir / MODEL_FILENAME
    if hashlib.sha256(model_file.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Контрольная сумма модели не совпадает")
    bundle = joblib.load(model_file)
    for key in ["schema_version", "threshold", "model_name", "score_kind"]:
        if bundle.get(key) != metadata.get(key):
            raise ValueError(f"Метаданные модели не согласованы: {key}")
    if not math.isfinite(float(bundle["threshold"])):
        raise ValueError("Некорректный порог модели")
    if list(bundle["pipeline"].classes_) != [0, 1]:
        raise ValueError("Модель должна различать классы 0=ham и 1=spam")
    return bundle, metadata
