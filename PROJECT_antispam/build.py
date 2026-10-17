"""Воспроизводимая сборка данных и обучение модели: python build.py.

Конвейер сохраняет Pipeline и выбранный порог для HTTP-сервиса.
Docker собирается отдельно с помощью start_docker.ps1.
"""

import argparse
import json
from pathlib import Path

from src.constants import DATA_ROOT, PROJECT_ROOT
from src.data import prepare_data
from src.modeling import train_experiment
from src.artifacts import save_artifact


def build(data_root=DATA_ROOT, external_root=None, output_root=PROJECT_ROOT, export=True):
    output_root = Path(output_root)
    prepared = prepare_data(
        data_root=data_root, external_root=external_root,
        output_dir=output_root / "data/processed",
    )
    print(json.dumps(prepared.summary, ensure_ascii=False, indent=2), flush=True)
    result = train_experiment(prepared, report_dir=output_root / "reports")
    if export:
        save_artifact(result["model"], result["summary"],
                      output_root / "microservice/app/model", train=prepared.train)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    return prepared, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--external-root", type=Path, default=PROJECT_ROOT / "data/external/raw")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--no-export", action="store_true", help="Не сохранять модель сервиса")
    args = parser.parse_args()
    build(args.data_root, args.external_root, args.output_root, export=not args.no_export)


if __name__ == "__main__":
    main()
