"""Краткие отчёты из фактических, а не вручную придуманных метрик."""

from pathlib import Path

import pandas as pd


def write_results(data_summary, summary, root):
    root = Path(root)
    docs = root / "docs"
    reports = root / "reports"
    docs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, metrics in summary["evaluations"].items():
        for policy, values in [("standard", metrics["default_threshold_metrics"]), ("selected", metrics)]:
            rows.append({"dataset": dataset, "policy": policy, **{
                key: values[key] for key in ["n", "precision", "recall", "f1", "f05", "fpr", "fp", "fn", "fpr_upper_95"]
            }})
    pd.DataFrame(rows).to_csv(reports / "final_comparison.csv", index=False)
    lines = [
        "# Фактические результаты эксперимента", "",
        f"Модель: **{summary['selected_model']}**. TF-IDF слов и биграмм, до 30 000 признаков.",
        f"Порог: **{summary['threshold']['threshold']:.6f}**. Выбран на validation с приоритетом минимума FP.",
        f"Сравнение шести конфигураций: {summary['cv_folds']}-fold CV только внутри train.", "",
        f"Исходных писем: {data_summary['raw_emails']}. Подготовлено: {data_summary['prepared_emails']}.",
        f"Удалено пересечений с контролем: **{data_summary['removed_external_overlaps']}**.",
        f"Train / validation / test: {data_summary['train']} / {data_summary['validation']} / {data_summary['test']}.",
        f"Контроль полный / очищенный: {data_summary['external_raw']} / {data_summary['external_unique']}.", "",
        "| Выборка | Порог | Precision | Recall | F1 | FP | FN | FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['policy']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['fp']} | {row['fn']} | {row['fpr']:.4%} |"
        )
    lines.extend([
        "", "Приоритет минимизации FP снижает Recall. Стандартный порог показан для сравнения;",
        "он не выбирался по тестовым результатам. Модель и выбранный порог после просмотра теста не менялись.",
        "", "## Неопределённость и ограничения", "",
    ])
    for dataset in ["test", "external_unique"]:
        metrics = summary["evaluations"][dataset]
        lines.append(
            f"- {dataset}: {metrics['fp']} FP из {metrics['ham']} ham; "
            f"верхняя односторонняя 95%-граница FPR — {metrics['fpr_upper_95']:.2%}."
        )
    lines.extend([
        "", "Нулевые наблюдаемые FP не означают нулевого риска. Корпуса исторические и связаны по происхождению.",
        "Мультиязычность и перенос на современную почту не проверены; точная дедупликация не устраняет все шаблоны.",
        "Для следующего выбора параметров нужен новый контроль, поскольку текущий уже просмотрен.",
        "", "## Состояние этапов", "",
        "1. Подготовка данных выполнена; обработанные наборы и полный аудит сохранены.",
        "2. Модель обучена; код, CV, метрики, предсказания и ноутбук подготовлены.",
        "3. Упаковка: сохранённый Pipeline и порог, FastAPI, Dockerfile, Compose и скрипты запуска. "
        "Проверки упаковки см. в `reports/export_verification.json`, `reports/docker_api_verification.json` и `docs/DOCKER.md`.",
        "4. Публичное размещение не выполнялось.",
        "5–6. Материалы доступны для командной презентации; оформление и отправка ссылки остаются за командой.",
        "", "Этот файл автоматически формируется из `data_summary.json` и результатов обучения.",
    ])
    (docs / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)
