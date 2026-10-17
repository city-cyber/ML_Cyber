"""Сравнение моделей, выбор порога и независимая оценка."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import beta
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, fbeta_score,
    make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .constants import CV_FOLDS, MAX_FPR, RANDOM_STATE, TARGET_PRECISION
from .inference import default_threshold, spam_scores


def false_positive_rate(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / max(tn + fp, 1)


def make_candidates():
    """TF-IDF обучается заново внутри каждого фолда Pipeline."""
    def pipeline(classifier):
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                min_df=2, max_df=0.995, max_features=30_000,
                sublinear_tf=True, dtype=np.float64,
            )),
            ("classifier", classifier),
        ])

    return {
        "Dummy: always ham": pipeline(DummyClassifier(strategy="constant", constant=0)),
        "ComplementNB alpha=0.5": pipeline(ComplementNB(alpha=0.5)),
        "ComplementNB alpha=1.0": pipeline(ComplementNB(alpha=1.0)),
        "LogisticRegression C=0.5": pipeline(LogisticRegression(
            C=0.5, class_weight={0: 2, 1: 1}, solver="liblinear",
            max_iter=1500, random_state=RANDOM_STATE,
        )),
        "LogisticRegression C=2.0": pipeline(LogisticRegression(
            C=2.0, class_weight={0: 2, 1: 1}, solver="liblinear",
            max_iter=1500, random_state=RANDOM_STATE,
        )),
        "LinearSVC C=1.0": pipeline(LinearSVC(
            C=1.0, class_weight={0: 2, 1: 1}, random_state=RANDOM_STATE,
            dual="auto", max_iter=5000,
        )),
    }


def compare_models(train, folds=CV_FOLDS):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    # Балансируем не только классы, но и редкий hard_ham.
    splits = list(cv.split(train.text, train.source))
    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "f05": make_scorer(fbeta_score, beta=0.5, zero_division=0),
        "fpr": make_scorer(false_positive_rate),
    }
    rows, details = [], []
    for name, candidate in make_candidates().items():
        print(f"CV ({folds} folds): {name}", flush=True)
        result = cross_validate(
            candidate, train.text, train.label, cv=splits, scoring=scoring,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        row = {"model": name, "fit_seconds_mean": float(result["fit_time"].mean())}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
        for fold in range(folds):
            details.append({"model": name, "fold": fold + 1, **{
                metric: float(result[f"test_{metric}"][fold]) for metric in scoring
            }})
    table = pd.DataFrame(rows)
    table["cv_constraints_met"] = (
        table.precision_mean.ge(TARGET_PRECISION) & table.fpr_mean.le(MAX_FPR)
        & table.recall_mean.gt(0)
    )
    # Если ограничения на CV недостижимы, фиксируем это, а не скрываем результат.
    table = table.sort_values(
        ["cv_constraints_met", "f05_mean", "fpr_mean", "recall_mean"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    winner = table.iloc[0].model
    return table, pd.DataFrame(details), winner


def binary_metrics(y_true, predictions, scores=None):
    y = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, predictions, labels=[0, 1]).ravel()]
    ham_count = tn + fp
    # Односторонняя точная верхняя 95%-граница Clopper–Pearson.
    upper = float(beta.ppf(0.95, fp + 1, tn)) if tn else 1.0
    metrics = {
        "n": len(y), "ham": ham_count, "spam": tp + fn,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "f05": float(fbeta_score(y, predictions, beta=0.5, zero_division=0)),
        "fpr": fp / max(ham_count, 1),
        "fpr_upper_95": upper if ham_count else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if scores is not None and len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
        metrics["average_precision"] = float(average_precision_score(y, scores))
    return metrics


def tune_threshold(y_true, scores, target_precision=TARGET_PRECISION, max_fpr=MAX_FPR):
    """Выбираем минимальное FP, затем максимальный Recall на validation.

    Порог не подбирается на обучающих предсказаниях или тесте.
    Отсутствие допустимого нетривиального порога отмечается явно.
    """
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Модель вернула неконечные оценки")
    unique = np.unique(scores)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        unique[:-1] + np.diff(unique) / 2,
        np.nextafter(unique[-1], np.inf),
    ]
    y = np.asarray(y_true)
    rows = []
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        tn = int(((y == 0) & ~predicted).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        rows.append({
            "threshold": float(threshold), "precision": precision, "recall": recall,
            "fpr": fpr, "fp": fp, "tp": tp,
            "f05": 1.25 * precision * recall / max(0.25 * precision + recall, 1e-15),
            "constraints_met": precision >= target_precision and fpr <= max_fpr and tp > 0,
        })
    table = pd.DataFrame(rows)
    feasible = table.loc[table.constraints_met]
    if feasible.empty:
        # Оставляем все письма в ham и явно сообщаем о невыполненных ограничениях.
        chosen = table.iloc[-1].to_dict()
        chosen["selection_status"] = "no_feasible_threshold_all_ham_fallback"
    else:
        chosen = feasible.sort_values(
            ["fp", "recall", "f05", "threshold"], ascending=[True, False, False, True]
        ).iloc[0].to_dict()
        chosen["selection_status"] = "constraints_met_on_validation"
    return chosen, table


def evaluate(model, frame, threshold):
    scores = spam_scores(model, frame.text)
    predictions = (scores >= threshold).astype(int)
    metrics = binary_metrics(frame.label, predictions, scores)
    columns = ["path", "source", "label", "subject", "text_hash"]
    result = frame[columns].copy()
    result["score"] = scores
    result["prediction"] = predictions
    result["error_type"] = np.select(
        [(frame.label.eq(0) & (predictions == 1)), (frame.label.eq(1) & (predictions == 0))],
        ["false_positive", "false_negative"], default="correct",
    )
    per_source = []
    for source, subset in result.groupby("source"):
        record = binary_metrics(subset.label, subset.prediction)
        record["source"] = source
        per_source.append(record)
    return metrics, result, pd.DataFrame(per_source)


def train_experiment(prepared, report_dir=None):
    cv_table, cv_folds, winner = compare_models(prepared.train)
    model = clone(make_candidates()[winner]).fit(prepared.train.text, prepared.train.label)
    validation_scores = spam_scores(model, prepared.validation.text)
    threshold_info, threshold_table = tune_threshold(prepared.validation.label, validation_scores)
    threshold = float(threshold_info["threshold"])
    # После подбора порога веса и словарь не переобучаются на validation/test.
    evaluations, predictions, by_source = {}, {}, {}
    for name, frame in [
        ("validation", prepared.validation), ("test", prepared.test),
        ("external_unique", prepared.external), ("external_all", prepared.external_raw),
    ]:
        metrics, prediction, sources = evaluate(model, frame, threshold)
        default_scores = spam_scores(model, frame.text)
        default_prediction = (default_scores >= default_threshold(model)).astype(int)
        metrics["default_threshold_metrics"] = binary_metrics(frame.label, default_prediction, default_scores)
        evaluations[name] = metrics
        predictions[name] = prediction
        by_source[name] = sources
    summary = {
        "selected_model": winner, "cv_folds": CV_FOLDS,
        "threshold": threshold_info, "default_threshold": default_threshold(model),
        "score_kind": "probability_estimate" if hasattr(model, "predict_proba") else "decision_margin",
        "target_precision": TARGET_PRECISION, "max_fpr": MAX_FPR,
        "selection_uses": "train CV for model; validation for threshold",
        "final_fit_uses": "train only; no refit after threshold selection",
        "evaluations": evaluations,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        cv_table.to_csv(report_dir / "cv_comparison.csv", index=False)
        cv_folds.to_csv(report_dir / "cv_folds.csv", index=False)
        threshold_table.to_csv(report_dir / "threshold_search.csv", index=False)
        for name in evaluations:
            predictions[name].to_csv(report_dir / f"predictions_{name}.csv", index=False)
            by_source[name].to_csv(report_dir / f"metrics_by_source_{name}.csv", index=False)
        (report_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from .reporting import write_results
        write_results(prepared.summary, summary, report_dir.parent)
    return {"model": model, "summary": summary, "cv_table": cv_table,
            "cv_folds": cv_folds, "threshold_table": threshold_table,
            "predictions": predictions, "by_source": by_source}
