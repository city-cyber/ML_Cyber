"""Таблицы и визуализации EDA для основного ноутбука."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer

from .data import FEATURES

PALETTE = {"ham": "#159895", "spam": "#e76f51"}


def configure_plots():
    sns.set_theme(style="whitegrid", palette="Set2", context="notebook")
    plt.rcParams.update({"figure.figsize": (11, 5), "axes.titlesize": 14, "figure.dpi": 110})


def finish_figure(fig, output_dir, name):
    fig.tight_layout()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def source_table(frame):
    return frame.groupby(["source", "class_name"]).agg(
        emails=("text", "size"), median_words=("n_words", "median"),
        html_share=("html", "mean"), empty_subjects=("subject", lambda s: s.eq("").sum()),
        parser_defects=("parse_defects", "sum"),
    ).round(3)


def plot_balance(frame, output_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=frame, x="source", hue="class_name", palette=PALETTE, ax=axes[0])
    axes[0].set(title="Число писем по источникам", xlabel="Источник", ylabel="Письма")
    counts = frame.class_name.value_counts().reindex(["ham", "spam"])
    axes[1].pie(counts, labels=counts.index, autopct="%.1f%%", colors=[PALETTE[k] for k in counts.index])
    axes[1].set_title("Доли целевых классов")
    finish_figure(fig, output_dir, "01_class_balance")


def plot_lengths(frame, output_dir=None):
    data = frame.assign(log_words=np.log1p(frame.n_words), log_chars=np.log1p(frame.n_chars))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(data=data, x="log_words", hue="class_name", bins=40, element="step", palette=PALETTE, ax=axes[0])
    axes[0].set(title="Распределение длины текста", xlabel="log(1 + число слов)", ylabel="Письма")
    sns.boxplot(data=data, x="source", y="log_chars", hue="source", legend=False, ax=axes[1])
    axes[1].set(title="Длина писем и выбросы по источникам", xlabel="Источник", ylabel="log(1 + число символов)")
    finish_figure(fig, output_dir, "02_lengths")


def plot_structure(frame, output_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rates = frame.groupby("source")[["html", "attachments"]].agg({"html": "mean", "attachments": lambda s: s.gt(0).mean()})
    rates.rename(columns={"html": "Есть HTML", "attachments": "Есть вложения"}).plot.bar(ax=axes[0])
    axes[0].set(title="Структура MIME", xlabel="Источник", ylabel="Доля писем", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=0)
    for ax, column, title in [
        (axes[1], "n_urls", "Ссылки в письмах"),
        (axes[2], "n_exclamations", "Восклицательные знаки"),
    ]:
        sns.boxplot(data=frame.assign(value=np.log1p(frame[column])), x="source", y="value", hue="source", legend=False, ax=ax)
        ax.set(title=title, xlabel="Источник", ylabel="log(1 + количество)")
    finish_figure(fig, output_dir, "03_mime_and_links")


def plot_correlations(frame, output_dir=None):
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = frame[FEATURES + ["label"]].astype(float).corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Ранговые корреляции Спирмена: признаки и метка")
    finish_figure(fig, output_dir, "04_correlations")
    return corr


def plot_pairs(frame, output_dir=None):
    # Равное число примеров каждого класса для читаемости графика.
    sample = pd.concat([part.sample(min(len(part), 180), random_state=42) for _, part in frame.groupby("label")])
    sample = sample.assign(log_words=np.log1p(sample.n_words), log_urls=np.log1p(sample.n_urls))
    grid = sns.pairplot(
        sample, vars=["log_words", "log_urls", "uppercase_ratio"],
        hue="class_name", palette=PALETTE, corner=True, diag_kind="hist",
        plot_kws={"alpha": 0.35, "s": 16},
    )
    grid.figure.suptitle("Парные распределения признаков (сбалансированная подвыборка)", y=1.02)
    finish_figure(grid.figure, output_dir, "05_pairplot")


def plot_words(frame, output_dir=None):
    vectorizer = CountVectorizer(stop_words="english", min_df=3, max_features=20_000, binary=True)
    matrix = vectorizer.fit_transform(frame.text)
    terms = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    results = []
    for label, ax in zip([0, 1], axes):
        mask = frame.label.to_numpy() == label
        share = np.asarray(matrix[mask].mean(axis=0)).ravel()
        other = np.asarray(matrix[~mask].mean(axis=0)).ravel()
        # Разница долей документов, а не частоты повторения слова в одном письме.
        top = np.argsort(share - other)[-15:]
        ax.barh(terms[top], (share - other)[top], color=PALETTE["spam" if label else "ham"])
        ax.set(title="Характерные слова: " + ("spam" if label else "ham"), xlabel="Разница долей писем с этим словом")
        results.extend({"class": "spam" if label else "ham", "term": terms[i], "document_share_difference": float(share[i] - other[i])} for i in top)
    finish_figure(fig, output_dir, "06_characteristic_words")
    return pd.DataFrame(results)


def statistical_checks(frame):
    rows = []
    for feature in ["n_words", "n_urls", "uppercase_ratio"]:
        ham = frame.loc[frame.label.eq(0), feature]
        spam = frame.loc[frame.label.eq(1), feature]
        test = mannwhitneyu(ham, spam, alternative="two-sided", method="asymptotic")
        rows.append({"feature": feature, "test": "Mann–Whitney U", "p_value": test.pvalue,
                     "rank_biserial_ham_minus_spam": 2 * test.statistic / (len(ham) * len(spam)) - 1,
                     "ham_median": ham.median(), "spam_median": spam.median()})
    table = pd.crosstab(frame.label, frame.html).reindex(index=[0, 1], columns=[False, True], fill_value=0)
    if (table.sum(axis=0) > 0).all():
        test = chi2_contingency(table, correction=False)
        rows.append({"feature": "html", "test": "Chi-square", "p_value": test.pvalue,
                     "cramers_v": np.sqrt(test.statistic / table.to_numpy().sum())})
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    result["p_holm"] = np.minimum(1, np.maximum.accumulate(result.p_value * np.arange(len(result), 0, -1)))
    result["significant_005"] = result.p_holm < 0.05
    return result
