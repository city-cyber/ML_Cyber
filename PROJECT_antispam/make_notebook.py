"""Создать исходный учебный ноутбук; результаты получает run_notebook.py."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n", "execution_count": None, "outputs": []})


md("""
# Итоговое задание. Антиспам-классификатор писем

**Дисциплина:** применение машинного обучения в кибербезопасности.  
**Данные:** SpamAssassin `easy_ham`, `hard_ham`, `spam_2` и отдельный контрольный архив задания.  
**Выполняемые этапы:** 1 — сбор и обработка данных; 2 — обучение и проверка ML-модели;
3 — упаковка модели в FastAPI и Docker.

Ноутбук следует структуре образца `PROJECT_final`: загрузка → предобработка → EDA с графиками,
статистическими проверками и выводами → сравнение моделей → результаты и выводы.
Функции вынесены в `src`, константы — в `src/constants.py`; `build.py` воспроизводит ML-конвейер.

**Автор и группа:** заполнить перед сдачей. Модель упакована в сервис; публичное размещение отложено.

## Оглавление

1. [Постановка задачи и план](#problem)
2. [Загрузка и обработка данных](#loading)
3. [Исследовательский анализ данных (EDA)](#eda)
4. [Обучение и кросс-валидация](#modeling)
5. [Контрольный тест и анализ ошибок](#evaluation)
6. [Результаты, ограничения и воспроизводимость](#results)
7. [Упаковка модели: FastAPI и Docker](#packaging)
""")
md("""
<a id="problem"></a>
# 0. Постановка задачи и план

**Бизнес-задача:** уменьшить нежелательную почту, сохранив доступ к обычным письмам.
Ошибочная блокировка полезного письма (FP) приоритетнее пропущенного спама (FN).

**ML-задача:** бинарная классификация текста письма: `ham = 0`, `spam = 1`.
На вход поступает письмо RFC 822/MIME; модель использует декодированные тему и тело.
Категории спама не определяются: меток «реклама», «работа» и т. п. в этих данных нет.

**Метрики:** Precision, Recall, F1, F0.5, FPR, число FP и FN; дополнительно PR-AUC (average precision) и ROC-AUC.

- $Precision=TP/(TP+FP)$: какая доля заблокированных писем действительно является спамом.
- $Recall=TP/(TP+FN)$: какая доля спама обнаружена.
- $F1=2PR/(P+R)$: баланс precision и recall.
- $F_{0.5}=1.25PR/(0.25P+R)$: больший вес precision.
- $FPR=FP/(FP+TN)$: доля обычных писем, ошибочно заблокированных.

Рабочие ограничения на валидации: **Precision ≥ 0.99, FPR ≤ 0.01, TP > 0**.
Это выбранные требования эксперимента, а не гарантии качества в реальной почте.
Среди допустимых порогов сначала минимизируем FP, затем максимизируем Recall.

Этап 3 (микросервис и Docker) реализован в разделе 6. Этап 4 (публикация) пока отложен.
Для этапов 5–6 подготовлены локальные материалы, но общая презентация и её отправка выполняются командой позднее.
""")
md("""
## 0.1 Импорт библиотек и настройка

Все пути вычисляются относительно расположения проекта. Ноутбук можно запускать из `project/`,
из корня проекта или из общей рабочей папки. Интернет при повторном выполнении не нужен,
если контрольный архив уже скачан. Зависимости указаны в `requirements.txt`.
""")
code("""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd() / 'PROJECT_antispam']
PROJECT_ROOT = next(p for p in candidates if (p / 'src/data.py').exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DATA_ROOT, RANDOM_STATE, CV_FOLDS, TARGET_PRECISION, MAX_FPR
from src.data import FEATURES, IDENTITIES, assert_disjoint, parse_email, prepare_data
from src.eda import (
    configure_plots, finish_figure, source_table, plot_balance, plot_lengths,
    plot_structure, plot_correlations, plot_pairs, plot_words, statistical_checks,
)
from src.modeling import binary_metrics, default_threshold, spam_scores, train_experiment

configure_plots()
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 90)
FIGURES = PROJECT_ROOT / 'reports/figures'
print('Проект:', PROJECT_ROOT)
print('Random state:', RANDOM_STATE, '| CV folds:', CV_FOLDS)
""")
md("""
<a id="loading"></a>
# 1. Загрузка и обработка данных

## 1.1 Источники данных

Используются три предоставленных подкорпуса **одного семейства SpamAssassin**, а не три независимых
по происхождению датасета. Это объединение нескольких наборов, но нельзя считать его полной
проверкой переноса на независимые домены. Отдельный контрольный архив также проверяем на пересечения.

| Набор | Метка | Особенности |
|---|---:|---|
| `easy_ham` | 0 | Обычная почта, относительно легко отличимая от спама |
| `hard_ham` | 0 | Обычная почта с HTML и рекламоподобными выражениями; важна для проверки FP |
| `spam_2` | 1 | Спам-письма |
| Контрольный архив | 0/1 из папок `ham`/`spam` | Предоставлен отдельно для итоговой проверки |

Источники: [официальное описание SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/readme.html),
[каталог корпусов](https://spamassassin.apache.org/old/publiccorpus/),
[контрольный набор из задания](https://disk.yandex.ru/d/mJ_9zG3dX3eZgg).

Для первичной загрузки контроля: `python -m src.download_external` из корня проекта.
SHA-256 архива проверяется; исходные локальные корпуса не изменяются.
""")
code("""
prepared = prepare_data(output_dir=PROJECT_ROOT / 'data/processed')
raw = prepared.raw
train = prepared.train
validation = prepared.validation
test = prepared.test
external = prepared.external
display(pd.Series(prepared.summary, name='Значение').to_frame())
""")
md("""
## 1.2 Знакомство с данными: размеры, типы, пропуски

Спам определяется меткой каталога, а не наличием слова «spam» в заголовке.
Просматриваем состав таблицы, технические признаки, пустые поля и дефекты парсера.
Исследуемые численные признаки нужны для EDA; в классификатор поступает только текст.
""")
code("""
display(raw[['source', 'class_name', 'subject', 'n_words', 'html']].head(8))
raw.info()
display(raw[FEATURES + ['label']].describe().T.round(3))
quality = pd.DataFrame({
    'missing': raw.isna().sum(),
    'empty_strings': [raw[c].eq('').sum() if raw[c].dtype == 'object' else 0 for c in raw.columns],
})
display(quality.loc[(quality.missing + quality.empty_strings).gt(0)])
display(source_table(raw))
""")
md("""
## 1.3 Предобработка MIME, HTML и кодировок

1. Удаляем из списка загрузки `__MACOSX`, AppleDouble `._*`, скрытые файлы и служебные индексы.
2. Разбираем MIME стандартным `email`, декодируем тему и transfer encoding тела.
3. Предпочитаем `text/plain`; при его отсутствии извлекаем видимый текст HTML без script/style/head.
   Вложения и вложенные письма не включаем в текст.
4. Нормализуем Unicode (NFKC) и пробелы. Регистр для классификации нормализует TF-IDF.
5. Исключаем пустой текст, конфликтующие метки и дубликаты. Длины и выбросы не обрезаем:
   длинное письмо может быть полезной рассылкой или важным сообщением.

Служебные заголовки `X-Spam-*`, `Received`, отправитель, имя файла и имя корпуса
не входят в признаки модели. Они способны раскрыть готовую метку или конкретный источник.
Парсер одинаков для исходных и контрольных писем. Контроль читается непосредственно из архива.
""")
code("""
audit_table = prepared.audit.groupby(['scope', 'reason']).size().rename('files').to_frame()
display(audit_table)
display(raw.groupby('source')[['parse_defects', 'replacement_chars']].agg(['sum', 'max']))
display(Markdown(
    f'**Вывод.** Загрузили {len(raw):,} исходных писем. '
    f'Контроль содержит {len(prepared.external_raw):,} читаемых писем; '
    f'после очистки внутри контроля осталось {len(external):,}. '
    'Причины каждого исключения записаны в `data/processed/data_audit.csv`.'
))
""")
md("""
## 1.4 Дубликаты и защита контрольного набора

Идентичность проверяется по SHA-256 исходных байтов, нормализованных темы и тела,
нормализованного тела и непустому `Message-ID`. У короткого тела (< 30 символов)
для проверки используется также тема, чтобы не объединить несвязанные короткие ответы.
Если одинаковая сущность имеет разные метки, такие строки исключаются из обучения и основного контроля.

**До разбиения** из обучающих корпусов исключаются все письма, пересекающиеся с полным внешним набором.
При этом используются только идентификаторы и содержимое контроля, а не его метки или метрики.
Дубликаты-шаблоны с изменёнными словами таким способом полностью не исключаются.
""")
code("""
display(Markdown(
    f'**Найдено {prepared.summary["removed_external_overlaps"]:,} пересечений** '
    'с внешним набором. Они исключены из исходных корпусов перед обучением. '
    'Оценка без этой операции могла бы быть завышена.'
))
counts = pd.concat([
    raw.source.value_counts().rename('raw'),
    prepared.clean.source.value_counts().rename('prepared'),
], axis=1).fillna(0).astype(int)
display(counts)
""")
md("""
## 1.5 Разбиение данных

Используем примерно **60% train / 20% validation / 20% test**.
Стратификация по источнику сохраняет доли обоих классов и редкого `hard_ham`.
Пять CV-фолдов строятся только внутри train, также со стратификацией по источнику.

**Train:** подробный EDA, обучение словаря, сравнение моделей на CV.  
**Validation:** выбор порога уже выбранной модели.  
**Test и внешний контроль:** финальная оценка после фиксации модели и порога.

После выбора порога модель не переобучается на train+validation: иначе шкала оценок
могла бы измениться, и выбранный порог перестал бы соответствовать обученной модели.
""")
code("""
split_tables = []
for name, frame in [('train', train), ('validation', validation), ('test', test), ('external_unique', external)]:
    part = pd.crosstab(frame.source, frame.class_name)
    part['split'] = name
    split_tables.append(part.reset_index())
display(pd.concat(split_tables).fillna(0))
for left, right in [(train, validation), (train, test), (validation, test),
                    (train, prepared.external_raw), (validation, prepared.external_raw), (test, prepared.external_raw)]:
    assert_disjoint(left, right)
print('Пересечений по всем четырём ключам идентичности нет.')
""")
md("""
<a id="eda"></a>
# 2. Исследовательский анализ данных (EDA)

Сохраняем подход образца: общий анализ → распределения → парные графики → корреляции →
статистические проверки → выводы. Здесь исследуется текст, поэтому сезонность, автокорреляция
и тест стационарности временных рядов к этой задаче не применяются.
Подробные графики и статистики далее построены **только по train**.

## 2.1 Баланс классов и состав источников
""")
code("""
display(source_table(train))
plot_balance(train, FIGURES)
ham_share = train.label.eq(0).mean()
hard_ham_share = train.source.eq('hard_ham').mean()
display(Markdown(
    f'**Вывод.** В train доля обычных писем — {ham_share:.1%}, '
    f'доля `hard_ham` — {hard_ham_share:.1%}. '
    'Общая accuracy не отражает приоритет сохранения полезной почты. '
    'Для `hard_ham` отдельно анализируем FP/FPR; небольшой размер этого набора ограничивает точность оценки.'
))
""")
md("""
## 2.2 Длина текста, пропуски и выбросы

Сравним число слов и символов. Логарифмическая шкала позволяет видеть распределение
при наличии очень длинных писем. Границы IQR считаем отдельно по каждому источнику.
Выброс по длине не означает ошибку разметки, поэтому автоматически такие письма не удаляем.
""")
code("""
plot_lengths(train, FIGURES)
length_table = train.groupby('source').n_words.quantile([0.25, 0.5, 0.75, 0.95, 0.99]).unstack()
length_table.columns = ['q25', 'median', 'q75', 'q95', 'q99']
length_table['upper_iqr'] = length_table.q75 + 1.5 * (length_table.q75 - length_table.q25)
length_table['above_iqr'] = [int((train.loc[train.source.eq(s), 'n_words'] > row.upper_iqr).sum()) for s, row in length_table.iterrows()]
display(length_table.round(1))
display(Markdown(
    '**Вывод.** Медианы числа слов по источникам: ' +
    ', '.join(f'{source}: {row["median"]:.0f}' for source, row in length_table.iterrows()) +
    '. Используем TF-IDF с sublinear_tf: повторения одного слова не должны полностью определять решение.'
))
""")
md("""
## 2.3 HTML, вложения, ссылки и стиль текста

Проверим гипотезу, что HTML и ссылки сами по себе не являются достаточными признаками спама.
Особенно важен `hard_ham`: это полезные письма, которые внешне похожи на спам.
Число URL — описательная оценка в plain-тексте и HTML, включая альтернативное представление;
оно не означает число уникальных ссылок и не используется как самостоятельное правило блокировки.
""")
code("""
plot_structure(train, FIGURES)
style_table = train.groupby('source')[['html', 'n_urls', 'n_exclamations', 'uppercase_ratio', 'digit_ratio']].mean()
display(style_table.round(3))
display(Markdown(
    '**Вывод.** Доля HTML по источникам: ' +
    ', '.join(f'{source}: {row.html:.1%}' for source, row in style_table.iterrows()) +
    '. Наличие HTML оцениваем вместе с текстовым содержанием; отдельная блокировка HTML-писем создаёт риск FP.'
))
""")
md("""
## 2.4 Парные распределения признаков

Как в `PROJECT_final`, используем `pairplot`, но для характеристик писем.
На графике одинаковое число представителей каждого класса (до 180), поэтому он не отражает исходные доли классов.
Пересечение облаков точек показывает ограничения классификации по простым численным признакам.
""")
code("""
plot_pairs(train, FIGURES)
""")
md("""
## 2.5 Корреляционный анализ

Используем корреляцию Спирмена: длины и счётчики имеют асимметричные распределения.
Корреляция с меткой — исследовательская характеристика конкретного корпуса, а не причинная связь.
Сильная корреляция числа слов и символов ожидаема: они описывают одну характеристику — длину.
""")
code("""
correlations = plot_correlations(train, FIGURES)
display(correlations['label'].drop('label').sort_values(key=abs, ascending=False).to_frame('spearman_with_label'))
""")
md("""
## 2.6 Лексика ham и spam

Для каждого класса покажем слова с наибольшей разницей долей документов, содержащих слово.
Один длинный текст с многократным повторением слова учитывается один раз.
Английские stop words удаляются только для этого графика; обучаемый TF-IDF не использует этот список.
Словарь для EDA строится только по train и не передаётся классификаторам.
""")
code("""
word_table = plot_words(train, FIGURES)
word_table.to_csv(PROJECT_ROOT / 'reports/characteristic_words.csv', index=False)
display(word_table.groupby('class').tail(5))
display(Markdown(
    '**Вывод.** Характерные слова связаны как с типом сообщения, так и с тематикой рассылок. '
    'Такие признаки помогают на историческом корпусе, но могут ухудшить перенос на современную почту. '
    'В модель не передаются названия исходных каталогов.'
))
""")
md("""
## 2.7 Проверка статистических гипотез

Нулевая гипотеза Манна–Уитни: распределения исследуемого признака в ham и spam одинаковы.
Для HTML проверяем независимость с классом χ²-тестом. Уровень значимости 0.05,
поправка Холма учитывает четыре проверки. Дополнительно показываем размеры эффектов:
rank-biserial для количественных признаков и Cramér's V для HTML.

Письма из рассылок могут быть зависимы; p-value здесь — исследовательский ориентир.
Статистическая значимость не доказывает высокое качество классификации и не заменяет тестовые метрики.
""")
code("""
statistics = statistical_checks(train)
statistics.to_csv(PROJECT_ROOT / 'reports/statistical_checks.csv', index=False)
display(statistics)
significant = statistics.loc[statistics.significant_005, 'feature'].tolist()
display(Markdown(
    '**Вывод.** После поправки Холма различия обнаружены для: ' +
    (', '.join(significant) if significant else 'ни одного из проверенных признаков') +
    '. Для модели выбираем текстовые признаки, сохраняющие больше информации, чем отдельные счётчики.'
))
""")
md("""
## 2.8 Выводы по EDA и план моделирования

- Сохраняем `hard_ham` при стратификации и отдельно проверяем его FPR.
- Не удаляем длинные письма и HTML как выбросы: это могло бы искусственно упростить задачу.
- Применяем TF-IDF со словами и биграммами, `sublinear_tf=True`, ограничением словаря 30 000.
- Пропуски темы допустимы, пустой текст исключается из основного набора и остаётся в аудите.
- Подробный EDA не затрагивает отложенные тестовые выборки.

Большое число пересечений с контролем подтверждает необходимость проверки идентичности до обучения.
Глобальная дедупликация выполнена до разбиения; TF-IDF обучается только внутри соответствующего train-фолда.
""")
md("""
<a id="modeling"></a>
# 3. Обучение и кросс-валидация

## 3.1 Выбор архитектур и baseline

| Модель | Роль в эксперименте |
|---|---|
| Dummy: always ham | Сохраняет всю обычную почту, но не находит спам; показывает, почему одного FPR недостаточно |
| ComplementNB, α = 0.5 / 1.0 | Быстрый текстовый baseline, сравнение сглаживания |
| LogisticRegression, C = 0.5 / 2.0 | Линейная модель на разреженном TF-IDF; вес ham вдвое выше |
| LinearSVC, C = 1.0 | Линейный SVM; хорошо подходит для большого числа разреженных признаков |

Для корпуса такого размера линейные модели позволяют быстро сравнить несколько вариантов
без скачивания трансформеров. Бустинг не обязателен по заданию; выбранные модели работают
непосредственно с разреженными векторами TF-IDF. Числовые признаки EDA не используются при обучении.

Каждая конфигурация проходит 5-fold CV внутри train. Сначала предпочитаем модели со средними
Precision ≥ 0.99 и FPR ≤ 0.01, затем максимизируем средний F0.5, далее учитываем FPR и Recall.
Если ограничения на CV недостижимы, выбираем лучший F0.5 и явно сохраняем этот факт в таблице.
Средние по фолдам — сравнение моделей, а не несмещённая итоговая оценка выбранного победителя.
""")
code("""
# train_experiment: CV → fit на train → порог на validation → фиксированная оценка.
# Одни и те же функции вызываются из ноутбука и build.py.
experiment = train_experiment(prepared, report_dir=PROJECT_ROOT / 'reports')
model = experiment['model']
summary = experiment['summary']
cv_table = experiment['cv_table']
threshold = float(summary['threshold']['threshold'])
display(cv_table)
display(Markdown(f'**Выбрана модель:** {summary["selected_model"]}.'))
""")
md("""
## 3.2 Сравнение моделей на кросс-валидации

Сравниваем среднее качество и разброс между фолдами при стандартном пороге каждой модели.
TF-IDF включён в Pipeline и заново обучается на каждом фолде: словарь и IDF проверочного фолда
не используются при обучении. Параметры кандидатов зафиксированы до просмотра тестовых результатов.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for metric, color in [('precision', '#159895'), ('recall', '#e76f51'), ('f1', '#457b9d')]:
    axes[0].errorbar(cv_table[metric + '_mean'], np.arange(len(cv_table)),
                     xerr=cv_table[metric + '_std'], fmt='o', capsize=3, label=metric, color=color)
axes[0].set_yticks(np.arange(len(cv_table)), cv_table.model)
axes[0].set(xlim=(-0.03, 1.05), title='Среднее ± SD по 5 фолдам', xlabel='Метрика')
axes[0].legend()
axes[1].barh(cv_table.model, cv_table.fpr_mean, color='#e76f51')
axes[1].axvline(MAX_FPR, color='black', linestyle='--', label='FPR = 1%')
axes[1].set(title='Ложные блокировки на CV', xlabel='Средний FPR')
axes[1].legend()
finish_figure(fig, FIGURES, '07_cv_comparison')
display(experiment['cv_folds'])
""")
md("""
## 3.3 Подбор порога с приоритетом обычной почты

Стандартный порог — 0.5 для моделей с `predict_proba` и 0 для `decision_function` SVC.
Оценки вероятности не считаем отдельно откалиброванными.
Рассматриваем середины интервалов между уникальными оценками на validation, плюс крайние решения.

Из порогов с Precision ≥ 0.99, FPR ≤ 0.01 и найденным спамом выбираем минимум FP,
затем максимум Recall. Если допустимого порога нет, сохраняем явный статус отказа и
решение «всё ham», которое не считается успешным антиспамом.

Такой выбор порога согласуется с [руководством scikit-learn](https://scikit-learn.org/stable/modules/classification_threshold.html):
порог нужно настраивать на данных, не использованных при fit. Здесь это validation.
""")
code("""
threshold_info = summary['threshold']
display(pd.Series(threshold_info, name='Выбранный порог').to_frame())
search = experiment['threshold_table']
display(search.loc[search.constraints_met].sort_values(['fp', 'recall'], ascending=[True, False]).head(10))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for metric in ['precision', 'recall']:
    axes[0].plot(search.threshold, search[metric], label=metric)
axes[0].axvline(threshold, color='black', linestyle='--', label='Выбранный порог')
axes[0].axhline(TARGET_PRECISION, color='gray', linestyle=':')
axes[0].set(title='Порог и качество на validation', xlabel='Порог по шкале модели', ylabel='Метрика')
axes[0].legend()
axes[1].plot(search.threshold, search.fpr, color='#e76f51', label='FPR')
axes[1].axvline(threshold, color='black', linestyle='--')
axes[1].axhline(MAX_FPR, color='gray', linestyle=':', label='Допустимый FPR')
axes[1].set(title='Порог и ложные блокировки', xlabel='Порог по шкале модели', ylabel='FPR')
axes[1].legend()
finish_figure(fig, FIGURES, '08_threshold_selection')
""")
md("""
<a id="evaluation"></a>
# 4. Контрольный тест и анализ ошибок

## 4.1 Метрики на отложенных данных

Показываем стандартный и выбранный пороги. После этой оценки модель и порог не меняются.
`external_all` содержит весь читаемый контрольный архив, включая повторы и пустой текст;
`external_unique` — основной очищенный контроль без повторов, пустого текста и конфликтующих меток.
Разницу размеров фиксируем в аудите, результаты полного набора также сохраняем.
Внешние метрики не используются для выбора модели, параметров или порога.
""")
code("""
metric_rows = []
for dataset, metrics in summary['evaluations'].items():
    for policy_name, values in [('standard', metrics['default_threshold_metrics']), ('selected', metrics)]:
        metric_rows.append({'dataset': dataset, 'policy': policy_name, **{
            key: values[key] for key in ['n', 'precision', 'recall', 'f1', 'f05', 'fpr', 'fp', 'fn', 'fpr_upper_95']
        }})
metric_table = pd.DataFrame(metric_rows)
metric_table.to_csv(PROJECT_ROOT / 'reports/final_comparison.csv', index=False)
display(metric_table.round(5))
display(Markdown(
    '**Вывод.** Выбранный порог контролирует риск по validation. '
    'Выполнение ограничений на test и внешнем контроле оцениваем по фактическим метрикам; '
    'никаких гарантий по ним заранее нет.'
))
""")
md("""
## 4.2 Матрицы ошибок

Строки — истинный класс, столбцы — предсказание. Верхний правый элемент — FP:
полезное письмо ошибочно признано спамом. Нижний левый — пропущенный спам.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for dataset, ax in zip(['test', 'external_unique'], axes):
    predictions = experiment['predictions'][dataset]
    ConfusionMatrixDisplay.from_predictions(predictions.label, predictions.prediction,
        labels=[0, 1], display_labels=['ham', 'spam'], cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(dataset)
finish_figure(fig, FIGURES, '09_confusion_matrices')
""")
md("""
## 4.3 Precision–Recall и ROC

Эти кривые характеризуют ранжирование и показаны для итоговой диагностики.
PR-кривая особенно полезна при изменении доли спама. Рабочую точку после просмотра кривых не меняем.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for dataset in ['test', 'external_unique']:
    predictions = experiment['predictions'][dataset]
    PrecisionRecallDisplay.from_predictions(predictions.label, predictions.score, name=dataset, ax=axes[0])
    RocCurveDisplay.from_predictions(predictions.label, predictions.score, name=dataset, ax=axes[1])
axes[0].set_title('Precision–Recall')
axes[1].set_title('ROC')
finish_figure(fig, FIGURES, '10_pr_roc')
""")
md("""
## 4.4 Отдельная проверка hard_ham и разбор ошибок

Для источника, содержащего только ham, основная метрика — FPR; Precision/Recall спама
в такой подвыборке не содержательны и не интерпретируются. Сравниваем число FP и объём ham.
Тесты не переобучаем и ручные исключения по найденным ошибкам не добавляем.
""")
code("""
source_metrics = experiment['by_source']['test']
display(source_metrics[['source', 'n', 'ham', 'spam', 'fp', 'fn', 'fpr', 'fpr_upper_95']])
for dataset in ['test', 'external_unique']:
    predictions = experiment['predictions'][dataset]
    errors = predictions.loc[predictions.error_type.ne('correct')]
    display(Markdown(f'**{dataset}: {len(errors)} ошибок.** Примеры (тема, без служебных заголовков):'))
    display(errors[['source', 'label', 'prediction', 'score', 'error_type', 'subject']].head(12))
    display(predictions.groupby(['source', 'error_type']).size().rename('emails').to_frame())
""")
md("""
## 4.5 Неопределённость ложноположительных срабатываний

Нулевое число FP не означает нулевую вероятность ошибки. Для FPR вычислена односторонняя
верхняя 95%-граница точного интервала Clopper–Pearson. Например, при 0 FP из 100 ham
она составляет примерно 3%. Это интервал для фиксированного классификатора при независимых
наблюдениях; сообщения из одной рассылки могут нарушать эту предпосылку.
На validation граница дополнительно смещена адаптивным выбором порога, поэтому ориентируемся на test/контроль.
""")
code("""
uncertainty = metric_table.loc[(metric_table.policy == 'selected') & metric_table.dataset.ne('validation'),
    ['dataset', 'n', 'fp', 'fpr', 'fpr_upper_95']]
display(uncertainty)
for dataset in ['test', 'external_unique']:
    values = summary['evaluations'][dataset]
    display(Markdown(
        f'**{dataset}:** FP = {values["fp"]} из {values["ham"]} ham; '
        f'наблюдаемый FPR = {values["fpr"]:.2%}, верхняя 95%-граница = {values["fpr_upper_95"]:.2%}.'
    ))
""")
md("""
## 4.6 Пример использования обученной модели в памяти

Это локальный вызов модели, без сервера, контейнера и сериализации.
Демонстрационные тексты не являются дополнительной оценкой качества.
""")
code("""
from email.message import EmailMessage

examples = []
for subject, body in [
    ('Team meeting', 'Please review the meeting agenda and project report before Monday.'),
    ('Congratulations! Claim your prize', 'WIN cash now! Click here to claim your FREE prize and special offer.'),
]:
    message = EmailMessage()
    message['Subject'] = subject
    message.set_content(body)
    parsed = parse_email(message.as_bytes())
    score = float(spam_scores(model, [parsed['text']])[0])
    examples.append({'subject': subject, 'score': score, 'threshold': threshold,
                     'prediction': 'spam' if score >= threshold else 'ham'})
display(pd.DataFrame(examples))
""")
md("""
<a id="results"></a>
# 5. Результаты, ограничения и воспроизводимость

## 5.1 Итоги эксперимента
""")
code("""
internal = summary['evaluations']['test']
control = summary['evaluations']['external_unique']
full_control = summary['evaluations']['external_all']
status = 'выполнены' if control['precision'] >= TARGET_PRECISION and control['fpr'] <= MAX_FPR else 'не выполнены'
display(Markdown(f'''
**Этап 1 выполнен:** из {len(raw):,} писем исходных корпусов подготовлено {len(prepared.clean):,}
непересекающихся с внешним контролем уникальных писем. Размеры train/validation/test:
{len(train):,}/{len(validation):,}/{len(test):,}. Исходные файлы и причины исключений сохранены.

**Этап 2 выполнен:** сравнены шесть конфигураций на {CV_FOLDS}-fold CV; выбрана
**{summary['selected_model']}**, порог **{threshold:.6f}**. Статус выбора порога:
`{summary['threshold']['selection_status']}`.

| Выборка | Precision | Recall | F1 | FP | FPR |
|---|---:|---:|---:|---:|---:|
| Внутренний test | {internal['precision']:.4f} | {internal['recall']:.4f} | {internal['f1']:.4f} | {internal['fp']} | {internal['fpr']:.4%} |
| Внешний контроль, уникальные | {control['precision']:.4f} | {control['recall']:.4f} | {control['f1']:.4f} | {control['fp']} | {control['fpr']:.4%} |
| Внешний контроль, полный | {full_control['precision']:.4f} | {full_control['recall']:.4f} | {full_control['f1']:.4f} | {full_control['fp']} | {full_control['fpr']:.4%} |

На очищенном внешнем контроле ограничения Precision ≥ {TARGET_PRECISION:.0%} и FPR ≤ {MAX_FPR:.0%}
**{status}**. Результат отражён фактически; внешние метрики не использованы для изменения решения.
'''))
""")
md("""
## 5.2 Ограничения и дальнейшие шаги

- Корпуса исторические, преимущественно англоязычные. Проверка на современной русскоязычной почте не проведена.
- Три локальных набора относятся к SpamAssassin. Самостоятельные LingSpam/Enron/другие домены
  могут улучшить разнообразие, но должны добавляться с отдельной проверкой лицензий и дубликатов.
- Точная дедупликация не исключает близкие шаблоны, цитирование и зависимость писем внутри рассылок.
  Для строгой следующей итерации полезны группировка по цепочкам и временной контроль.
- Снижение FP обычно уменьшает Recall. Нельзя бесконечно повышать порог: классификатор «всё ham» бесполезен.
- После просмотра итогового контроля его нужно считать использованным: для новых решений потребуется свежий контроль.
- Для мультиязычности нужны размеченные письма разных языков и проверка метрик отдельно по языкам;
  возможна следующая итерация с символьными n-граммами или multilingual embeddings.
- Для concept drift нужны мониторинг FP по обратной связи, изменение распределений текста,
  версионирование данных и регулярное переобучение с новой независимой валидацией порога.
- При полуобучении нельзя автоматически считать все предсказания истинной разметкой:
  полезны ручная проверка спорных примеров и отдельная оценка подтверждённых меток.

Модель обучена в памяти текущего kernel. `build.py` заново обучает её, сохраняет обработанные данные,
метрики, предсказания, Pipeline и порог для сервиса. `export_model.py` восстанавливает выбранную модель
на сохранённом train и проверяет совпадение предсказаний без повторной настройки.
FastAPI и Docker описаны в разделе 6. Публичный хостинг и отправка презентации не выполнялись.
""")
md(r"""
## 5.3 Как воспроизвести

Из папки `PROJECT_antispam`:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.download_external
.venv\Scripts\python build.py
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python run_notebook.py
```

На Linux/macOS вместо `.venv\Scripts\python` используется `.venv/bin/python`.
`build.py` — программный ML-конвейер, `run_notebook.py` выполняет все ячейки и записывает готовый ноутбук.
`make_notebook.py` генерирует исходный ноутбук, стирая его результаты; для обычного запуска он не нужен.

Сохранены `data/processed/*.csv.gz`, аудит, параметры эксперимента, результаты CV,
поиск порога, предсказания, PNG-графики и итоговые метрики в `reports/`.
Фиксированы random state, разбиение, список моделей и зависимости.

Методические источники: [защита от утечки в scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html),
[настройка порога](https://scikit-learn.org/stable/modules/classification_threshold.html),
[описание SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/readme.html).
""")

md("""
<a id="packaging"></a>
# 6. Упаковка модели: FastAPI и Docker

## 6.1 Артефакт обученной модели

Сохраняются TF-IDF и классификатор как единый `Pipeline`, выбранный порог, тип оценки,
версии библиотек и SHA-256 файла. Сервис проверяет контрольную сумму, версии и согласованность
метаданных при запуске. Загружается только собственный локальный артефакт;
пользовательские файлы не десериализуются как модели.

`export_model.py` обучает уже выбранную конфигурацию только на сохранённом train и
проверяет точное совпадение решений и численное совпадение оценок на validation, test и внешнем контроле.
Параметры и порог по результатам упаковки не подбираются.
""")
code("""
from src.artifacts import DEFAULT_MODEL_DIR, load_artifact, save_artifact

if not (DEFAULT_MODEL_DIR / 'model.joblib').exists():
    save_artifact(model, summary, DEFAULT_MODEL_DIR, train=train)
service_bundle, service_metadata = load_artifact(DEFAULT_MODEL_DIR)
np.testing.assert_allclose(
    spam_scores(service_bundle['pipeline'], test.text), spam_scores(model, test.text),
    rtol=1e-12, atol=1e-12,
)
assert service_bundle['threshold'] == threshold
display(pd.Series({key: service_metadata[key] for key in [
    'model_name', 'threshold', 'score_kind', 'training_rows', 'model_sha256', 'packages',
]}).to_frame('Артефакт сервиса'))
print('Экспортированная модель совпадает с моделью эксперимента.')
""")
md(r"""
## 6.2 HTTP-сервис и запуск контейнера

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Состояние загруженной модели |
| GET | `/model` | Версия, порог и контрольная сумма модели |
| POST | `/predict` | JSON: `text` либо `subject`, `body`, `body_format` |
| POST | `/predict/eml` | Загрузка RFC 822/MIME `.eml` |
| GET | `/docs` | Swagger для интерактивной демонстрации |

Пример запроса:

```json
{"subject": "Team meeting", "body": "Please review the report before Monday."}
```

Ответ включает `label`, `is_spam`, `score`, `threshold`, `score_kind`, `model_name`.
JSON и `.eml` используют тот же парсер MIME/HTML, что и обучение.
Пустой текст возвращает HTTP 422; для `.eml` действует ограничение 2 МиБ (HTTP 413).
Обучение при HTTP-запросах не выполняется; модель загружается один раз при старте.

Из папки проекта в PowerShell:

```powershell
.\start_docker.ps1                  # сборка образа и запуск с ожиданием healthcheck
.\start_docker.ps1 -SkipBuild       # запуск готового образа
.\start_docker.ps1 -Port 8080       # другой порт
.\stop_docker.ps1                   # остановка контейнера проекта
```

Контейнер запускается пользователем без root-прав, содержит модель и зависимости инференса;
обучающие данные и ноутбук не копируются в образ. Swagger: `http://127.0.0.1:8000/docs`.
Подробности и команды проверки: `docs/DOCKER.md`.
""")
code("""
verification_file = PROJECT_ROOT / 'reports/docker_api_verification.json'
if verification_file.exists():
    docker_verification = json.loads(verification_file.read_text(encoding='utf-8'))
    display(pd.DataFrame(docker_verification['prediction_checks']))
    display(Markdown(
        '**Этап 3 выполнен.** Docker-контейнер собран и проверен реальными HTTP-запросами: '
        'healthcheck, информация о модели, JSON, загрузка .eml и валидация входа. '
        'Предсказания контейнера совпали с сохранёнными результатами эксперимента. '
        'Это сохранённый отчёт проверки; текущее состояние контейнера проверяется через `/health`.'
    ))
else:
    display(Markdown('Конфигурация сервиса подготовлена. Для сохранения HTTP-проверки выполните '
                     '`python scripts/smoke_api.py` после запуска контейнера.'))
""")


def main():
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    for i, cell in enumerate(cells):
        cell["id"] = f"antispam-{i:03d}"
    target = ROOT / "project/PROJECT.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Created {target}: {len(cells)} cells")


if __name__ == "__main__":
    main()
