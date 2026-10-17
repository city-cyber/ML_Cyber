"""Чтение RFC 822/MIME и подготовка независимых выборок без утечки."""

import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import decode_part, body_parts, fingerprint, normalize, parse_email

from .constants import (
    DATASETS, DATA_ROOT, MISLABELED_SPAM, PROJECT_ROOT, RANDOM_STATE,
    TEST_SIZE, VALIDATION_SIZE,
)

IDENTITIES = ["raw_hash", "text_hash", "body_hash", "message_id"]
FEATURES = [
    "n_chars", "n_words", "subject_length", "n_urls", "n_exclamations",
    "uppercase_ratio", "digit_ratio", "html", "attachments",
]


def load_corpus(root: Path, sources: dict, external=False):
    rows, audit = [], []
    for source, label in sources.items():
        directory = Path(root) / source
        if not directory.is_dir():
            raise FileNotFoundError(f"Отсутствует каталог: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            reason = None
            if "__MACOSX" in path.parts or path.name.startswith("."):
                reason = "macos_metadata"
            elif not (path.suffix.lower() == ".eml" or re.match(r"^\d+\.[0-9a-f]{32}$", path.name)):
                reason = "not_email"
            elif path.name.endswith(MISLABELED_SPAM.split(".")[-1]) and label == 1:
                reason = "known_mislabeled_ham"
            if reason:
                audit.append({"path": relative, "source": source, "reason": reason})
                continue
            try:
                row = parse_email(path.read_bytes())
            except Exception as exc:
                audit.append({"path": relative, "source": source, "reason": "parse_error", "error": str(exc)})
                continue
            row.update({
                "path": relative, "source": source, "label": label,
                "class_name": "spam" if label else "ham", "external": external,
            })
            rows.append(row)
    if not rows:
        raise ValueError(f"В {root} не найдено писем")
    return pd.DataFrame(rows), pd.DataFrame(audit)


def clean_corpus(frame: pd.DataFrame):
    """Конфликтующие метки исключаем; идентичные письма оставляем один раз."""
    data = frame.copy()
    reasons = pd.Series("", index=data.index, dtype="object")
    reasons.loc[data.text.str.strip().eq("")] = "empty_text"
    for key in IDENTITIES:
        valid = data[key].ne("")
        counts = data.loc[valid].groupby(key).label.nunique()
        conflicts = set(counts[counts > 1].index)
        reasons.loc[valid & data[key].isin(conflicts)] = f"conflicting_{key}"
    seen = {key: set() for key in IDENTITIES}
    for index, row in data.loc[reasons.eq("")].iterrows():
        repeated = [key for key in IDENTITIES if row[key] and row[key] in seen[key]]
        if repeated:
            reasons.loc[index] = "duplicate_" + repeated[0]
        for key in IDENTITIES:
            if row[key]:
                seen[key].add(row[key])
    excluded = data.loc[reasons.ne(""), ["path", "source", "label"]].copy()
    excluded["reason"] = reasons.loc[excluded.index]
    return data.loc[reasons.eq("")].reset_index(drop=True), excluded


def load_external_archive(archive: Path):
    """Читаем .eml прямо из tar.xz, не полагаясь на распакованные файлы ОС."""
    rows, audit = [], []
    with tarfile.open(archive, "r:xz") as collection:
        for member in collection:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or not parts or parts[0] not in {"ham", "spam"}:
                continue
            if PurePosixPath(member.name).suffix.lower() != ".eml":
                continue
            source = parts[0]
            label = int(source == "spam")
            try:
                with collection.extractfile(member) as stream:
                    row = parse_email(stream.read())
            except Exception as exc:
                audit.append({"path": member.name, "source": source, "reason": "parse_error", "error": str(exc)})
                continue
            row.update({"path": member.name, "source": source, "label": label,
                        "class_name": source, "external": True})
            rows.append(row)
    if not rows:
        raise ValueError(f"В {archive} нет размеченных ham/spam .eml")
    return pd.DataFrame(rows), pd.DataFrame(audit)


def overlap_mask(frame: pd.DataFrame, reference: pd.DataFrame):
    mask = pd.Series(False, index=frame.index)
    for key in IDENTITIES:
        values = set(reference.loc[reference[key].ne(""), key])
        mask |= frame[key].ne("") & frame[key].isin(values)
    return mask


def assert_disjoint(left, right):
    for key in IDENTITIES:
        left_values = set(left.loc[left[key].ne(""), key])
        right_values = set(right.loc[right[key].ne(""), key])
        if left_values & right_values:
            raise AssertionError(f"Утечка между выборками по {key}")


@dataclass
class PreparedData:
    raw: pd.DataFrame
    clean: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    external_raw: pd.DataFrame
    external: pd.DataFrame
    audit: pd.DataFrame
    summary: dict


def prepare_data(data_root=DATA_ROOT, external_root=None, output_dir=None):
    external_root = Path(external_root or PROJECT_ROOT / "data/external/raw")
    raw, file_audit = load_corpus(Path(data_root), DATASETS)
    archive = external_root if external_root.is_file() else external_root.parent / "test_task_emails.tar.xz"
    if archive.is_file():
        external_raw, external_file_audit = load_external_archive(archive)
    else:
        external_raw, external_file_audit = load_corpus(
            external_root, {"ham": 0, "spam": 1}, external=True
        )
    clean, duplicate_audit = clean_corpus(raw)
    external, external_duplicate_audit = clean_corpus(external_raw)
    # Только идентичность внешних писем; их метки не используются для обучения.
    overlap = overlap_mask(clean, external_raw)
    overlap_audit = clean.loc[overlap, ["path", "source", "label"]].copy()
    overlap_audit["reason"] = "overlap_with_external"
    clean = clean.loc[~overlap].reset_index(drop=True)
    train_val, test = train_test_split(
        clean, test_size=TEST_SIZE, stratify=clean.source, random_state=RANDOM_STATE
    )
    train, validation = train_test_split(
        train_val, test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=train_val.source, random_state=RANDOM_STATE,
    )
    splits = [part.reset_index(drop=True) for part in [train, validation, test]]
    for i, left in enumerate(splits):
        assert_disjoint(left, external_raw)
        for right in splits[i + 1:]:
            assert_disjoint(left, right)
    external_file_audit["scope"] = "external"
    external_duplicate_audit["scope"] = "external"
    audit = pd.concat([
        file_audit, duplicate_audit, overlap_audit,
        external_file_audit, external_duplicate_audit,
    ], ignore_index=True)
    audit["scope"] = audit["scope"].fillna("training_corpora")
    summary = {
        "raw_emails": len(raw), "raw_by_source": raw.source.value_counts().to_dict(),
        "clean_emails_before_external_exclusion": int(len(clean) + overlap.sum()),
        "removed_external_overlaps": int(overlap.sum()), "prepared_emails": len(clean),
        "train": len(splits[0]), "validation": len(splits[1]), "test": len(splits[2]),
        "external_raw": len(external_raw), "external_unique": len(external),
        "external_labels": external_raw.class_name.value_counts().to_dict(),
        "audit_reasons": audit.groupby(["scope", "reason"]).size().to_dict(),
    }
    # JSON не поддерживает tuple-ключи.
    summary["audit_reasons"] = {" / ".join(k): int(v) for k, v in summary["audit_reasons"].items()}
    prepared = PreparedData(raw, clean, *splits, external_raw, external, audit, summary)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ["clean", "train", "validation", "test", "external"]:
            getattr(prepared, name).to_csv(output_dir / f"{name}.csv.gz", index=False)
        audit.to_csv(output_dir / "data_audit.csv", index=False)
        (output_dir / "data_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return prepared
