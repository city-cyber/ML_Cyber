"""Загрузка и безопасное извлечение контрольного архива задания."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import requests

from .constants import EXTERNAL_SHA256, EXTERNAL_URL, PROJECT_ROOT


def download_external(destination: Path) -> dict:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "test_task_emails.tar.xz"
    if not archive.exists():
        response = requests.get(
            "https://cloud-api.yandex.net/v1/disk/public/resources/download",
            params={"public_key": EXTERNAL_URL}, timeout=60,
        )
        response.raise_for_status()
        download = requests.get(response.json()["href"], timeout=120)
        download.raise_for_status()
        temporary = archive.with_suffix(".part")
        temporary.write_bytes(download.content)
        digest = hashlib.sha256(download.content).hexdigest()
        if digest != EXTERNAL_SHA256:
            temporary.unlink()
            raise ValueError(f"Контрольная сумма архива изменилась: {digest}")
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXTERNAL_SHA256:
        raise ValueError("SHA-256 локального архива не совпадает с исходным")
    raw = destination / "raw"
    raw.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:xz") as collection:
        members = collection.getmembers()
        total_size = sum(member.size for member in members)
        if total_size > 300_000_000 or len(members) > 30_000:
            raise ValueError("Неожиданный размер контрольного архива")
        count = 0
        top = set()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if not parts or member.isdir():
                continue
            if not member.isfile() or ".." in parts or PurePosixPath(member.name).is_absolute():
                raise ValueError(f"Недопустимый участник архива: {member.name}")
            target = raw.joinpath(*parts).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError("Выход за пределы каталога извлечения")
            target.parent.mkdir(parents=True, exist_ok=True)
            with collection.extractfile(member) as source:
                target.write_bytes(source.read())
            top.add(parts[0])
            count += 1
    manifest = {
        "public_url": EXTERNAL_URL, "archive": archive.name,
        "sha256": digest, "files": count, "top_level": sorted(top),
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "data/external")
    args = parser.parse_args()
    print(json.dumps(download_external(args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
