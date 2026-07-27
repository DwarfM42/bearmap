"""data/raw/manifest.parquet への追記を担う共通ユーティリティ。

台帳は追記専用（append-only）のログとして扱う。同じURLを再取得しても
既存行を上書きせず、取得のたびに新しい行を追加する。取得のたびに
sha256が変われば、提供元データが更新されたことが後から追跡できる。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

MANIFEST_SCHEMA = {
    "url": pl.Utf8,
    "local_path": pl.Utf8,
    "source_provider": pl.Utf8,
    "source_dataset": pl.Utf8,
    "fetched_at": pl.Utf8,
    "source_last_modified": pl.Utf8,
    "file_size_bytes": pl.Int64,
    "sha256": pl.Utf8,
    "license": pl.Utf8,
    "notes": pl.Utf8,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_record(
    *,
    url: str,
    local_path: Path,
    source_provider: str,
    source_dataset: str,
    source_last_modified: str | None,
    license_: str,
    notes: str = "",
) -> dict:
    return {
        "url": url,
        "local_path": str(local_path.as_posix()),
        "source_provider": source_provider,
        "source_dataset": source_dataset,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_last_modified": source_last_modified or "",
        "file_size_bytes": local_path.stat().st_size,
        "sha256": sha256_file(local_path),
        "license": license_,
        "notes": notes,
    }


def append_manifest(manifest_path: Path, records: list[dict]) -> None:
    if not records:
        return
    new_rows = pl.DataFrame(records, schema=MANIFEST_SCHEMA)
    if manifest_path.exists():
        existing = pl.read_parquet(manifest_path)
        combined = pl.concat([existing, new_rows], how="vertical_relaxed")
    else:
        combined = new_rows
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(manifest_path)
