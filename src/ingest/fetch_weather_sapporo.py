"""気象庁「過去の気象データ検索」から札幌（観測所番号 prec_no=14, block_no=47412）の
日別データページ（HTML）を月単位でrawのまま取得する。

このURLは実際にブラウザ相当のGETで内容を確認済み（2026-07-27、prec_no=14,
block_no=47412のページに「札幌」の表記と積雪・降水量の実データを確認）。
気象庁は表形式のHTMLページを提供しており、構造化CSVの直接ダウンロードは
別途フォーム操作が必要なため、本スクリプトではP2bの範囲としてこのHTMLページを
そのまま保存する（rawの定義：提供元から得られた形式のまま、加工しない）。

対象期間は札幌市ヒグマ出没データと同じ2017〜2025年（暦年）。
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from manifest import append_manifest, build_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather_sapporo"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.parquet"

SOURCE_PROVIDER = "気象庁"
SOURCE_DATASET = "過去の気象データ検索（札幌、日別値）"
LICENSE = "気象庁ウェブサイト利用規約（出典明記により二次利用可）"

PREC_NO = 14
BLOCK_NO = 47412
YEARS = range(2017, 2026)
REQUEST_INTERVAL_SEC = 0.5


def fetch_month(year: int, month: int) -> dict:
    url = (
        "https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php"
        f"?prec_no={PREC_NO}&block_no={BLOCK_NO}&year={year}&month={month}&day=&view="
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / f"{year}-{month:02d}.html"
    local_path.write_bytes(resp.content)
    last_modified = resp.headers.get("Last-Modified")
    return build_record(
        url=url,
        local_path=local_path.relative_to(PROJECT_ROOT),
        source_provider=SOURCE_PROVIDER,
        source_dataset=SOURCE_DATASET,
        source_last_modified=last_modified,
        license_=LICENSE,
        notes=f"prec_no={PREC_NO}, block_no={BLOCK_NO}（札幌）, {year}年{month}月",
    )


def main() -> None:
    records = []
    for year in YEARS:
        for month in range(1, 13):
            print(f"fetching {year}-{month:02d} ...")
            records.append(fetch_month(year, month))
            time.sleep(REQUEST_INTERVAL_SEC)
    append_manifest(MANIFEST_PATH, records)
    print(f"done. {len(records)} files recorded in manifest.")


if __name__ == "__main__":
    main()
