"""e-Stat「統計データダウンロード」から令和2年国勢調査 4次メッシュ(500m)
人口・世帯データ（北海道, JGD2011）を取得する。

URLは実際にブラウザで統計地理情報システムを操作し、
統計表「人口及び世帯（JGD2011）」→「都道府県」→「01 北海道」の
ダウンロードリンクとして確認したもの（2026-07-27）。

【重要な仮定】国勢調査は5年に1度のみ実施されるため、本データは2020年
（令和2年）の値のみである。本プロジェクトの出没データは2017-2025年に
またがるが、この人口メッシュ値は全年に対して同一の値を機械的に割り当てる
（2020年国勢調査人口を2017-2025年の代理として使う）。年ごとの人口変化を
補間する処理は行っていない（2015年国勢調査データを別途取得すれば線形補間も
可能だが、本パスでは2020年単年のみを取得し、非補間版として扱う）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from manifest import append_manifest, build_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "population_mesh"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.parquet"

URL = "https://www.e-stat.go.jp/gis/statmap-search/data?statsId=T001141&code=01&downloadType=2"
SOURCE_PROVIDER = "総務省統計局（e-Stat 統計地理情報システム）"
SOURCE_DATASET = "令和2年国勢調査 4次メッシュ(500m) 人口及び世帯（JGD2011）北海道"
LICENSE = "政府統計の総合窓口(e-Stat)利用規約（出典明記により二次利用可）"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "T001141_hokkaido_mesh500_2020.zip"

    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        z.extract(inner_name, RAW_DIR)
    txt_path = RAW_DIR / inner_name

    record_zip = build_record(
        url=URL,
        local_path=zip_path.relative_to(PROJECT_ROOT),
        source_provider=SOURCE_PROVIDER,
        source_dataset=SOURCE_DATASET,
        source_last_modified=resp.headers.get("Last-Modified"),
        license_=LICENSE,
        notes=(
            "2020年（令和2年）国勢調査、単年データ。2017-2025年の全年に同一値を機械的に割り当てる仮定を"
            "使用する（補間なし、非補間版）。列T001141001が人口（総数）。KEY_CODEは9桁の4次メッシュコード。"
        ),
    )
    record_txt = build_record(
        url=URL + f" (zip内: {inner_name})",
        local_path=txt_path.relative_to(PROJECT_ROOT),
        source_provider=SOURCE_PROVIDER,
        source_dataset=SOURCE_DATASET + "（展開後CSV, Shift-JIS）",
        source_last_modified=resp.headers.get("Last-Modified"),
        license_=LICENSE,
        notes="zipから展開したCSV本体。エンコーディングはShift-JIS。",
    )
    append_manifest(MANIFEST_PATH, [record_zip, record_txt])
    print(f"saved {zip_path} and extracted {txt_path}")


if __name__ == "__main__":
    main()
