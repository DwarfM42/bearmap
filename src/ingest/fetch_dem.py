"""国土地理院 標高タイル（10mメッシュDEM、"dem"レイヤ）を、
札幌市分析範囲（出没データのバウンディングボックス）についてタイル形式のまま取得する。

URLパターン https://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt は
2026-07-27に実際にGETで取得し、標高値（テキスト形式）が返ることを確認済み。
ズームレベル14（10mメッシュ相当）を使用。ユーザー登録不要で取得できる
公開タイルサービスであることを確認済み。

再実行可能：既存タイルファイルを上書きし、manifestには新しい行を追記する
（ただしタイルは483枚あるため、manifestへの追記も483行になる）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import requests

from manifest import append_manifest, build_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dem_sapporo"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.parquet"

SOURCE_PROVIDER = "国土地理院（地理院タイル、標高タイル dem 10mメッシュ）"
SOURCE_DATASET = "基盤地図情報数値標高モデル 標高タイル(dem, 10mメッシュ)"
LICENSE = "国土地理院コンテンツ利用規約（出典明記により無償利用可）"

ZOOM = 14
BBOX = (42.86, 141.04, 43.18, 141.50)  # (south, west, north, east) — sapporo_records.parquetの実測範囲
REQUEST_INTERVAL_SEC = 0.15


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def main() -> None:
    s, w, n, e = BBOX
    x_min, y_max = latlon_to_tile(s, w, ZOOM)
    x_max, y_min = latlon_to_tile(n, e, ZOOM)

    records = []
    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    count = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://cyberjapandata.gsi.go.jp/xyz/dem/{ZOOM}/{x}/{y}.txt"
            resp = requests.get(url, timeout=30)
            count += 1
            if resp.status_code != 200:
                # 海域や日本国外相当のタイルは404になりうる。スキップしてログのみ残す。
                print(f"[{count}/{total}] {x},{y}: HTTP {resp.status_code}, skipped")
                time.sleep(REQUEST_INTERVAL_SEC)
                continue
            local_path = RAW_DIR / f"{x}_{y}.txt"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(resp.content)
            records.append(
                build_record(
                    url=url,
                    local_path=local_path.relative_to(PROJECT_ROOT),
                    source_provider=SOURCE_PROVIDER,
                    source_dataset=SOURCE_DATASET,
                    source_last_modified=resp.headers.get("Last-Modified"),
                    license_=LICENSE,
                    notes=f"zoom={ZOOM}, x={x}, y={y}",
                )
            )
            if count % 50 == 0:
                print(f"[{count}/{total}] fetched")
            time.sleep(REQUEST_INTERVAL_SEC)

    append_manifest(MANIFEST_PATH, records)
    print(f"done. {len(records)}/{total} tiles saved and recorded in manifest.")


if __name__ == "__main__":
    main()
