"""OSM Overpass APIの公開ミラーが不安定（bbox+tagクエリで頻繁に504タイムアウト）だったため、
Geofabrikが配布する北海道地域のOSM抽出ファイル（.osm.pbf、単一ファイル）を直接ダウンロードする
方式に切り替えた。ダウンロード自体は単純なHTTP GETであり、クエリタイムアウトの問題がない。

道路・土地利用・河川データはこのpbfファイルから、build_covariates.pyでosmiumを使って
ローカルに抽出する（分析範囲のバウンディングボックスで絞り込み）。

再実行可能：既存rawファイルを上書きし、manifestには新しい行を追記する。
"""

from __future__ import annotations

from pathlib import Path

import requests

from manifest import append_manifest, build_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "osm_pbf"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.parquet"

URL = "https://download.geofabrik.de/asia/japan/hokkaido-latest.osm.pbf"
SOURCE_PROVIDER = "Geofabrik GmbH (OpenStreetMap contributors data extract)"
SOURCE_DATASET = "北海道 OSM地域抽出ファイル (hokkaido-latest.osm.pbf)"
LICENSE = "ODbL 1.0 (Open Database License) — © OpenStreetMap contributors"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / "hokkaido-latest.osm.pbf"

    with requests.get(URL, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        last_modified = resp.headers.get("Last-Modified")
        with open(local_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (20 << 20) < (1 << 20):
                    print(f"  {downloaded / 1e6:.0f} MB downloaded...")

    record = build_record(
        url=URL,
        local_path=local_path.relative_to(PROJECT_ROOT),
        source_provider=SOURCE_PROVIDER,
        source_dataset=SOURCE_DATASET,
        source_last_modified=last_modified,
        license_=LICENSE,
        notes="北海道全域のOSM抽出（道路・土地利用・河川・建物等を含む）。分析範囲での絞り込みはbuild_covariates.pyで実施。",
    )
    append_manifest(MANIFEST_PATH, [record])
    print(f"saved {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
