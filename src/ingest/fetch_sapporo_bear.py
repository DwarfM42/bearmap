"""札幌市内のヒグマ出没情報CSV（2017〜2025年）をrawのまま取得する。

再実行可能：既存のrawファイルがあっても毎回再取得し、上書きする
（提供元が値を訂正・追記する可能性があるため）。取得の都度、
manifestに新しい行を追記する（既存行は消さない）。
"""

from __future__ import annotations

from pathlib import Path

import requests

from manifest import append_manifest, build_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sapporo_bear"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.parquet"

SOURCE_PROVIDER = "札幌市環境局環境都市推進部環境管理担当課"
SOURCE_DATASET = "札幌市内のヒグマ出没情報"
LICENSE = "CC-BY 4.0"

# 2026-07-27にckan.pf-sapporo.jpのデータセットページ
# (https://ckan.pf-sapporo.jp/dataset/sapporo_bear_appearance) から
# 実際に確認したリソースURL。
RESOURCE_URLS = {
    2017: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/6d2ebe8d-d683-41b6-83b5-0395a3e795ae/download/2017sapporobearappearance.csv",
    2018: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/e33993cc-4ef1-4916-9cad-1e9d585f9427/download/2018sapporobearappearance.csv",
    2019: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/6a9c917a-1fe1-4217-876b-e1ffa5138144/download/2019sapporobearappearance.csv",
    2020: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/9647f46b-6e07-4209-8b3e-45c8b329e579/download/2020sapporobearappearance.csv",
    2021: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/a9255555-4afa-4450-8c00-8bac4b24d088/download/2021sapporobearappearance.csv",
    2022: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/37fd8fe6-b1c1-4c0a-b3a8-85cc3958603d/download/2022sapporobearappearance.csv",
    2023: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/3d6c0e28-7247-4503-b248-258e59192b99/download/2023sapporobearappearance.csv",
    2024: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/b289a37b-9149-4e34-981f-6743488d5779/download/2024sapporobearappearance.csv",
    2025: "https://ckan.pf-sapporo.jp/dataset/0d3197ef-c473-48ac-86bd-0fc34084b0ee/resource/76c539c8-cd17-4449-a972-6ddc8c3d5306/download/2025sapporobearappearance.csv",
}


def fetch_year(year: int, url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / f"{year}.csv"
    local_path.write_bytes(resp.content)
    last_modified = resp.headers.get("Last-Modified")
    return build_record(
        url=url,
        local_path=local_path.relative_to(PROJECT_ROOT),
        source_provider=SOURCE_PROVIDER,
        source_dataset=SOURCE_DATASET,
        source_last_modified=last_modified,
        license_=LICENSE,
        notes=f"{year}年分" + ("（年度途中の暫定値の可能性あり）" if year == 2025 else ""),
    )


def main() -> None:
    records = []
    for year, url in sorted(RESOURCE_URLS.items()):
        print(f"fetching {year} ...")
        records.append(fetch_year(year, url))
    append_manifest(MANIFEST_PATH, records)
    print(f"done. {len(records)} files recorded in manifest.")


if __name__ == "__main__":
    main()
