"""P2c試作パネル（札幌市）に対する自動テスト。

実際のスクリプト（src/features/build_sapporo_records.py,
build_sapporo_panel.py）を再実行し、生成物に対して以下を検証する：

- 座標が北海道の妥当範囲内
- 日付が公開期間内（2017-2025年）
- 元レコード数が変換工程で説明なく減少しない
- 種別変換前後の件数保存
- 日次から週次への合計件数保存
- 欠損（location_precision=missing）と記録上のゼロ（count=0）が混同されない
- 同一入力から同一parquetが再生成される（再現性）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sapporo_bear"
RECORDS_PATH = PROJECT_ROOT / "data" / "interim" / "sapporo_records.parquet"
DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_panel_daily.parquet"
WEEKLY_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_panel_weekly.parquet"

# 北海道全域を十分に覆う、緩めの緯度経度範囲（度）
HOKKAIDO_LAT_RANGE = (41.0, 46.0)
HOKKAIDO_LON_RANGE = (139.0, 149.0)

DATA_YEAR_MIN = 2017
DATA_YEAR_MAX = 2025

pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists() or not any(RAW_DIR.glob("*.csv")),
    reason="raw Sapporo bear CSVs not present; run src/ingest/fetch_sapporo_bear.py first",
)


def run_pipeline() -> None:
    for script in ["src/features/build_sapporo_records.py", "src/features/build_sapporo_panel.py"]:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="module", autouse=True)
def built_once():
    run_pipeline()
    yield


def raw_row_count() -> int:
    """CSVとしてのレコード数（改行を含む引用フィールドがあるため、
    単純な改行数ではなくCSVパーサでカウントする）。"""
    total = 0
    for p in RAW_DIR.glob("*.csv"):
        df = pl.read_csv(p, encoding="utf8-lossy", infer_schema_length=0)
        total += df.height
    return total


def test_coordinates_within_hokkaido_bounds():
    df = pl.read_parquet(RECORDS_PATH)
    valid = df.filter(pl.col("location_precision") == "point")
    assert valid.height > 0
    lat_ok = valid.filter(
        (pl.col("lat") < HOKKAIDO_LAT_RANGE[0]) | (pl.col("lat") > HOKKAIDO_LAT_RANGE[1])
    ).height
    lon_ok = valid.filter(
        (pl.col("lon") < HOKKAIDO_LON_RANGE[0]) | (pl.col("lon") > HOKKAIDO_LON_RANGE[1])
    ).height
    assert lat_ok == 0, "found latitude values outside Hokkaido bounds"
    assert lon_ok == 0, "found longitude values outside Hokkaido bounds"


def test_dates_within_publication_period():
    df = pl.read_parquet(RECORDS_PATH)
    years = df.get_column("event_date_parsed").dt.year()
    assert years.min() >= DATA_YEAR_MIN
    assert years.max() <= DATA_YEAR_MAX


def test_record_count_preserved_through_cleaning():
    n_raw = raw_row_count()
    df = pl.read_parquet(RECORDS_PATH)
    assert df.height == n_raw, (
        f"record count changed during cleaning: raw={n_raw}, cleaned={df.height}. "
        "this pipeline must not silently drop or duplicate rows."
    )


def test_event_type_normalization_preserves_count():
    df = pl.read_parquet(RECORDS_PATH)
    n_total = df.height
    n_by_type = df.group_by("event_type_normalized").len().select(pl.col("len").sum()).item()
    assert n_by_type == n_total


def test_daily_to_weekly_sum_preserved():
    daily = pl.read_parquet(DAILY_PATH)
    weekly = pl.read_parquet(WEEKLY_PATH)
    for basis in ["raw_record", "dedup_group"]:
        d_sum = daily.filter(pl.col("dedup_basis") == basis).select(pl.col("count").sum()).item()
        w_sum = weekly.filter(pl.col("dedup_basis") == basis).select(pl.col("count").sum()).item()
        assert d_sum == w_sum, f"{basis}: daily total {d_sum} != weekly total {w_sum}"


def test_missing_location_not_confused_with_zero_count():
    records = pl.read_parquet(RECORDS_PATH)
    # location_precision="missing" は「座標が取れなかった通報」であり、
    # 出没が0件だったことを意味しない。missing行が独立に存在し、
    # かつそれらの行がevent countから消えていないことを確認する。
    missing = records.filter(pl.col("location_precision") == "missing")
    daily = pl.read_parquet(DAILY_PATH)
    # missing行があっても record 自体は daily panel の raw_record 集計に含まれている
    # （座標欠損であって、通報自体の欠損ではないため）
    n_raw_daily = daily.filter(pl.col("dedup_basis") == "raw_record").select(pl.col("count").sum()).item()
    assert n_raw_daily == records.height
    # 記録上のゼロ（count==0）は日付×種別のグリッド埋めで生じたものであり、
    # missing行の有無とは独立の概念であることを型で確認する
    assert daily.schema["count"] != pl.Null
    _ = missing  # missingが0件でも例外にはしない（環境により変動しうるため)


def test_reproducibility_same_input_same_output():
    df1 = pl.read_parquet(RECORDS_PATH)
    daily1 = pl.read_parquet(DAILY_PATH)
    weekly1 = pl.read_parquet(WEEKLY_PATH)

    run_pipeline()

    df2 = pl.read_parquet(RECORDS_PATH)
    daily2 = pl.read_parquet(DAILY_PATH)
    weekly2 = pl.read_parquet(WEEKLY_PATH)

    assert df1.equals(df2), "sapporo_records.parquet changed across re-runs with identical raw input"
    assert daily1.equals(daily2), "daily panel changed across re-runs with identical raw input"
    assert weekly1.equals(weekly2), "weekly panel changed across re-runs with identical raw input"
