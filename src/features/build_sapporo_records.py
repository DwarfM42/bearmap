"""札幌市ヒグマ出没rawデータ（2017〜2025年）から、レコード単位の
クリーニング済みテーブルを作る。

このスクリプトは値の削除・要約は行わない（元のレコード数を保つ）。
行うのは、列名の標準化、種別の正規化、重複候補グループの付与のみ。

出力: data/interim/sapporo_records.parquet
1レコード = 1通報（原記録）。重複候補はフラグ（possible_duplicate_group）
で示すのみで、削除・統合はしない。
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sapporo_bear"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "sapporo_records.parquet"

SOURCE_PROVIDER = "札幌市環境局環境都市推進部環境管理担当課"
SOURCE_DATASET = "札幌市内のヒグマ出没情報"

# P2_GATE.md 第1節の結論：日付・時刻フィールドの意味は公式資料で確認できず、
# 未確定のまま扱う。H6（曜日周期）の解釈にはこの値をそのまま使わない。
DATE_SEMANTICS = "undetermined"

# 重複候補グループの付与ルール（試作版）。
# Tsuruga et al. (2026) のようなGPS移動データに基づく較正は行っておらず、
# 恣意的な固定閾値である。本番の識別戦略には使わず、
# 「原記録ベース」と「重複候補集約ベース」の両方を生成できることを
# 示すための試作ルールとして明示的にバージョン管理する。
DUPLICATE_RULE_VERSION = "v1_naive_1500m_3day_sameward"
DUP_DISTANCE_M = 1500.0
DUP_DAYS = 3


def normalize_event_type(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw
    if "カメラ" in s:
        return "camera"
    if "捕獲" in s:
        return "capture"
    if "らしき" in s:
        return "sighting_uncertain"
    if "目撃" in s:
        return "sighting"
    if any(k in s for k in ["フン", "糞", "足跡", "食痕", "枝折り", "痕跡"]):
        return "track_sign"
    return "unknown"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_year(path: Path, year: int) -> pl.DataFrame:
    df = pl.read_csv(path, encoding="utf8-lossy", infer_schema_length=0)
    n_raw = df.height
    df = df.with_columns(
        [
            pl.lit(year).alias("_source_year"),
            pl.int_range(0, pl.len()).alias("_row_index"),
        ]
    )
    df = df.with_columns(
        [
            (pl.lit(f"sapporo_bear:{year}:") + pl.col("_row_index").cast(pl.Utf8).str.zfill(4)).alias(
                "source_record_id"
            ),
            pl.col("日付").alias("event_date_raw"),
            pl.col("時刻").alias("event_time_raw"),
            pl.col("区").alias("ward"),
            pl.col("出没場所").alias("location_text"),
            pl.col("緯度").cast(pl.Float64, strict=False).alias("lat"),
            pl.col("経度").cast(pl.Float64, strict=False).alias("lon"),
            pl.col("状況").alias("event_type_original"),
        ]
    )
    assert df.height == n_raw, "loading must not change row count"
    return df.select(
        [
            "source_record_id",
            "event_date_raw",
            "event_time_raw",
            "ward",
            "location_text",
            "lat",
            "lon",
            "event_type_original",
            "_source_year",
        ]
    )


def assign_duplicate_groups(df: pl.DataFrame) -> pl.Series:
    """区・日付差・距離に基づく素朴な同一個体候補クラスタリング（Union-Find）。"""
    n = df.height
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rows = df.select(["ward", "event_date_parsed", "lat", "lon"]).to_dicts()
    # ward, date でソートしてから近傍探索（全件総当りは行わず、区単位でしか候補にしない）
    by_ward: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_ward.setdefault(r["ward"] or "unknown", []).append(i)

    for idxs in by_ward.values():
        for a_pos in range(len(idxs)):
            i = idxs[a_pos]
            ri = rows[i]
            if ri["event_date_parsed"] is None or ri["lat"] is None or ri["lon"] is None:
                continue
            for b_pos in range(a_pos + 1, len(idxs)):
                j = idxs[b_pos]
                rj = rows[j]
                if rj["event_date_parsed"] is None or rj["lat"] is None or rj["lon"] is None:
                    continue
                day_diff = abs((ri["event_date_parsed"] - rj["event_date_parsed"]).days)
                if day_diff > DUP_DAYS:
                    continue
                dist = haversine_m(ri["lat"], ri["lon"], rj["lat"], rj["lon"])
                if dist <= DUP_DISTANCE_M:
                    union(i, j)

    roots = [find(i) for i in range(n)]
    # グループIDを安定した文字列にする（root行のsource_record_idを使う）
    ids = df.get_column("source_record_id").to_list()
    group_ids = [ids[r] for r in roots]
    return pl.Series("possible_duplicate_group", group_ids)


def main() -> None:
    csv_paths = sorted(RAW_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"no raw CSV found under {RAW_DIR}. run fetch_sapporo_bear.py first.")

    frames = []
    n_total_raw = 0
    for p in csv_paths:
        year = int(p.stem)
        df = load_year(p, year)
        n_total_raw += df.height
        frames.append(df)
    all_df = pl.concat(frames, how="vertical_relaxed")
    assert all_df.height == n_total_raw, "concatenation must not change row count"

    all_df = all_df.with_columns(
        [
            pl.col("event_date_raw").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("event_date_parsed"),
            pl.lit(SOURCE_PROVIDER).alias("source_provider"),
            pl.lit(SOURCE_DATASET).alias("source_dataset"),
            pl.lit(DATE_SEMANTICS).alias("date_semantics"),
            pl.when(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
            .then(pl.lit("point"))
            .otherwise(pl.lit("missing"))
            .alias("location_precision"),
            pl.col("event_type_original").map_elements(normalize_event_type, return_dtype=pl.Utf8).alias(
                "event_type_normalized"
            ),
        ]
    )

    dup_group = assign_duplicate_groups(all_df)
    all_df = all_df.with_columns(dup_group)
    all_df = all_df.with_columns(pl.lit(DUPLICATE_RULE_VERSION).alias("duplicate_rule_version"))

    assert all_df.height == n_total_raw, "final row count must equal sum of raw row counts"

    out_cols = [
        "source_provider",
        "source_dataset",
        "source_record_id",
        "event_date_raw",
        "event_date_parsed",
        "event_time_raw",
        "date_semantics",
        "ward",
        "location_text",
        "lat",
        "lon",
        "location_precision",
        "event_type_original",
        "event_type_normalized",
        "possible_duplicate_group",
        "duplicate_rule_version",
        "_source_year",
    ]
    result = all_df.select(out_cols)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(OUT_PATH)
    print(f"rows in: {n_total_raw}, rows out: {result.height}")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
