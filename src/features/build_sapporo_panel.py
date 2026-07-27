"""札幌市ヒグマ出没レコード（data/interim/sapporo_records.parquet）から
日次・週次の集計パネルを作る。

「原記録ベース」（1通報=1件として数える）と「重複候補集約ベース」
（possible_duplicate_groupを1件として数える）の両方を生成する。
重複候補は削除しない。集約ベースはグループ内の最も早い日付を代表日とし、
グループ内でevent_type_normalizedが揃っていればその種別、
揃っていなければ"mixed"とする。

集計上のゼロ（count=0）は「その日・週にその種別の記録がなかった」ことを
意味するのみで、ヒグマの不在を意味しない。データ収録期間内の全ての
日付・週について、記録がなくても0件の行を明示的に生成する
（欠損とは区別する。データ収録期間外は生成しない＝欠損として扱う）。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDS_PATH = PROJECT_ROOT / "data" / "interim" / "sapporo_records.parquet"
DAILY_OUT = PROJECT_ROOT / "data" / "processed" / "sapporo_panel_daily.parquet"
WEEKLY_OUT = PROJECT_ROOT / "data" / "processed" / "sapporo_panel_weekly.parquet"

RAW_BASIS = "raw_record"
DEDUP_BASIS = "dedup_group"


def build_raw_basis_daily(records: pl.DataFrame) -> pl.DataFrame:
    return (
        records.group_by(["event_date_parsed", "event_type_normalized"])
        .agg(pl.len().alias("count"))
        .rename({"event_date_parsed": "date"})
        .with_columns(pl.lit(RAW_BASIS).alias("dedup_basis"))
    )


def build_dedup_basis_daily(records: pl.DataFrame) -> pl.DataFrame:
    # グループごとに代表日（最小日付）と代表種別（揃っていればその種別、
    # 揃っていなければ"mixed"）を決める。
    group_repr = (
        records.group_by("possible_duplicate_group")
        .agg(
            [
                pl.col("event_date_parsed").min().alias("date"),
                pl.col("event_type_normalized").n_unique().alias("_n_types"),
                pl.col("event_type_normalized").first().alias("_first_type"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_n_types") == 1)
            .then(pl.col("_first_type"))
            .otherwise(pl.lit("mixed"))
            .alias("event_type_normalized")
        )
    )
    return (
        group_repr.group_by(["date", "event_type_normalized"])
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit(DEDUP_BASIS).alias("dedup_basis"))
    )


def zero_fill_daily(daily: pl.DataFrame, date_min, date_max, categories: list[str]) -> pl.DataFrame:
    all_dates = pl.date_range(date_min, date_max, interval="1d", eager=True)
    grid = pl.DataFrame({"date": all_dates}).join(
        pl.DataFrame({"event_type_normalized": categories}), how="cross"
    )
    out = []
    for basis in [RAW_BASIS, DEDUP_BASIS]:
        sub = daily.filter(pl.col("dedup_basis") == basis).drop("dedup_basis")
        filled = grid.join(sub, on=["date", "event_type_normalized"], how="left").with_columns(
            pl.col("count").fill_null(0), pl.lit(basis).alias("dedup_basis")
        )
        out.append(filled)
    return pl.concat(out, how="vertical_relaxed").sort(["dedup_basis", "event_type_normalized", "date"])


def daily_to_weekly(daily_filled: pl.DataFrame) -> pl.DataFrame:
    with_week = daily_filled.with_columns(
        [
            pl.col("date").dt.iso_year().alias("iso_year"),
            pl.col("date").dt.week().alias("iso_week"),
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday() - 1)).alias("week_start_date"),
        ]
    )
    weekly = with_week.group_by(["dedup_basis", "event_type_normalized", "iso_year", "iso_week", "week_start_date"]).agg(
        pl.col("count").sum().alias("count")
    )
    return weekly.sort(["dedup_basis", "event_type_normalized", "iso_year", "iso_week"])


def main() -> None:
    records = pl.read_parquet(RECORDS_PATH)
    n_records_in = records.height

    daily_raw = build_raw_basis_daily(records)
    daily_dedup = build_dedup_basis_daily(records)
    daily_combined = pl.concat([daily_raw, daily_dedup], how="vertical_relaxed")

    # 保存則の確認：ゼロ埋め前の合計件数がレコード数と一致すること
    assert daily_raw.select(pl.col("count").sum()).item() == n_records_in
    n_groups = records.select(pl.col("possible_duplicate_group").n_unique()).item()
    assert daily_dedup.select(pl.col("count").sum()).item() == n_groups

    date_min, date_max = records.select(
        pl.col("event_date_parsed").min().alias("mn"), pl.col("event_date_parsed").max().alias("mx")
    ).row(0)
    categories = sorted(
        set(records.get_column("event_type_normalized").unique().to_list()) | {"mixed"}
    )

    daily_filled = zero_fill_daily(daily_combined, date_min, date_max, categories)

    # ゼロ埋め後も、非ゼロ件数の合計は変わらないこと（ゼロ埋めは行の追加のみ）
    assert daily_filled.filter(pl.col("dedup_basis") == RAW_BASIS).select(pl.col("count").sum()).item() == n_records_in
    assert daily_filled.filter(pl.col("dedup_basis") == DEDUP_BASIS).select(pl.col("count").sum()).item() == n_groups

    weekly = daily_to_weekly(daily_filled)
    assert weekly.filter(pl.col("dedup_basis") == RAW_BASIS).select(pl.col("count").sum()).item() == n_records_in
    assert weekly.filter(pl.col("dedup_basis") == DEDUP_BASIS).select(pl.col("count").sum()).item() == n_groups

    DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)
    daily_filled.write_parquet(DAILY_OUT)
    weekly.write_parquet(WEEKLY_OUT)

    print(f"records in: {n_records_in}, duplicate groups: {n_groups}")
    print(f"daily panel rows: {daily_filled.height} -> {DAILY_OUT}")
    print(f"weekly panel rows: {weekly.height} -> {WEEKLY_OUT}")


if __name__ == "__main__":
    main()
