"""P4a-3: 2020年（時刻欠損急増）と2025年（総件数・重複率急増）の断層診断。

出没データのみを用いた記述的な深掘り。原因の断定は行わない
（原因不明の場合は原因不明のまま記録する）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from common import add_hour_column, build_dedup_representatives, load_records  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[2] / "reports" / "p4a_discontinuity_diagnosis.md"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sapporo_bear"


def diagnose_2020(records: pl.DataFrame) -> list[str]:
    lines = ["## 2020年：時刻欠損急増の診断", ""]
    y2020 = records.filter(pl.col("_source_year") == 2020)
    y2020h = add_hour_column(y2020)

    lines.append(f"- 2020年の総件数: {y2020.height}件")
    lines.append(f"- うち時刻欠損: {y2020h.filter(pl.col('hour').is_null()).height}件 "
                 f"({y2020h.filter(pl.col('hour').is_null()).height / y2020.height:.1%})")

    lines.append("\n### 月別の時刻欠損率（2020年）")
    monthly = y2020h.with_columns(pl.col("event_date_parsed").dt.month().alias("month")).group_by("month").agg(
        pl.len().alias("n"), pl.col("hour").is_null().mean().alias("missing_rate")
    ).sort("month")
    for row in monthly.iter_rows(named=True):
        lines.append(f"- {row['month']}月: n={row['n']}, 時刻欠損率={row['missing_rate']:.1%}")

    lines.append("\n### 種別ごとの時刻欠損率（2020年）")
    by_type = y2020h.group_by("event_type_normalized").agg(
        pl.len().alias("n"), pl.col("hour").is_null().mean().alias("missing_rate")
    ).sort("n", descending=True)
    for row in by_type.iter_rows(named=True):
        lines.append(f"- {row['event_type_normalized']}: n={row['n']}, 時刻欠損率={row['missing_rate']:.1%}")

    lines.append("\n### 時刻欠損レコードの時刻欄の生値（先頭20件）")
    missing_raw = y2020h.filter(pl.col("hour").is_null()).select(["event_time_raw", "event_type_original"]).head(20)
    for row in missing_raw.iter_rows(named=True):
        lines.append(f"- time_raw={row['event_time_raw']!r}, status={row['event_type_original']!r}")

    lines.append(
        "\n**所見**: 2020年の時刻欠損は特定の月・種別に強く偏っているか、"
        "生値を見て空文字なのか異常値なのかを上記から確認できる。"
        "自治体側の記録様式変更に関する公表資料は確認できておらず、原因は特定できない（原因不明のまま記録）。"
    )
    return lines


def diagnose_2025(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> list[str]:
    lines = ["\n## 2025年：総件数・重複率急増の診断", ""]
    y2025 = records.filter(pl.col("_source_year") == 2025)
    y2025_dedup = dedup_reps.filter(pl.col("_source_year") == 2025)

    lines.append(f"- 2025年の総件数（原記録）: {y2025.height}件、重複集約後: {y2025_dedup.height}件")
    lines.append(f"- 重複率: {1 - y2025_dedup.height / y2025.height:.1%}")

    lines.append("\n### 月別件数・重複率（2025年）")
    monthly_raw = y2025.with_columns(pl.col("event_date_parsed").dt.month().alias("month")).group_by("month").agg(
        pl.len().alias("n_raw")
    )
    monthly_dedup = y2025_dedup.with_columns(pl.col("event_date_parsed").dt.month().alias("month")).group_by("month").agg(
        pl.len().alias("n_dedup")
    )
    monthly = monthly_raw.join(monthly_dedup, on="month", how="left").with_columns(
        pl.col("n_dedup").fill_null(0), (1 - pl.col("n_dedup") / pl.col("n_raw")).alias("dup_rate")
    ).sort("month")
    for row in monthly.iter_rows(named=True):
        lines.append(f"- {row['month']}月: 原記録={row['n_raw']}, 重複集約後={row['n_dedup']}, 重複率={row['dup_rate']:.1%}")

    lines.append("\n### 種別ごとの件数（2025年 vs 2017-2024年平均）")
    y_other = records.filter(pl.col("_source_year") != 2025)
    other_years_n = y_other.get_column("_source_year").n_unique()
    by_type_2025 = y2025.group_by("event_type_normalized").agg(pl.len().alias("n_2025"))
    by_type_other = y_other.group_by("event_type_normalized").agg(
        (pl.len() / other_years_n).alias("n_avg_other_years")
    )
    merged = by_type_2025.join(by_type_other, on="event_type_normalized", how="outer_coalesce").fill_null(0).sort(
        "n_2025", descending=True
    )
    for row in merged.iter_rows(named=True):
        lines.append(f"- {row['event_type_normalized']}: 2025年={row['n_2025']}, 他年平均={row['n_avg_other_years']:.1f}")

    lines.append("\n### 地域（区）別件数（2025年 vs 他年平均）")
    by_ward_2025 = y2025.group_by("ward").agg(pl.len().alias("n_2025"))
    by_ward_other = y_other.group_by("ward").agg((pl.len() / other_years_n).alias("n_avg_other_years"))
    merged_ward = by_ward_2025.join(by_ward_other, on="ward", how="outer_coalesce").fill_null(0).sort(
        "n_2025", descending=True
    )
    for row in merged_ward.iter_rows(named=True):
        lines.append(f"- {row['ward']}: 2025年={row['n_2025']}, 他年平均={row['n_avg_other_years']:.1f}")

    lines.append("\n### 特定の重複グループが増加分の大半を占めていないか")
    group_sizes = y2025.group_by("possible_duplicate_group").len().sort("len", descending=True)
    top10 = group_sizes.head(10)
    top10_sum = top10.get_column("len").sum()
    lines.append(f"- 2025年の重複グループ数: {group_sizes.height}、上位10グループの合計記録数: {top10_sum} "
                 f"（2025年原記録全体{y2025.height}件中 {top10_sum / y2025.height:.1%}）")
    for row in top10.iter_rows(named=True):
        lines.append(f"  - グループ{row['possible_duplicate_group']}: {row['len']}件")

    lines.append("\n### 座標精度・時刻記録率（2025年 vs 他年）")
    y2025h = add_hour_column(y2025)
    lines.append(f"- 2025年 時刻欠損率: {y2025h.filter(pl.col('hour').is_null()).height / y2025.height:.1%}")
    other_h = add_hour_column(y_other)
    lines.append(f"- 他年平均 時刻欠損率: {other_h.filter(pl.col('hour').is_null()).height / y_other.height:.1%}")
    lines.append(f"- 2025年 座標欠損: {y2025.filter(pl.col('location_precision') == 'missing').height}件")

    lines.append(
        "\n**所見**: 2025年は特定の月（記述参照）・種別・地域に偏って増加しているか、"
        "上位重複グループが増加分の大半を占めるかを確認した。"
        "公開ページ・自治体資料で2025年特有の記録方式変更・注意喚起の記載は本調査の範囲では確認できておらず、"
        "「実際の出没増加」と「通報・記録運用の変化」を本記述統計だけで分離することはできない（原因不明のまま記録）。"
    )
    return lines


def check_csv_column_structure() -> list[str]:
    lines = ["\n## CSV列構成・カテゴリ表記の変更確認", ""]
    for p in sorted(RAW_DIR.glob("*.csv")):
        with open(p, encoding="utf-8-sig") as f:
            header = f.readline().strip()
        lines.append(f"- {p.name}: {header}")
    return lines


def main() -> None:
    records = load_records()
    dedup_reps = build_dedup_representatives(records)

    lines = ["# P4a-3: 2020年・2025年の断層診断", ""]
    lines.extend(diagnose_2020(records))
    lines.extend(diagnose_2025(records, dedup_reps))
    lines.extend(check_csv_column_structure())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
