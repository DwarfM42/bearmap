"""P5b/P5c: H3(秋季食物資源)・H5(融雪日と春ピーク)で使うアウトカム変数を、
原記録ベース・重複集約ベース(基準v1)の両方について年別に算出する。

同数ピーク・件数僅少・ピーク不明瞭年の扱い（事前固定）：
- 春季(3-5月)の合計件数が5件未満の年は「件数僅少」としてフラグを立て、
  ピーク週・重心・50%到達日を参考値としつつ、解釈上は「不明瞭」として注記する。
- 週別件数が同数で並ぶ場合は、シーズン内で最も早い週を採用する（恣意的だが
  事前固定した規約として一貫して適用する）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from common import load_records, build_dedup_representatives  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "h3_h5_outcomes_annual.parquet"

AUTUMN_MONTHS = [9, 10, 11]
SPRING_MONTHS = [3, 4, 5]
SPARSE_THRESHOLD = 5


def compute_for_basis(df: pl.DataFrame, basis_name: str) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("event_date_parsed").dt.year().alias("year"),
            pl.col("event_date_parsed").dt.month().alias("month"),
            pl.col("event_date_parsed").dt.week().alias("iso_week"),
            pl.col("event_date_parsed").dt.ordinal_day().alias("doy"),
        ]
    )
    years = sorted(df.get_column("year").unique().to_list())
    rows = []
    for y in years:
        year_df = df.filter(pl.col("year") == y)
        annual_total = year_df.height

        autumn_df = year_df.filter(pl.col("month").is_in(AUTUMN_MONTHS))
        autumn_count = autumn_df.height
        autumn_share = autumn_count / annual_total if annual_total > 0 else None
        if autumn_count > 0:
            weekly = autumn_df.group_by("iso_week").len().sort(["len", "iso_week"], descending=[True, False])
            autumn_peak_week = weekly.row(0)[0]
            autumn_peak_value = weekly.row(0)[1]
            centroid = (autumn_df.get_column("doy") * 1.0).mean()
        else:
            autumn_peak_week, autumn_peak_value, centroid = None, 0, None

        spring_df = year_df.filter(pl.col("month").is_in(SPRING_MONTHS))
        spring_count = spring_df.height
        spring_sparse = spring_count < SPARSE_THRESHOLD
        if spring_count > 0:
            sp_weekly = spring_df.group_by("iso_week").len().sort(["len", "iso_week"], descending=[True, False])
            spring_peak_week = sp_weekly.row(0)[0]
            spring_centroid_doy = (spring_df.get_column("doy") * 1.0).mean()
            sp_sorted = spring_df.sort("doy")
            cum = 0
            half = spring_count / 2.0
            cum50_doy = None
            for r in sp_sorted.iter_rows(named=True):
                cum += 1
                if cum >= half and cum50_doy is None:
                    cum50_doy = r["doy"]
        else:
            spring_peak_week, spring_centroid_doy, cum50_doy = None, None, None

        rows.append(
            {
                "basis": basis_name,
                "year": y,
                "annual_total": annual_total,
                "autumn_count": autumn_count,
                "autumn_share_of_annual": autumn_share,
                "autumn_peak_week": autumn_peak_week,
                "autumn_peak_value": autumn_peak_value,
                "autumn_centroid_doy": centroid,
                "spring_count": spring_count,
                "spring_sparse_flag": spring_sparse,
                "spring_peak_week": spring_peak_week,
                "spring_centroid_doy": spring_centroid_doy,
                "spring_cum50_doy": cum50_doy,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    records = load_records()
    dedup_reps = build_dedup_representatives(records)

    raw_out = compute_for_basis(records, "raw_record")
    dedup_out = compute_for_basis(dedup_reps, "dedup_group_v1")
    combined = pl.concat([raw_out, dedup_out], how="vertical_relaxed")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUT_PATH)
    print(f"written {combined.height} rows to {OUT_PATH}")
    print(combined)


if __name__ == "__main__":
    main()
