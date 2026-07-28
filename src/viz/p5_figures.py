"""P5b/P5c: H3(豊凶×秋季出没)・H5(融雪日×春季重心)の散布図。
nが小さいため（H3: n=6, H5: n=9）、単純な散布図で年ラベル・2025年の強調表示を行い、
小標本であることが視覚的にも分かるようにする。
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_matplotlib, savefig  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAST_PATH = PROJECT_ROOT / "data" / "processed" / "mast_ishikari_annual.csv"
OUTCOMES_PATH = PROJECT_ROOT / "data" / "processed" / "h3_h5_outcomes_annual.parquet"
WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_weather_annual.parquet"


def fig_h3_scatter() -> None:
    mast = pl.read_csv(MAST_PATH).filter(pl.col("mizunara_rating_code").is_not_null())
    outcomes = pl.read_parquet(OUTCOMES_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, basis in zip(axes, ["raw_record", "dedup_group_v1"]):
        sub = outcomes.filter(pl.col("basis") == basis)
        j = mast.join(sub, on="year", how="inner").sort("year")
        colors = ["#e34948" if y == 2025 else "#2a78d6" for y in j.get_column("year").to_list()]
        ax.scatter(j.get_column("mizunara_rating_code"), j.get_column("autumn_count"), c=colors, s=80, zorder=3)
        for row in j.iter_rows(named=True):
            ax.annotate(str(row["year"]), (row["mizunara_rating_code"], row["autumn_count"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("ミズナラ豊凶評価(石狩振興局、0=凶作〜3=豊作、目視読み取り)")
        ax.set_ylabel("秋季出没件数")
        ax.set_title(basis)
        ax.set_xticks([0, 1, 2, 3])
    fig.suptitle("H3: ミズナラ豊凶(石狩振興局)と秋季出没件数（赤=2025年、n=6）")
    fig.tight_layout()
    savefig(fig, "p5_h3_mast_vs_autumn_scatter")


def fig_h5_scatter() -> None:
    weather = pl.read_parquet(WEATHER_PATH)
    outcomes = pl.read_parquet(OUTCOMES_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, basis in zip(axes, ["raw_record", "dedup_group_v1"]):
        sub = outcomes.filter(pl.col("basis") == basis)
        j = weather.join(sub, on="year", how="inner").sort("year")
        j = j.with_columns(pl.col("snowmelt_day_persistent14").dt.ordinal_day().alias("melt_doy"))
        colors = ["#e34948" if y == 2025 else "#2a78d6" for y in j.get_column("year").to_list()]
        ax.scatter(j.get_column("melt_doy"), j.get_column("spring_centroid_doy"), c=colors, s=80, zorder=3)
        for row in j.iter_rows(named=True):
            ax.annotate(str(row["year"]), (row["melt_doy"], row["spring_centroid_doy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("融雪日(持続14日定義、年間通日)")
        ax.set_ylabel("春季件数の時間重心(年間通日)")
        ax.set_title(basis)
    fig.suptitle("H5: 融雪日と春季出没の時間重心（赤=2025年、n=9）")
    fig.tight_layout()
    savefig(fig, "p5_h5_snowmelt_vs_spring_centroid_scatter")


def main() -> None:
    setup_matplotlib()
    fig_h3_scatter()
    fig_h5_scatter()


if __name__ == "__main__":
    main()
