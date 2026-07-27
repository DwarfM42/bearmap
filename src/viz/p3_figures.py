"""P3: 札幌市データの記述統計・観測過程診断図を生成する。

原記録ベースと重複候補集約ベースの両方について、
年次件数・季節性(H4)・曜日周期(H6)・時刻分布・空間分布・
データ品質診断の図を作成する。日付フィールドの意味は未確定のため、
曜日周期の図・本文注記では「差の有無」のみを記述し、
原因（野外活動由来か行政処理由来か）には言及しない。

実行:
    uv run python src/viz/p3_figures.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from common import (
    BASIS_HATCH,
    BASIS_LABEL_JA,
    CATEGORY_COLOR,
    CATEGORY_LABEL_JA,
    CATEGORY_ORDER,
    DEDUP_BASIS,
    RAW_BASIS,
    SEQUENTIAL_BLUE,
    WEEKDAY_LABEL_JA,
    WEEKDAY_ORDER,
    add_hour_column,
    build_dedup_representatives,
    is_holiday,
    load_records,
    round_to_mesh,
    savefig,
    setup_matplotlib,
)

WEEKDAY_MAP = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


def add_calendar_columns(df: pl.DataFrame, date_col: str = "event_date_parsed") -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(date_col).dt.weekday().replace_strict(WEEKDAY_MAP).alias("weekday"),
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.iso_year().alias("iso_year"),
            pl.col(date_col).dt.week().alias("iso_week"),
        ]
    )


def year_color(years: list[int]) -> dict[int, str]:
    n = len(years)
    idx = np.linspace(0.15, 0.98, n)
    cmap = plt.get_cmap(BLUE_CMAP)
    return {y: cmap(idx[i]) for i, y in enumerate(sorted(years))}


# ---------------------------------------------------------------------------
# 1. 年次件数
# ---------------------------------------------------------------------------

def fig_annual_counts_by_type(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        pivot = (
            df.group_by(["_source_year", "category"]).len()
            .pivot(on="category", index="_source_year", values="len")
            .sort("_source_year")
            .fill_null(0)
        )
        years = pivot.get_column("_source_year").to_list()
        bottom = np.zeros(len(years))
        for cat in CATEGORY_ORDER:
            vals = np.array(pivot.get_column(cat).to_list()) if cat in pivot.columns else np.zeros(len(years))
            ax.bar(years, vals, bottom=bottom, color=CATEGORY_COLOR[cat], label=CATEGORY_LABEL_JA[cat], width=0.7)
            bottom += vals
        ax.set_title(BASIS_LABEL_JA[basis])
        ax.set_xlabel("年")
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
    axes[0].set_ylabel("件数")
    handles = [Patch(color=CATEGORY_COLOR[c], label=CATEGORY_LABEL_JA[c]) for c in CATEGORY_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle("年次件数（種別別、原記録 vs 重複候補集約）", y=1.14)
    savefig(fig, "p3_annual_counts_by_type")


def fig_annual_total_and_duprate(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    raw_by_year = records.group_by("_source_year").len().sort("_source_year").rename({"len": "raw_count"})
    dedup_by_year = dedup_reps.group_by("_source_year").len().sort("_source_year").rename({"len": "dedup_count"})
    merged = raw_by_year.join(dedup_by_year, on="_source_year", how="left").with_columns(
        (1 - pl.col("dedup_count") / pl.col("raw_count")).alias("dup_rate")
    )
    years = merged.get_column("_source_year").to_list()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(years))
    w = 0.35
    axes[0].bar(x - w / 2, merged.get_column("raw_count").to_list(), width=w, label=BASIS_LABEL_JA[RAW_BASIS], color="#2a78d6")
    axes[0].bar(x + w / 2, merged.get_column("dedup_count").to_list(), width=w, label=BASIS_LABEL_JA[DEDUP_BASIS], color="#2a78d6", hatch="////", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(years, rotation=45)
    axes[0].set_ylabel("件数")
    axes[0].set_title("年次総件数")
    axes[0].legend(frameon=False)

    axes[1].plot(years, merged.get_column("dup_rate").to_list(), marker="o", color="#eb6834")
    axes[1].set_xticks(years)
    axes[1].set_xticklabels(years, rotation=45)
    axes[1].set_ylabel("重複率 (1 - 集約件数/原記録件数)")
    axes[1].set_title("年次重複率（試作ルールv1による）")
    axes[1].set_ylim(0, max(0.6, merged.get_column("dup_rate").max() * 1.2))

    fig.suptitle("年次総件数と重複率")
    fig.tight_layout()
    savefig(fig, "p3_annual_total_and_duprate")


# ---------------------------------------------------------------------------
# 2. 季節性（H4）
# ---------------------------------------------------------------------------

def fig_seasonality_weekly_overlay(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        cal = add_calendar_columns(df)
        weekly = cal.group_by(["iso_year", "iso_week"]).len().sort(["iso_year", "iso_week"])
        years = sorted(weekly.get_column("iso_year").unique().to_list())
        colors = year_color(years)
        for y in years:
            sub = weekly.filter(pl.col("iso_year") == y).sort("iso_week")
            ax.plot(sub.get_column("iso_week").to_list(), sub.get_column("len").to_list(), color=colors[y], linewidth=1.3, marker=".", markersize=3, label=str(y))
        ax.set_title(BASIS_LABEL_JA[basis])
        ax.set_xlabel("ISO週番号")
    axes[0].set_ylabel("件数（未平滑化・生データ）")
    axes[1].legend(title="年", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("週別出没件数の年重ね描き（平滑化なし・生データ）")
    fig.tight_layout()
    savefig(fig, "p3_seasonality_weekly_overlay")


def fig_seasonality_weekly_by_type(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(CATEGORY_ORDER), figsize=(4 * len(CATEGORY_ORDER), 4), sharey=False)
    for ax, cat in zip(axes, CATEGORY_ORDER):
        for basis, df, ls in [(RAW_BASIS, records, "-"), (DEDUP_BASIS, dedup_reps, "--")]:
            cal = add_calendar_columns(df.filter(pl.col("category") == cat))
            weekly = cal.group_by("iso_week").len().sort("iso_week")
            ax.plot(weekly.get_column("iso_week").to_list(), weekly.get_column("len").to_list(), color=CATEGORY_COLOR[cat], linestyle=ls, linewidth=1.5, label=BASIS_LABEL_JA[basis])
        ax.set_title(CATEGORY_LABEL_JA[cat], fontsize=10)
        ax.set_xlabel("ISO週番号")
    axes[0].set_ylabel("件数（2017-2025年合計、未平滑化）")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("種別ごとの週別出没件数（全年合計、原記録=実線／重複集約=破線）")
    fig.tight_layout()
    savefig(fig, "p3_seasonality_weekly_by_type")


# ---------------------------------------------------------------------------
# 3. 曜日周期（H6）
# ---------------------------------------------------------------------------

def fig_weekday_counts(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(WEEKDAY_ORDER))
    w = 0.35
    for i, (basis, df) in enumerate([(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        cal = add_calendar_columns(df)
        counts = cal.group_by("weekday").len()
        vals = [counts.filter(pl.col("weekday") == wd).get_column("len").to_list()[0] if counts.filter(pl.col("weekday") == wd).height else 0 for wd in WEEKDAY_ORDER]
        offset = (i - 0.5) * w
        ax.bar(x + offset, vals, width=w, color="#2a78d6", hatch=BASIS_HATCH[basis], edgecolor="white" if basis == DEDUP_BASIS else "#2a78d6", label=BASIS_LABEL_JA[basis])
    ax.set_xticks(x)
    ax.set_xticklabels([WEEKDAY_LABEL_JA[d] for d in WEEKDAY_ORDER])
    ax.set_ylabel("件数（2017-2025年合計）")
    ax.set_title("曜日別出没件数（全期間合計）\n※曜日差の有無のみを報告。原因（野外活動 or 行政処理）は断定しない")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "p3_weekday_counts")


def fig_weekday_by_year_heatmap(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        cal = add_calendar_columns(df)
        pivot = cal.group_by(["_source_year", "weekday"]).len().pivot(on="weekday", index="_source_year", values="len").sort("_source_year").fill_null(0)
        years = pivot.get_column("_source_year").to_list()
        mat = np.array([[pivot.filter(pl.col("_source_year") == y).get_column(wd).to_list()[0] if wd in pivot.columns else 0 for wd in WEEKDAY_ORDER] for y in years])
        im = ax.imshow(mat, aspect="auto", cmap=BLUE_CMAP)
        ax.set_xticks(range(len(WEEKDAY_ORDER)))
        ax.set_xticklabels([WEEKDAY_LABEL_JA[d] for d in WEEKDAY_ORDER])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_title(BASIS_LABEL_JA[basis])
        fig.colorbar(im, ax=ax, label="件数", shrink=0.8)
    fig.suptitle("年×曜日 出没件数ヒートマップ")
    fig.tight_layout()
    savefig(fig, "p3_weekday_by_year_heatmap")


def fig_weekday_by_type(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        cal = add_calendar_columns(df)
        x = np.arange(len(WEEKDAY_ORDER))
        n_cat = len(CATEGORY_ORDER)
        w = 0.8 / n_cat
        for i, cat in enumerate(CATEGORY_ORDER):
            sub = cal.filter(pl.col("category") == cat).group_by("weekday").len()
            vals = [sub.filter(pl.col("weekday") == wd).get_column("len").to_list()[0] if sub.filter(pl.col("weekday") == wd).height else 0 for wd in WEEKDAY_ORDER]
            ax.bar(x + (i - n_cat / 2) * w + w / 2, vals, width=w, color=CATEGORY_COLOR[cat], label=CATEGORY_LABEL_JA[cat])
        ax.set_xticks(x)
        ax.set_xticklabels([WEEKDAY_LABEL_JA[d] for d in WEEKDAY_ORDER])
        ax.set_title(BASIS_LABEL_JA[basis])
    axes[0].set_ylabel("件数")
    handles = [Patch(color=CATEGORY_COLOR[c], label=CATEGORY_LABEL_JA[c]) for c in CATEGORY_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle("曜日別出没件数（種別別）", y=1.14)
    savefig(fig, "p3_weekday_by_type")


def fig_weekday_month_heatmap(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        cal = add_calendar_columns(df)
        pivot = cal.group_by(["month", "weekday"]).len().pivot(on="weekday", index="month", values="len").sort("month").fill_null(0)
        months = pivot.get_column("month").to_list()
        mat = np.array([[pivot.filter(pl.col("month") == m).get_column(wd).to_list()[0] if wd in pivot.columns else 0 for wd in WEEKDAY_ORDER] for m in months])
        im = ax.imshow(mat, aspect="auto", cmap=BLUE_CMAP)
        ax.set_xticks(range(len(WEEKDAY_ORDER)))
        ax.set_xticklabels([WEEKDAY_LABEL_JA[d] for d in WEEKDAY_ORDER])
        ax.set_yticks(range(len(months)))
        ax.set_yticklabels([f"{m}月" for m in months])
        ax.set_title(BASIS_LABEL_JA[basis])
        fig.colorbar(im, ax=ax, label="件数", shrink=0.8)
    fig.suptitle("月×曜日 出没件数ヒートマップ（2017-2025年合計）")
    fig.tight_layout()
    savefig(fig, "p3_weekday_month_heatmap")


def fig_weekday_holiday(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(2)
    w = 0.35
    for i, (basis, df) in enumerate([(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        date_min = df.get_column("event_date_parsed").min()
        date_max = df.get_column("event_date_parsed").max()
        all_dates = pl.date_range(date_min, date_max, interval="1d", eager=True)
        daily_counts = df.group_by("event_date_parsed").len().rename({"event_date_parsed": "date"})
        grid = pl.DataFrame({"date": all_dates}).join(daily_counts, on="date", how="left").with_columns(pl.col("len").fill_null(0))
        grid = grid.with_columns(pl.col("date").map_elements(is_holiday, return_dtype=pl.Boolean).alias("is_holiday"))
        means = grid.group_by("is_holiday").agg(pl.col("len").mean().alias("mean_count"))
        weekday_mean = means.filter(~pl.col("is_holiday")).get_column("mean_count").to_list()
        weekday_mean = weekday_mean[0] if weekday_mean else 0.0
        holiday_mean = means.filter(pl.col("is_holiday")).get_column("mean_count").to_list()
        holiday_mean = holiday_mean[0] if holiday_mean else 0.0
        offset = (i - 0.5) * w
        ax.bar(x + offset, [weekday_mean, holiday_mean], width=w, color="#2a78d6", hatch=BASIS_HATCH[basis], edgecolor="white" if basis == DEDUP_BASIS else "#2a78d6", label=BASIS_LABEL_JA[basis])
    ax.set_xticks(x)
    ax.set_xticklabels(["平日", "土日・祝日"])
    ax.set_ylabel("1日あたり平均件数（ゼロ埋め込みの日次件数の平均）")
    ax.set_title("平日 vs 土日・祝日の1日あたり平均件数\n※差の有無のみを報告し、原因は断定しない")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "p3_weekday_holiday")


# ---------------------------------------------------------------------------
# 4. 時刻分布
# ---------------------------------------------------------------------------

def fig_timeofday_distribution(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        withhour = add_hour_column(df).filter(pl.col("hour").is_not_null())
        for cat in CATEGORY_ORDER:
            sub = withhour.filter(pl.col("category") == cat)
            if sub.height == 0:
                continue
            counts = sub.group_by("hour").len().sort("hour")
            full = pl.DataFrame({"hour": list(range(24))}).join(counts, on="hour", how="left").with_columns(pl.col("len").fill_null(0))
            ax.plot(full.get_column("hour").to_list(), full.get_column("len").to_list(), color=CATEGORY_COLOR[cat], linewidth=1.5, marker=".", label=CATEGORY_LABEL_JA[cat])
        ax.set_title(BASIS_LABEL_JA[basis])
        ax.set_xlabel("時刻（時）")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("件数（時刻が記録された通報のみ）")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("時刻分布（種別別、時刻記録がある通報のみ）")
    fig.tight_layout()
    savefig(fig, "p3_timeofday_distribution")


def fig_timeofday_by_month_heatmap(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        withhour = add_hour_column(df).filter(pl.col("hour").is_not_null())
        cal = add_calendar_columns(withhour)
        pivot = cal.group_by(["month", "hour"]).len().pivot(on="hour", index="month", values="len").sort("month").fill_null(0)
        months = pivot.get_column("month").to_list()
        hour_cols = [h for h in range(24) if str(h) in pivot.columns or h in pivot.columns]
        mat = np.zeros((len(months), 24))
        for mi, m in enumerate(months):
            row = pivot.filter(pl.col("month") == m)
            for h in range(24):
                col = h if h in pivot.columns else str(h)
                if col in pivot.columns:
                    v = row.get_column(col).to_list()
                    mat[mi, h] = v[0] if v else 0
        im = ax.imshow(mat, aspect="auto", cmap=BLUE_CMAP)
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(len(months)))
        ax.set_yticklabels([f"{m}月" for m in months])
        ax.set_title(BASIS_LABEL_JA[basis])
        ax.set_xlabel("時刻（時）")
        fig.colorbar(im, ax=ax, label="件数", shrink=0.8)
    fig.suptitle("月×時刻 出没件数ヒートマップ（時刻記録がある通報のみ）")
    fig.tight_layout()
    savefig(fig, "p3_timeofday_by_month_heatmap")


def fig_timeofday_missing_rate(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for basis, df, ls in [(RAW_BASIS, records, "-"), (DEDUP_BASIS, dedup_reps, "--")]:
        withhour = add_hour_column(df)
        rate = (
            withhour.group_by("_source_year")
            .agg((pl.col("hour").is_null().sum() / pl.len()).alias("missing_rate"))
            .sort("_source_year")
        )
        ax.plot(rate.get_column("_source_year").to_list(), rate.get_column("missing_rate").to_list(), marker="o", linestyle=ls, color="#2a78d6", label=BASIS_LABEL_JA[basis])
    ax.set_xlabel("年")
    ax.set_ylabel("時刻欠損率")
    ax.set_ylim(0, 1)
    ax.set_title("年別の時刻欠損率")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "p3_timeofday_missing_rate")


# ---------------------------------------------------------------------------
# 5. 空間記述（1kmメッシュ、詳細座標は出さない）
# ---------------------------------------------------------------------------

def _mesh_grid(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = df.filter(pl.col("location_precision") == "point")
    lat_lon = [round_to_mesh(r["lat"], r["lon"]) for r in valid.select(["lat", "lon"]).to_dicts()]
    lat_bins = [x[0] for x in lat_lon]
    lon_bins = [x[1] for x in lat_lon]
    mesh_df = pl.DataFrame({"lat_bin": lat_bins, "lon_bin": lon_bins}).group_by(["lat_bin", "lon_bin"]).len()
    lat_unique = sorted(mesh_df.get_column("lat_bin").unique().to_list())
    lon_unique = sorted(mesh_df.get_column("lon_bin").unique().to_list())
    mat = np.zeros((len(lat_unique), len(lon_unique)))
    lat_idx = {v: i for i, v in enumerate(lat_unique)}
    lon_idx = {v: i for i, v in enumerate(lon_unique)}
    for row in mesh_df.iter_rows(named=True):
        mat[lat_idx[row["lat_bin"]], lon_idx[row["lon_bin"]]] = row["len"]
    return mat, np.array(lat_unique), np.array(lon_unique)


def fig_spatial_mesh_by_type(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(CATEGORY_ORDER), figsize=(4 * len(CATEGORY_ORDER), 8))
    for row_i, (basis, df) in enumerate([(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        for col_i, cat in enumerate(CATEGORY_ORDER):
            ax = axes[row_i, col_i]
            sub = df.filter(pl.col("category") == cat)
            if sub.filter(pl.col("location_precision") == "point").height == 0:
                ax.axis("off")
                continue
            mat, lat_u, lon_u = _mesh_grid(sub)
            ax.imshow(mat, cmap=BLUE_CMAP, origin="lower", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(CATEGORY_LABEL_JA[cat], fontsize=10)
            if col_i == 0:
                ax.set_ylabel(BASIS_LABEL_JA[basis], fontsize=10)
    fig.suptitle("1kmメッシュ出没件数（種別別、2017-2025年合計、座標は概算メッシュに丸め済み）")
    fig.tight_layout()
    savefig(fig, "p3_spatial_mesh_by_type")


def fig_spatial_mesh_yearly(records: pl.DataFrame, dedup_reps: pl.DataFrame, basis: str, df: pl.DataFrame) -> None:
    years = sorted(df.get_column("_source_year").unique().to_list())
    ncols = 3
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).reshape(-1)
    for i, y in enumerate(years):
        ax = axes_flat[i]
        sub = df.filter(pl.col("_source_year") == y)
        if sub.filter(pl.col("location_precision") == "point").height == 0:
            ax.axis("off")
            continue
        mat, lat_u, lon_u = _mesh_grid(sub)
        ax.imshow(mat, cmap=BLUE_CMAP, origin="lower", aspect="auto")
        ax.set_title(str(y), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(years), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(f"1kmメッシュ出没件数の年別比較（全種別合計、{BASIS_LABEL_JA[basis]}、座標は概算メッシュに丸め済み）")
    fig.tight_layout()
    savefig(fig, f"p3_spatial_mesh_yearly_{basis}")


# ---------------------------------------------------------------------------
# 6. データ品質診断
# ---------------------------------------------------------------------------

def fig_quality_rates_by_year(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) missing-location rate
    ax = axes[0, 0]
    r = records.group_by("_source_year").agg((pl.col("location_precision") == "missing").mean().alias("v")).sort("_source_year")
    ax.plot(r.get_column("_source_year").to_list(), r.get_column("v").to_list(), marker="o", color="#2a78d6")
    ax.set_title("座標欠損率")
    ax.set_ylim(0, max(0.05, r.get_column("v").max() * 1.5))

    # (b) missing-time rate
    ax = axes[0, 1]
    withhour = add_hour_column(records)
    r = withhour.group_by("_source_year").agg(pl.col("hour").is_null().mean().alias("v")).sort("_source_year")
    ax.plot(r.get_column("_source_year").to_list(), r.get_column("v").to_list(), marker="o", color="#eb6834")
    ax.set_title("時刻欠損率")
    ax.set_ylim(0, 1)

    # (c) duplicate rate
    ax = axes[1, 0]
    raw_by_year = records.group_by("_source_year").len().sort("_source_year").rename({"len": "raw_count"})
    dedup_by_year = dedup_reps.group_by("_source_year").len().sort("_source_year").rename({"len": "dedup_count"})
    merged = raw_by_year.join(dedup_by_year, on="_source_year", how="left").with_columns((1 - pl.col("dedup_count") / pl.col("raw_count")).alias("v"))
    ax.plot(merged.get_column("_source_year").to_list(), merged.get_column("v").to_list(), marker="o", color="#1baf7a")
    ax.set_title("重複率（試作ルールv1）")
    ax.set_ylim(0, max(0.5, merged.get_column("v").max() * 1.2))

    # (d) mean duplicate-group size
    ax = axes[1, 1]
    gsize = records.group_by(["_source_year", "possible_duplicate_group"]).len().group_by("_source_year").agg(pl.col("len").mean().alias("v")).sort("_source_year")
    ax.plot(gsize.get_column("_source_year").to_list(), gsize.get_column("v").to_list(), marker="o", color="#eda100")
    ax.set_title("重複候補グループの平均サイズ（記録数/グループ）")

    for ax in axes.reshape(-1):
        ax.set_xlabel("年")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("年別データ品質診断（原記録ベース）")
    fig.tight_layout()
    savefig(fig, "p3_quality_rates_by_year")


def fig_quality_category_composition(records: pl.DataFrame, dedup_reps: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (basis, df) in zip(axes, [(RAW_BASIS, records), (DEDUP_BASIS, dedup_reps)]):
        pivot = df.group_by(["_source_year", "category"]).len().pivot(on="category", index="_source_year", values="len").sort("_source_year").fill_null(0)
        years = pivot.get_column("_source_year").to_list()
        totals = np.zeros(len(years))
        for cat in CATEGORY_ORDER:
            vals = np.array(pivot.get_column(cat).to_list()) if cat in pivot.columns else np.zeros(len(years))
            totals += vals
        bottom = np.zeros(len(years))
        for cat in CATEGORY_ORDER:
            vals = np.array(pivot.get_column(cat).to_list()) if cat in pivot.columns else np.zeros(len(years))
            share = np.divide(vals, totals, out=np.zeros_like(vals, dtype=float), where=totals > 0)
            ax.bar(years, share, bottom=bottom, color=CATEGORY_COLOR[cat], width=0.7, label=CATEGORY_LABEL_JA[cat])
            bottom += share
        ax.set_title(BASIS_LABEL_JA[basis])
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
    axes[0].set_ylabel("構成比")
    handles = [Patch(color=CATEGORY_COLOR[c], label=CATEGORY_LABEL_JA[c]) for c in CATEGORY_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle("年別の種別構成比（100%積み上げ）", y=1.14)
    savefig(fig, "p3_quality_category_composition")


def _decimal_digits(x: float | None) -> int | None:
    if x is None:
        return None
    s = f"{x:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def fig_quality_coordinate_precision(records: pl.DataFrame) -> None:
    valid = records.filter(pl.col("location_precision") == "point")
    valid = valid.with_columns(
        pl.col("lat").map_elements(_decimal_digits, return_dtype=pl.Int64).alias("lat_digits")
    )
    bins = [2, 4, 6, 8]
    bin_labels = ["0-2桁", "3-4桁", "5-6桁", "7-8桁", "9桁以上"]
    valid = valid.with_columns(
        pl.col("lat_digits").cut(bins, labels=bin_labels).alias("digit_bin")
    )
    pivot = valid.group_by(["_source_year", "digit_bin"]).len().pivot(on="digit_bin", index="_source_year", values="len").sort("_source_year").fill_null(0)
    years = pivot.get_column("_source_year").to_list()
    present_bins = [b for b in bin_labels if b in pivot.columns]
    mat = np.array([[pivot.filter(pl.col("_source_year") == y).get_column(b).to_list()[0] for b in present_bins] for y in years])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap=BLUE_CMAP, aspect="auto")
    ax.set_xticks(range(len(present_bins)))
    ax.set_xticklabels(present_bins)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("緯度の小数桁数（座標入力精度の目安）")
    ax.set_ylabel("年")
    ax.set_title("年別の座標小数桁数分布\n（記録方式変更を疑わせる断層の検出用）")
    fig.colorbar(im, ax=ax, label="件数")
    fig.tight_layout()
    savefig(fig, "p3_quality_coordinate_precision")


def main() -> None:
    setup_matplotlib()
    records = load_records()
    dedup_reps = build_dedup_representatives(records)

    fig_annual_counts_by_type(records, dedup_reps)
    fig_annual_total_and_duprate(records, dedup_reps)

    fig_seasonality_weekly_overlay(records, dedup_reps)
    fig_seasonality_weekly_by_type(records, dedup_reps)

    fig_weekday_counts(records, dedup_reps)
    fig_weekday_by_year_heatmap(records, dedup_reps)
    fig_weekday_by_type(records, dedup_reps)
    fig_weekday_month_heatmap(records, dedup_reps)
    fig_weekday_holiday(records, dedup_reps)

    fig_timeofday_distribution(records, dedup_reps)
    fig_timeofday_by_month_heatmap(records, dedup_reps)
    fig_timeofday_missing_rate(records, dedup_reps)

    fig_spatial_mesh_by_type(records, dedup_reps)
    fig_spatial_mesh_yearly(records, dedup_reps, RAW_BASIS, records)
    fig_spatial_mesh_yearly(records, dedup_reps, DEDUP_BASIS, dedup_reps)

    fig_quality_rates_by_year(records, dedup_reps)
    fig_quality_category_composition(records, dedup_reps)
    fig_quality_coordinate_precision(records)

    print("all P3 figures generated.")


if __name__ == "__main__":
    main()
