"""P3図表で共有するユーティリティ（配色、日本語フォント設定、
重複集約ベースの代表レコード作成、1kmメッシュ丸め、祝日判定）。

配色はdatavizスキルの検証済みカテゴリカル順序（references/palette.md）
から、本データセットで使う上位5カテゴリ分をそのままの順序で使う。
新しい配色を作らず、検証済みの並びを部分列として使うだけなので
再検証は行っていない。
"""

from __future__ import annotations

import math
from pathlib import Path

import jpholiday
import matplotlib.pyplot as plt
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDS_PATH = PROJECT_ROOT / "data" / "interim" / "sapporo_records.parquet"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# --- 配色（dataviz skill references/palette.md のカテゴリカル順序、light mode） ---
CATEGORY_ORDER = ["sighting", "sighting_uncertain", "track_sign", "camera", "other"]
CATEGORY_LABEL_JA = {
    "sighting": "目撃",
    "sighting_uncertain": "目撃(未確定種)",
    "track_sign": "痕跡(フン/足跡等)",
    "camera": "カメラ確認",
    "other": "その他(捕獲/不明)",
}
CATEGORY_COLOR = {
    "sighting": "#2a78d6",  # slot1 blue
    "sighting_uncertain": "#eb6834",  # slot2 orange
    "track_sign": "#1baf7a",  # slot3 aqua
    "camera": "#eda100",  # slot4 yellow
    "other": "#e87ba4",  # slot5 magenta
}
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

RAW_BASIS = "raw_record"
DEDUP_BASIS = "dedup_group"
BASIS_LABEL_JA = {RAW_BASIS: "原記録ベース", DEDUP_BASIS: "重複候補集約ベース"}
# raw/dedupは「同じ実体の異なる集計」であって別カテゴリではないため、
# 色ではなくhatch（模様）で区別する。
BASIS_HATCH = {RAW_BASIS: "", DEDUP_BASIS: "////"}
BASIS_LINESTYLE = {RAW_BASIS: "-", DEDUP_BASIS: "--"}

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_LABEL_JA = {"Mon": "月", "Tue": "火", "Wed": "水", "Thu": "木", "Fri": "金", "Sat": "土", "Sun": "日"}


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = ["Noto Sans JP", "Yu Gothic", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = INK_MUTED
    plt.rcParams["axes.labelcolor"] = INK_SECONDARY
    plt.rcParams["xtick.color"] = INK_SECONDARY
    plt.rcParams["ytick.color"] = INK_SECONDARY
    plt.rcParams["text.color"] = INK_PRIMARY
    plt.rcParams["grid.color"] = GRIDLINE
    plt.rcParams["figure.facecolor"] = SURFACE
    plt.rcParams["axes.facecolor"] = SURFACE
    plt.rcParams["savefig.facecolor"] = SURFACE
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def collapse_category(series: pl.Series) -> pl.Series:
    """event_type_normalizedのうち稀な値(capture, unknown)を'other'に畳み込む。"""
    return series.map_elements(
        lambda v: v if v in ("sighting", "sighting_uncertain", "track_sign", "camera") else "other",
        return_dtype=pl.Utf8,
    )


def load_records() -> pl.DataFrame:
    df = pl.read_parquet(RECORDS_PATH)
    df = df.with_columns(collapse_category(pl.col("event_type_normalized")).alias("category"))
    return df


def build_dedup_representatives(records: pl.DataFrame) -> pl.DataFrame:
    """重複候補グループごとに1行を選ぶ（最も早い日付のレコードを代表とする）。

    代表カテゴリはグループ内で種別が揃っていればその種別、
    揃っていなければ'other'寄りに倒さず'mixed'として明示する
    （build_sapporo_panel.pyの日次集計と同じ規約）。
    """
    with_rank = records.with_columns(
        pl.col("event_date_parsed").rank(method="ordinal").over("possible_duplicate_group").alias("_rank_in_group")
    )
    representatives = with_rank.filter(pl.col("_rank_in_group") == 1).drop("_rank_in_group")

    type_counts = records.group_by("possible_duplicate_group").agg(pl.col("category").n_unique().alias("_n_types"))
    representatives = representatives.join(type_counts, on="possible_duplicate_group", how="left")
    representatives = representatives.with_columns(
        pl.when(pl.col("_n_types") == 1).then(pl.col("category")).otherwise(pl.lit("mixed")).alias("category")
    ).drop("_n_types")
    return representatives


def parse_hour(time_raw: str) -> int | None:
    if not time_raw:
        return None
    try:
        h = int(str(time_raw).split(":")[0])
        if 0 <= h <= 23:
            return h
    except (ValueError, IndexError):
        pass
    return None


def add_hour_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("event_time_raw").map_elements(parse_hour, return_dtype=pl.Int64).alias("hour")
    )


def is_holiday(d) -> bool:
    if d is None:
        return False
    return jpholiday.is_holiday(d) or d.weekday() >= 5  # 土日も休日側に含める


def round_to_mesh(lat: float, lon: float, mesh_km: float = 1.0) -> tuple[float, float] | tuple[None, None]:
    """緯度経度を概算1kmメッシュに丸める（記述用途の近似。厳密なJIS地域メッシュではない）。
    札幌付近(北緯43度)の経度1度は概ね81km。
    """
    if lat is None or lon is None:
        return None, None
    lat_step = mesh_km / 111.0
    lon_step = mesh_km / (111.0 * math.cos(math.radians(43.0)))
    lat_bin = math.floor(lat / lat_step) * lat_step + lat_step / 2
    lon_bin = math.floor(lon / lon_step) * lon_step + lon_step / 2
    return lat_bin, lon_bin


def savefig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path
