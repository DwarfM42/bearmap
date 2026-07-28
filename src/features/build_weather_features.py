"""P5a-2: 気象庁「過去の気象データ検索」の生HTML(data/raw/weather_sapporo/)から
日別値（気温・降水量・積雪深）を抽出し、年別の季節指標を算出する。

列位置は2020年1月・4月分のHTMLを実際に目視確認して特定した
（表ヘッダが複雑な多段構成のため、位置による対応付けを値の妥当性で検証済み）：
  0=日, 3=降水量合計(mm), 6=気温平均(℃), 7=気温最高, 8=気温最低,
  17=降雪合計(cm), 18=最深積雪(cm)

融雪日の定義（事前固定）：
  「積雪深が0cmになり、かつその後14日間0cmが続く（根雪明け後の再降雪でリセットされない）
  最初の日」とする。この定義に加え、感度分析として
  「単純に最初に0cmを記録した日」（再降雪を許容しない、より早い定義）も併記する。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather_sapporo"
OUT_DAILY = PROJECT_ROOT / "data" / "interim" / "sapporo_weather_daily.parquet"
OUT_ANNUAL = PROJECT_ROOT / "data" / "processed" / "sapporo_weather_annual.parquet"

COL_PRECIP = 3
COL_TEMP_MEAN = 6
COL_TEMP_MAX = 7
COL_TEMP_MIN = 8
COL_SNOWFALL = 17
COL_SNOWDEPTH = 18


def _to_float(s: str) -> float | None:
    s = s.strip()
    if s in ("", "--", "×", "///"):
        return None
    s = s.replace(")", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_month_file(path: Path) -> list[dict]:
    year, month = path.stem.split("-")
    year, month = int(year), int(month)
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    table = soup.find("table", id="tablefix1")
    if table is None:
        return []
    rows = table.find_all("tr")
    out = []
    for r in rows[4:]:  # 最初の4行はヘッダ
        cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
        if len(cells) < 19 or not cells[0].isdigit():
            continue
        day = int(cells[0])
        out.append(
            {
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "precip_mm": _to_float(cells[COL_PRECIP]),
                "temp_mean_c": _to_float(cells[COL_TEMP_MEAN]),
                "temp_max_c": _to_float(cells[COL_TEMP_MAX]),
                "temp_min_c": _to_float(cells[COL_TEMP_MIN]),
                "snowfall_cm": _to_float(cells[COL_SNOWFALL]),
                "snowdepth_cm": _to_float(cells[COL_SNOWDEPTH]),
            }
        )
    return out


def compute_snowmelt_day(daily_year: pl.DataFrame, persist_days: int = 14) -> dict:
    """指定した1年分(1月〜7月を対象、根雪明け探索のため)の日別データから融雪日を求める。
    persist_days日間0cmが続く最初の0cmの日を「融雪日」とする（主定義）。
    感度分析として、単純に最初に0cmを記録した日も返す。

    重要な注記：気象庁の「最深積雪」列は、積雪が無い日は数値0ではなく「--」（欠測扱いの記号）
    で表記される。これは真の欠測ではなく「積雪なし」を意味すると判断し、本関数内でのみ
    Noneを0cmとして扱う（3月以降、実測値が並ぶ中で生じるNoneに限定。真の機器欠測との
    区別はできないため、この仮定を明示する）。
    """
    sub = daily_year.filter(pl.col("date").dt.month() <= 7).sort("date")
    depths = sub.select(["date", "snowdepth_cm"]).to_dicts()
    naive_zero_day = None
    persistent_zero_day = None
    for i, row in enumerate(depths):
        d = row["snowdepth_cm"]
        d_eff = 0.0 if d is None else d  # 「--」=積雪なし=0cmとみなす（注記参照）
        if d_eff <= 0 and naive_zero_day is None:
            naive_zero_day = row["date"]
        if d_eff <= 0 and persistent_zero_day is None:
            window = depths[i : i + persist_days]
            if len(window) == persist_days and all(
                (0.0 if w["snowdepth_cm"] is None else w["snowdepth_cm"]) <= 0 for w in window
            ):
                persistent_zero_day = row["date"]
    return {"snowmelt_day_persistent14": persistent_zero_day, "snowmelt_day_naive_first_zero": naive_zero_day}


def main() -> None:
    all_rows = []
    for path in sorted(RAW_DIR.glob("*.html")):
        all_rows.extend(parse_month_file(path))
    daily = pl.DataFrame(all_rows).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    daily = daily.sort("date")
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(OUT_DAILY)
    print(f"daily rows: {daily.height}, written to {OUT_DAILY}")

    years = sorted(daily.get_column("date").dt.year().unique().to_list())
    annual_rows = []
    for y in years:
        year_data = daily.filter(pl.col("date").dt.year() == y)
        melt = compute_snowmelt_day(year_data)

        def season_stats(months: list[int], prefix: str) -> dict:
            sub = year_data.filter(pl.col("date").dt.month().is_in(months))
            return {
                f"{prefix}_temp_mean_c": sub.get_column("temp_mean_c").mean(),
                f"{prefix}_precip_total_mm": sub.get_column("precip_mm").sum(),
                f"{prefix}_snowfall_total_cm": sub.get_column("snowfall_cm").sum() if prefix == "spring" else None,
                f"{prefix}_max_snowdepth_cm": sub.get_column("snowdepth_cm").max() if prefix == "spring" else None,
            }

        row = {"year": y}
        row.update(melt)
        row.update(season_stats([3, 4, 5], "spring"))
        row.update(season_stats([6, 7, 8], "summer"))
        row.update(season_stats([9, 10, 11], "autumn"))
        annual_rows.append(row)

    annual = pl.DataFrame(annual_rows, infer_schema_length=None)
    OUT_ANNUAL.parent.mkdir(parents=True, exist_ok=True)
    annual.write_parquet(OUT_ANNUAL)
    print(f"annual rows: {annual.height}, written to {OUT_ANNUAL}")
    print(annual)


if __name__ == "__main__":
    main()
