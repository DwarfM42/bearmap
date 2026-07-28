"""P5b(H3)・P5c(H5)の記述的比較・順位相関・leave-one-year-out頑健性チェック。

n<7(H3, n=6)は統計モデル・有意差検定を行わず記述的比較のみ。
n=7〜10(H5, n=9)は単変量の探索的関連のみで多変量回帰は行わない。
（P5_MODEL_PLAN相当の基準はユーザー指示のP5a節にて事前固定）
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports"

MAST_PATH = PROJECT_ROOT / "data" / "processed" / "mast_ishikari_annual.csv"
OUTCOMES_PATH = PROJECT_ROOT / "data" / "processed" / "h3_h5_outcomes_annual.parquet"
WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_weather_annual.parquet"


def spearman_with_loo(x: list[float], y: list[float], years: list[int]) -> dict:
    rho, p = sstats.spearmanr(x, y)
    loo = {}
    n = len(x)
    for i in range(n):
        xi = x[:i] + x[i + 1 :]
        yi = y[:i] + y[i + 1 :]
        if len(set(xi)) < 2 or len(set(yi)) < 2:
            loo[years[i]] = None
            continue
        r, _ = sstats.spearmanr(xi, yi)
        loo[years[i]] = r
    return {"rho": rho, "p_value": p, "n": n, "loo_rho_excluding_year": loo}


def h3_analysis() -> list[str]:
    lines = ["## H3: 秋季食物資源(ミズナラ、石狩振興局)と秋季出没の対応", ""]
    mast = pl.read_csv(MAST_PATH).filter(pl.col("mizunara_rating_code").is_not_null())
    outcomes = pl.read_parquet(OUTCOMES_PATH)

    lines.append(f"**n = {mast.height}年（{sorted(mast.get_column('year').to_list())}）。"
                 f"n<7のため統計モデル・有意差検定は行わず、記述的比較・順位相関(参考値)にとどめる。**")
    lines.append("")
    lines.append("石狩振興局のミズナラ豊凶評価（目視読み取り、中程度の信頼度。詳細は`data/processed/mast_ishikari_annual.csv`参照）：")
    lines.append("")
    lines.append("| 年 | 評価コード(0=凶作,1=不作,2=並作,3=豊作) | ラベル |")
    lines.append("|---|---|---|")
    for row in mast.sort("year").iter_rows(named=True):
        lines.append(f"| {row['year']} | {row['mizunara_rating_code']} | {row['mizunara_rating_label']} |")
    lines.append("")

    outcome_vars = ["autumn_count", "autumn_share_of_annual", "autumn_peak_value"]
    for basis in ["raw_record", "dedup_group_v1"]:
        lines.append(f"### {basis}")
        sub_outcomes = outcomes.filter(pl.col("basis") == basis)
        joined = mast.join(sub_outcomes, on="year", how="inner").sort("year")
        lines.append("")
        lines.append("| 年 | ミズナラ評価 | 秋季件数 | 秋季割合 | 秋季週最大値 |")
        lines.append("|---|---|---|---|---|")
        for row in joined.iter_rows(named=True):
            lines.append(
                f"| {row['year']} | {row['mizunara_rating_code']} | {row['autumn_count']} | "
                f"{row['autumn_share_of_annual']:.3f} | {row['autumn_peak_value']} |"
            )
        lines.append("")

        for excl_2025 in [False, True]:
            j = joined.filter(pl.col("year") != 2025) if excl_2025 else joined
            label = "2025年除外" if excl_2025 else "2025年含む"
            if j.height < 4:
                lines.append(f"- {label}：n={j.height}のため順位相関の計算は省略。")
                continue
            years = j.get_column("year").to_list()
            mast_vals = j.get_column("mizunara_rating_code").to_list()
            for var in outcome_vars:
                yv = j.get_column(var).cast(pl.Float64).to_list()
                res = spearman_with_loo([float(v) for v in mast_vals], yv, years)
                direction = "負の関係(評価コードが低い=凶作ほど出没が多い、というH3の予想と整合)" if res["rho"] < 0 else "正の関係(H3の予想と逆方向)" if res["rho"] > 0 else "無相関"
                lines.append(
                    f"- {label}・{var}：Spearman rho={res['rho']:.3f}（参考値、n={res['n']}のためp値は主要根拠にしない, p={res['p_value']:.3f}）。{direction}。"
                    f" leave-one-year-out rho範囲：{min(v for v in res['loo_rho_excluding_year'].values() if v is not None):.3f}〜"
                    f"{max(v for v in res['loo_rho_excluding_year'].values() if v is not None):.3f}"
                )
        lines.append("")

    lines.append(
        "**解釈上の注意**：石狩振興局のミズナラ豊凶評価は北海道庁資料の図を目視で読み取ったものであり、"
        "地域境界での誤読可能性を伴う中程度の信頼度である。また、この評価は石狩振興局全体（札幌市を含むより"
        "広い範囲）のものであり、札幌市内・出没ホットスポット周辺の局地的な食物量とは限らない。"
        "**北海道全体または振興局単位の豊凶指標を札幌市局地の食物量と同一視しない。** n=6という小標本のため、"
        "上記の順位相関は方向性の参考情報として報告するにとどめ、統計的に有意な関係が『ある』『ない』という"
        "結論は書かない。2025年（凶作・出没件数急増）を含めるかどうかで順位相関の大きさが変わる場合、"
        "それは2025年という単年に強く依存した関係である可能性を意味する。"
    )
    return lines


def h5_analysis() -> list[str]:
    lines = ["", "---", "", "## H5: 融雪日と春季ピークの対応", ""]
    weather = pl.read_parquet(WEATHER_PATH)
    outcomes = pl.read_parquet(OUTCOMES_PATH)
    lines.append(f"**n = {weather.height}年（2017-2025、気象データは全期間で取得済み）。"
                 "n=7〜10のため単変量の探索的関連にとどめ、多変量回帰は行わない。**")
    lines.append("")

    peak_vars = ["spring_peak_week", "spring_centroid_doy", "spring_cum50_doy"]
    peak_labels = {"spring_peak_week": "春季最大週(ISO週番号)", "spring_centroid_doy": "春季件数の時間重心(年間通日)", "spring_cum50_doy": "春季累積50%到達日(年間通日)"}
    melt_defs = ["snowmelt_day_persistent14", "snowmelt_day_naive_first_zero"]

    for basis in ["raw_record", "dedup_group_v1"]:
        lines.append(f"### {basis}")
        sub_outcomes = outcomes.filter(pl.col("basis") == basis)
        joined = weather.join(sub_outcomes, on="year", how="inner").sort("year")
        joined = joined.with_columns(
            [
                pl.col("snowmelt_day_persistent14").dt.ordinal_day().alias("melt_doy_persistent14"),
                pl.col("snowmelt_day_naive_first_zero").dt.ordinal_day().alias("melt_doy_naive"),
            ]
        )
        lines.append("")
        lines.append("| 年 | 融雪日(持続14日) | 融雪日(単純初回) | 春季最大週 | 春季重心(通日) | 春季累積50%到達日(通日) |")
        lines.append("|---|---|---|---|---|---|")
        for row in joined.iter_rows(named=True):
            lines.append(
                f"| {row['year']} | {row['snowmelt_day_persistent14']} | {row['snowmelt_day_naive_first_zero']} | "
                f"{row['spring_peak_week']} | {row['spring_centroid_doy']:.1f} | {row['spring_cum50_doy']} |"
            )
        lines.append("")

        for melt_def in melt_defs:
            melt_col = "melt_doy_persistent14" if melt_def == "snowmelt_day_persistent14" else "melt_doy_naive"
            for excl_2025 in [False, True]:
                j = joined.filter(pl.col("year") != 2025) if excl_2025 else joined
                label = "2025年除外" if excl_2025 else "2025年含む"
                if j.filter(pl.col(melt_col).is_not_null()).height < 4:
                    continue
                years = j.get_column("year").to_list()
                melt_vals = [float(v) for v in j.get_column(melt_col).to_list()]
                for var in peak_vars:
                    yv = [float(v) for v in j.get_column(var).to_list()]
                    res = spearman_with_loo(melt_vals, yv, years)
                    lines.append(
                        f"- [{melt_def}] {label}・{peak_labels[var]}：Spearman rho={res['rho']:.3f}"
                        f"（参考値、n={res['n']}、p={res['p_value']:.3f}）。"
                        f"leave-one-year-out rho範囲：{min(v for v in res['loo_rho_excluding_year'].values() if v is not None):.3f}〜"
                        f"{max(v for v in res['loo_rho_excluding_year'].values() if v is not None):.3f}"
                    )
        lines.append("")

    lines.append(
        "**解釈上の注意**：融雪日の定義（持続14日 vs 単純初回0cm）によって年ごとの値が異なりうる"
        "（P5a-2参照）。n=9であっても多変量回帰は行わず、単変量の探索的関連にとどめる。"
        "有意な相関が見られなくても「融雪と春ピークに関係がない」と断定はしない。"
    )
    return lines


def main() -> None:
    lines = ["# P5b/P5c: H3・H5 記述的分析", ""]
    lines.extend(h3_analysis())
    lines.extend(h5_analysis())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "p5_h3_h5_analysis.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
