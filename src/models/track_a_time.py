"""P4b トラックA（時間トラック）: 季節性(H4)・曜日周期(H6)の正式モデル。

季節性は週次パネル、曜日周期は日次パネルを使う（PROJECT_SPEC.md P2の
「時間単位は週。ただし曜日効果の検証(H6)のため、日単位の系列も別途保持する」
という設計をP4bのモデルレベルで実装したもの。P4_MODEL_PLAN.mdの「週単位」という
記述は季節性モデルを指し、曜日モデルは同計画3節が明示する「曜日」変数を
実際に推定可能にするため日次データを用いる。この解釈はPROJECT_SPEC.mdの当初設計を
そのまま踏襲するものであり、変更履歴に記載するような仕様変更ではない）。

4つの重複判定条件（原記録・基準v1・厳しい・緩い）を並行して実行する。
2025年の扱いは3仕様（全期間／2025年除外／2025年ダミー）を並行して実行する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_records  # noqa: E402
from analyze_dedup_sensitivity import CONDITIONS, assign_groups  # noqa: E402
from common_model import fit_nb, loo_year_cv, nb_loglik, zero_check  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "reports"
RESULTS_PATH = OUT_DIR / "p4b_track_a_results.parquet"

MAIN_CATEGORY = "sighting"
SECONDARY_CATEGORIES = ["track_sign", "camera"]


def build_group_table(records: pl.DataFrame, condition_name: str) -> pl.DataFrame:
    """指定条件で重複候補グループを付与したレコード表を返す（raw条件はグループ化しない）。"""
    if condition_name == "raw":
        return records.with_columns(pl.col("source_record_id").alias("group"))
    params = CONDITIONS[condition_name]
    grp = assign_groups(records, **params)
    return records.with_columns(grp)


def representative_rows(df_grouped: pl.DataFrame) -> pl.DataFrame:
    ranked = df_grouped.with_columns(
        pl.col("event_date_parsed").rank(method="ordinal").over("group").alias("_r")
    )
    return ranked.filter(pl.col("_r") == 1).drop("_r")


def zero_fill_daily(df: pl.DataFrame, date_min, date_max, categories: list[str]) -> pl.DataFrame:
    all_dates = pl.date_range(date_min, date_max, interval="1d", eager=True)
    grid = pl.DataFrame({"date": all_dates}).join(pl.DataFrame({"category": categories}), how="cross")
    counts = df.group_by(["event_date_parsed", "category"]).len().rename({"event_date_parsed": "date"})
    out = grid.join(counts, on=["date", "category"], how="left").with_columns(pl.col("len").fill_null(0).alias("count"))
    return out.with_columns(
        [
            pl.col("date").dt.weekday().alias("weekday_num"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.year().alias("year"),
        ]
    )


def zero_fill_weekly(df: pl.DataFrame, date_min, date_max, categories: list[str]) -> pl.DataFrame:
    all_dates = pl.date_range(date_min, date_max, interval="1d", eager=True)
    with_week = pl.DataFrame({"date": all_dates}).with_columns(
        [pl.col("date").dt.iso_year().alias("iso_year"), pl.col("date").dt.week().alias("iso_week"), pl.col("date").dt.month().alias("month")]
    )
    # 週の代表月は、その週で最も多い日数を占める月とする（簡便法：週初の月を使う）
    week_month = with_week.group_by(["iso_year", "iso_week"]).agg(pl.col("month").first().alias("month"), pl.col("date").min().alias("week_start"))
    grid = week_month.join(pl.DataFrame({"category": categories}), how="cross")

    dated = df.with_columns(
        [pl.col("event_date_parsed").dt.iso_year().alias("iso_year"), pl.col("event_date_parsed").dt.week().alias("iso_week")]
    )
    counts = dated.group_by(["iso_year", "iso_week", "category"]).len()
    out = grid.join(counts, on=["iso_year", "iso_week", "category"], how="left").with_columns(pl.col("len").fill_null(0).alias("count"))
    return out.with_columns(pl.col("week_start").dt.year().alias("year"))


def run_seasonal_models(weekly: pl.DataFrame, category: str, condition: str, results: list) -> None:
    sub = weekly.filter(pl.col("category") == category).to_pandas()
    sub["year"] = sub["year"].astype(int)
    sub["month"] = sub["month"].astype(int)
    sub["is_2025"] = (sub["year"] == 2025).astype(int)

    specs = {
        "full_period_year_FE": ("count ~ C(month) + C(year)", sub),
        "excl_2025_year_FE": ("count ~ C(month) + C(year)", sub[sub["year"] != 2025]),
        "full_period_2025_dummy": ("count ~ C(month) + is_2025", sub),
    }
    for spec_name, (formula, data) in specs.items():
        try:
            fit = fit_nb(data, formula)
            # C(year)を含む仕様はleave-one-year-out CVが原理的に成立しない
            # （held-out年のカテゴリ水準が学習データに存在せず予測不能）。
            # year固定効果を含まない仕様（2025ダミー版）でのみ計算する。
            if "C(year)" in formula:
                cv = {"total_loglik": None, "note": "C(year)を含む仕様のためLOO-CV対象外（held-out年の水準が学習データに存在しないため）"}
            else:
                cv = loo_year_cv(data, formula, "year") if data["year"].nunique() > 2 else {"total_loglik": None}
            alpha = fit.alpha
            mu = fit.results.predict()  # 学習データそのものに対する予測（引数なし＝再パース不要で安全）
            zc = zero_check(data["count"].to_numpy(), np.asarray(mu), alpha)
            results.append(
                {
                    "track": "A_seasonal", "condition": condition, "category": category, "spec": spec_name,
                    "formula": formula, "n_obs": fit.n_obs, "aic": fit.aic, "prsquared": fit.prsquared,
                    "loo_cv_loglik": cv["total_loglik"], "alpha": alpha,
                    "obs_zero_rate": zc["observed_zero_rate"], "nb_expected_zero_rate": zc["nb_expected_zero_rate"],
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append({"track": "A_seasonal", "condition": condition, "category": category, "spec": spec_name, "formula": formula, "error": str(e)})


def run_category_interaction_model(weekly: pl.DataFrame, condition: str, results: list) -> dict | None:
    sub = weekly.filter(pl.col("category").is_in([MAIN_CATEGORY, *SECONDARY_CATEGORIES])).to_pandas()
    sub["year"] = sub["year"].astype(int)
    sub["month"] = sub["month"].astype(int)
    sub["season"] = sub["month"].map(lambda m: "spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8) else "autumn" if m in (9, 10, 11) else "winter")
    formula_with_int = "count ~ C(season)*C(category) + C(year)"
    formula_no_int = "count ~ C(season) + C(category) + C(year)"
    try:
        fit_int = fit_nb(sub, formula_with_int)
        fit_no_int = fit_nb(sub, formula_no_int)
        lr_stat = 2 * (fit_int.llf - fit_no_int.llf)
        from scipy import stats as sstats
        df_diff = len(fit_int.results.params) - len(fit_no_int.results.params)
        p_value = 1 - sstats.chi2.cdf(lr_stat, df_diff) if df_diff > 0 else None
        results.append({"track": "A_category_interaction", "condition": condition, "category": "pooled", "spec": "season_x_category", "formula": formula_with_int, "n_obs": fit_int.n_obs, "aic": fit_int.aic, "prsquared": fit_int.prsquared})
        results.append({"track": "A_category_interaction", "condition": condition, "category": "pooled", "spec": "season_plus_category_no_interaction", "formula": formula_no_int, "n_obs": fit_no_int.n_obs, "aic": fit_no_int.aic, "prsquared": fit_no_int.prsquared})
        return {"condition": condition, "lr_stat": lr_stat, "df_diff": df_diff, "p_value": p_value, "aic_with_int": fit_int.aic, "aic_no_int": fit_no_int.aic}
    except Exception as e:  # noqa: BLE001
        results.append({"track": "A_category_interaction", "condition": condition, "category": "pooled", "spec": "season_x_category", "error": str(e)})
        return None


def run_weekday_models(daily: pl.DataFrame, category: str, condition: str, results: list) -> dict:
    sub = daily.filter(pl.col("category") == category).to_pandas()
    sub["year"] = sub["year"].astype(int)
    sub["month"] = sub["month"].astype(int)
    sub["weekday_num"] = sub["weekday_num"].astype(int)
    sub["is_2025"] = (sub["year"] == 2025).astype(int)

    specs = {
        "full_period_year_FE": ("count ~ C(weekday_num) + C(month) + C(year)", sub),
        "excl_2025_year_FE": ("count ~ C(weekday_num) + C(month) + C(year)", sub[sub["year"] != 2025]),
        "full_period_2025_dummy": ("count ~ C(weekday_num) + C(month) + is_2025", sub),
    }
    lr_results = {}
    for spec_name, (formula, data) in specs.items():
        try:
            fit = fit_nb(data, formula)
            null_formula = formula.replace("C(weekday_num) + ", "")
            fit_null = fit_nb(data, null_formula)
            lr_stat = 2 * (fit.llf - fit_null.llf)
            from scipy import stats as sstats
            df_diff = len(fit.results.params) - len(fit_null.results.params)
            p_value = 1 - sstats.chi2.cdf(lr_stat, df_diff) if df_diff > 0 else None

            weekday_coefs = {k: v for k, v in fit.results.params.items() if "weekday_num" in k}
            weekday_ci = fit.results.conf_int()
            weekday_ci_dict = {idx: (weekday_ci.loc[idx, 0], weekday_ci.loc[idx, 1]) for idx in weekday_coefs}

            results.append(
                {
                    "track": "A_weekday", "condition": condition, "category": category, "spec": spec_name,
                    "formula": formula, "n_obs": fit.n_obs, "aic": fit.aic, "prsquared": fit.prsquared,
                    "weekday_lr_stat": lr_stat, "weekday_lr_pvalue": p_value, "weekday_lr_df": df_diff,
                }
            )
            lr_results[spec_name] = {"lr_stat": lr_stat, "p_value": p_value, "weekday_coefs": weekday_coefs, "weekday_ci": weekday_ci_dict}
        except Exception as e:  # noqa: BLE001
            results.append({"track": "A_weekday", "condition": condition, "category": category, "spec": spec_name, "formula": formula, "error": str(e)})
    return lr_results


def main() -> None:
    records = load_records()
    categories = ["sighting", "sighting_uncertain", "track_sign", "camera", "other"]
    date_min = records.get_column("event_date_parsed").min()
    date_max = records.get_column("event_date_parsed").max()

    results = []
    weekday_lr_summary = {}
    interaction_summary = {}

    for condition in ["raw", "厳しい(strict)", "基準(baseline, v1)", "緩い(loose)"]:
        print(f"=== condition: {condition} ===")
        grouped = build_group_table(records, "raw" if condition == "raw" else condition)
        reps = grouped if condition == "raw" else representative_rows(grouped)

        weekly = zero_fill_weekly(reps, date_min, date_max, categories)
        daily = zero_fill_daily(reps, date_min, date_max, categories)

        run_seasonal_models(weekly, MAIN_CATEGORY, condition, results)
        for cat in SECONDARY_CATEGORIES:
            run_seasonal_models(weekly, cat, condition, results)

        interaction_summary[condition] = run_category_interaction_model(weekly, condition, results)

        lr = run_weekday_models(daily, MAIN_CATEGORY, condition, results)
        weekday_lr_summary[condition] = lr
        for cat in SECONDARY_CATEGORIES:
            run_weekday_models(daily, cat, condition, results)

    results_df = pl.DataFrame(results, infer_schema_length=None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(RESULTS_PATH)
    print(f"\nwritten {results_df.height} rows to {RESULTS_PATH}")

    # 曜日係数の条件間・2025除外での安定性を要約
    print("\n=== weekday coefficient stability (main category, full_period_year_FE) ===")
    for cond, lr in weekday_lr_summary.items():
        spec = lr.get("full_period_year_FE", {})
        print(cond, "LR p-value:", spec.get("p_value"))
        coefs = spec.get("weekday_coefs", {})
        if coefs:
            max_wd = max(coefs, key=coefs.get)
            print("  max weekday coef:", max_wd, coefs[max_wd])

    print("\n=== season x category interaction ===")
    for cond, info in interaction_summary.items():
        print(cond, info)


if __name__ == "__main__":
    main()
