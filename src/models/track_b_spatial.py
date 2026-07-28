"""P4b トラックB（空間トラック）: 人口・道路等の共変量投入後の残差空間構造(H2)、
人間活動変数投入前後の改善量(H1関連)を検証する。

空間単位：1kmメッシュ（JIS 3次メッシュ）。時間単位：年（暦年）。
4つの重複判定条件（原記録・基準v1・厳しい・緩い）を並行して実行する。
2025年の扱いは3仕様（全期間／2025年除外／2025年ダミー）を並行して実行する。

zero-inflated/hurdle/GAM/CAR-ICARは機械的に全実行しない。まずNB GLMベースラインを
実行し、残差診断（ゼロ過剰チェック、Moran's I）の結果を見てから、必要と判断した
拡張モデルのみ追加する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_records, jis_3rd_mesh_code  # noqa: E402
from analyze_dedup_sensitivity import CONDITIONS, assign_groups  # noqa: E402
from common_model import fit_nb, morans_i, build_distance_band_weights, vif_table, zero_check  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COVARIATES_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_covariates_1km.parquet"
OUT_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = OUT_DIR / "p4b_track_b_results.parquet"
RESID_PATH = OUT_DIR / "p4b_track_b_residuals.parquet"

COVARIATE_COLS = [
    "population_2020", "road_len_major_m", "dist_to_road_m",
    "landuse_residential_ratio", "landuse_forest_ratio",
    "dist_to_forest_edge_m", "dist_to_river_m",
    "elevation_mean_m", "slope_mean_deg",
]


def build_group_table(records: pl.DataFrame, condition_name: str) -> pl.DataFrame:
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


def build_mesh_year_panel(reps: pl.DataFrame, covariates: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    valid = reps.filter(pl.col("location_precision") == "point")
    codes_sighting = []
    codes_all = []
    for row in valid.select(["lat", "lon", "category", "event_date_parsed"]).to_dicts():
        code = jis_3rd_mesh_code(row["lat"], row["lon"])
        codes_all.append({"mesh_code": code, "year": row["event_date_parsed"].year, "is_sighting": row["category"] == "sighting"})
    events = pl.DataFrame(codes_all)

    mesh_codes = covariates.get_column("mesh_code").to_list()
    grid = pl.DataFrame({"mesh_code": mesh_codes}).join(pl.DataFrame({"year": years}), how="cross")

    count_all = events.group_by(["mesh_code", "year"]).len().rename({"len": "count_all"})
    count_sighting = events.filter(pl.col("is_sighting")).group_by(["mesh_code", "year"]).len().rename({"len": "count_sighting"})

    panel = grid.join(count_all, on=["mesh_code", "year"], how="left").join(count_sighting, on=["mesh_code", "year"], how="left")
    panel = panel.with_columns([pl.col("count_all").fill_null(0), pl.col("count_sighting").fill_null(0)])
    panel = panel.join(covariates, on="mesh_code", how="left")
    return panel


def run_track_b_for_condition(reps: pl.DataFrame, covariates: pl.DataFrame, condition: str, dep_var: str, results: list, resid_rows: list) -> None:
    years = list(range(2017, 2026))
    panel = build_mesh_year_panel(reps, covariates, years)
    pdf = panel.to_pandas()
    pdf = pdf.dropna(subset=COVARIATE_COLS)  # DEM欠損30セルなど、共変量が欠けているセルを除外
    pdf["year"] = pdf["year"].astype(int)
    pdf["is_2025"] = (pdf["year"] == 2025).astype(int)
    y_col = dep_var

    covariate_formula = " + ".join(COVARIATE_COLS) + " + landuse_forest_ratio:dist_to_forest_edge_m"

    specs = {
        "null_no_covariates_year_FE": f"{y_col} ~ C(year)",
        "full_covariates_year_FE": f"{y_col} ~ {covariate_formula} + C(year)",
        "full_covariates_excl_2025": (f"{y_col} ~ {covariate_formula} + C(year)", pdf[pdf['year'] != 2025]),
        "full_covariates_2025_dummy": f"{y_col} ~ {covariate_formula} + is_2025",
    }

    fitted_full_main = None
    for spec_name, spec in specs.items():
        if isinstance(spec, tuple):
            formula, data = spec
        else:
            formula, data = spec, pdf
        try:
            fit = fit_nb(data, formula)
            alpha = fit.alpha
            mu = fit.results.predict()  # 学習データそのものに対する予測（引数なし＝再パース不要で安全）
            zc = zero_check(data[y_col].to_numpy(), np.asarray(mu), alpha)
            row = {
                "track": "B_spatial", "condition": condition, "dep_var": dep_var, "spec": spec_name,
                "n_obs": fit.n_obs, "aic": fit.aic, "prsquared": fit.prsquared, "alpha": alpha,
                "obs_zero_rate": zc["observed_zero_rate"], "nb_expected_zero_rate": zc["nb_expected_zero_rate"],
            }
            results.append(row)
            if spec_name == "full_covariates_year_FE":
                fitted_full_main = (fit, data)
        except Exception as e:  # noqa: BLE001
            results.append({"track": "B_spatial", "condition": condition, "dep_var": dep_var, "spec": spec_name, "error": str(e)})

    # VIF（共変量投入モデルの説明変数について）
    try:
        vifs = vif_table(pdf, COVARIATE_COLS)
        for var, v in vifs.items():
            results.append({"track": "B_spatial_VIF", "condition": condition, "dep_var": dep_var, "spec": "VIF", "variable": var, "vif": v})
    except Exception as e:  # noqa: BLE001
        results.append({"track": "B_spatial_VIF", "condition": condition, "dep_var": dep_var, "spec": "VIF", "error": str(e)})

    # 残差のMoran's I（基準モデル: full_covariates_year_FE、メッシュ単位で年合計した残差を使う）
    if fitted_full_main is not None:
        fit, data = fitted_full_main
        data = data.copy()
        data["resid"] = data[y_col].to_numpy() - np.asarray(fit.results.predict())
        mesh_resid = data.groupby("mesh_code")["resid"].sum().reset_index()
        mesh_resid = mesh_resid.merge(covariates.to_pandas()[["mesh_code", "lat_center", "lon_center"]], on="mesh_code", how="left")
        W = build_distance_band_weights(mesh_resid["lat_center"].to_numpy(), mesh_resid["lon_center"].to_numpy(), band_km=1.5)
        mi = morans_i(mesh_resid["resid"].to_numpy(), W)
        results.append({"track": "B_spatial_MoransI", "condition": condition, "dep_var": dep_var, "spec": "full_covariates_year_FE", "morans_i": mi["I"], "p_value": mi["p_value"], "n_cells": mi["n"]})
        for _, r in mesh_resid.iterrows():
            resid_rows.append({"condition": condition, "dep_var": dep_var, "mesh_code": r["mesh_code"], "lat_center": r["lat_center"], "lon_center": r["lon_center"], "resid_sum": r["resid"]})


def main() -> None:
    records = load_records()
    covariates = pl.read_parquet(COVARIATES_PATH).drop_nulls(subset=COVARIATE_COLS)
    print(f"covariate cells after dropping DEM-missing: {covariates.height}")

    results = []
    resid_rows = []
    for condition in ["raw", "厳しい(strict)", "基準(baseline, v1)", "緩い(loose)"]:
        print(f"=== Track B condition: {condition} ===")
        grouped = build_group_table(records, condition)
        reps = grouped if condition == "raw" else representative_rows(grouped)
        for dep_var in ["count_sighting", "count_all"]:
            run_track_b_for_condition(reps, covariates, condition, dep_var, results, resid_rows)

    results_df = pl.DataFrame(results, infer_schema_length=None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(RESULTS_PATH)
    pl.DataFrame(resid_rows).write_parquet(RESID_PATH)
    print(f"\nwritten {results_df.height} rows to {RESULTS_PATH}")
    print(f"written {len(resid_rows)} residual rows to {RESID_PATH}")


if __name__ == "__main__":
    main()
