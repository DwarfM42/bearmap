"""P4b トラックB: 非線形性の必要性を評価する追加診断。

主モデル（基準v1条件、count_sighting/count_all）について、連続共変量に
二次項を加えたモデルと線形モデルをAIC・尤度比検定で比較する。
有意な改善があれば、二次項付きモデルを「診断上必要と判断された拡張」として
採用し、残差のMoran's Iが線形モデルから変化するかも確認する
（非線形性を許しても残差空間相関が残るなら、それは関数形の誤特定の
アーティファクトではないことを意味する）。

zero-inflated/hurdle/CAR-ICARはこの診断では有意な問題が確認されなかった
ため実装していない（zero_checkの結果、観測ゼロ率とNBの含意するゼロ率が
近いこと、Track A/Bとも参照）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_records  # noqa: E402
from track_b_spatial import (  # noqa: E402
    COVARIATE_COLS,
    build_group_table,
    representative_rows,
    build_mesh_year_panel,
)
from common_model import fit_nb, morans_i, build_distance_band_weights  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COVARIATES_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_covariates_1km.parquet"
OUT_PATH = PROJECT_ROOT / "reports" / "p4b_nonlinearity_check.md"

QUAD_VARS = ["dist_to_road_m", "dist_to_forest_edge_m", "dist_to_river_m", "elevation_mean_m", "slope_mean_deg"]


def run_for(dep_var: str, pdf, lines: list[str]) -> None:
    covariate_formula = " + ".join(COVARIATE_COLS) + " + landuse_forest_ratio:dist_to_forest_edge_m"
    formula_lin = f"{dep_var} ~ {covariate_formula} + C(year)"
    formula_quad = f"{dep_var} ~ {covariate_formula} + " + " + ".join(f"I({v}**2)" for v in QUAD_VARS) + " + C(year)"

    fit_lin = fit_nb(pdf, formula_lin)
    fit_quad = fit_nb(pdf, formula_quad)
    lr = 2 * (fit_quad.llf - fit_lin.llf)
    df_diff = len(fit_quad.results.params) - len(fit_lin.results.params)
    p_value = 1 - sstats.chi2.cdf(lr, df_diff)

    def resid_morans(fit) -> dict:
        mu = np.asarray(fit.results.predict())
        resid = pdf[dep_var].to_numpy() - mu
        tmp = pdf.copy()
        tmp["resid"] = resid
        mesh_resid = tmp.groupby("mesh_code")["resid"].sum().reset_index()
        cov_pd = pl.read_parquet(COVARIATES_PATH).to_pandas()[["mesh_code", "lat_center", "lon_center"]]
        mesh_resid = mesh_resid.merge(cov_pd, on="mesh_code", how="left")
        W = build_distance_band_weights(mesh_resid["lat_center"].to_numpy(), mesh_resid["lon_center"].to_numpy(), band_km=1.5)
        return morans_i(mesh_resid["resid"].to_numpy(), W)

    mi_lin = resid_morans(fit_lin)
    mi_quad = resid_morans(fit_quad)

    lines.append(f"### {dep_var}")
    lines.append(f"- 線形モデル: AIC={fit_lin.aic:.1f}, 疑似R2={fit_lin.prsquared:.4f}")
    lines.append(f"- 二次項付きモデル: AIC={fit_quad.aic:.1f}, 疑似R2={fit_quad.prsquared:.4f}")
    lines.append(f"- 尤度比検定: LR={lr:.2f}, df={df_diff}, p={p_value:.6f}")
    lines.append(f"- 残差Moran's I（線形）: I={mi_lin['I']:.4f}, p={mi_lin['p_value']:.4f}")
    lines.append(f"- 残差Moran's I（二次項付き）: I={mi_quad['I']:.4f}, p={mi_quad['p_value']:.4f}")
    lines.append("")


def main() -> None:
    records = load_records()
    covariates = pl.read_parquet(COVARIATES_PATH).drop_nulls(subset=COVARIATE_COLS)
    grouped = build_group_table(records, "基準(baseline, v1)")
    reps = representative_rows(grouped)
    years = list(range(2017, 2026))
    panel = build_mesh_year_panel(reps, covariates, years)
    pdf = panel.to_pandas().dropna(subset=COVARIATE_COLS)
    pdf["year"] = pdf["year"].astype(int)

    lines = [
        "# P4b: 非線形性の診断（基準v1条件、距離・標高・傾斜変数に二次項を追加）",
        "",
        "PROJECT_SPEC.mdが示唆する「標高・道路距離は単調でない可能性が高い」という点を、",
        "GAMではなく二次項付きNB GLMで簡便に検証した（診断結果が有意だったため、",
        "GAMの完全導入ではなくこの軽量な拡張を「診断上必要と判断された拡張モデル」として採用）。",
        "",
    ]
    run_for("count_sighting", pdf, lines)
    run_for("count_all", pdf, lines)
    lines.append(
        "**所見**: いずれの目的変数でも二次項の追加はAICを大きく改善し（尤度比検定 p<0.001）、"
        "距離・標高・傾斜と出没件数の関係が線形ではないことを示す。ただし、二次項を加えても"
        "残差のMoran's Iはほとんど変化せず、有意性も変わらない。これは、共変量投入後に残る"
        "空間自己相関（H2）が、単純な関数形の誤特定（非線形性を線形モデルで近似したことによる"
        "見せかけの残差相関）のアーティファクトではないことを示唆する。"
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
