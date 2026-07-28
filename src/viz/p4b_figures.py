"""P4b: トラックBの残差空間図。

共変量投入後（基準v1条件、原記録・重複集約条件別）の残差を1kmメッシュ上に
可視化する。正の残差＝共変量から予測される件数より実際の出没が多かったメッシュ、
負の残差＝少なかったメッシュ。詳細座標は出さず、1kmメッシュ集計のみ表示する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_matplotlib, savefig  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESID_PATH = PROJECT_ROOT / "reports" / "p4b_track_b_residuals.parquet"

# 診断的に見やすいdiverging配色（dataviz skillの規定：blue<->red、中立グレーは0付近）
DIVERGING_CMAP = "RdBu_r"  # 正=赤(件数超過)、負=青(件数不足)。中立=白


def plot_residual_map(ax, sub: pl.DataFrame, vmax: float) -> None:
    lat_unique = sorted(sub.get_column("lat_center").unique().to_list())
    lon_unique = sorted(sub.get_column("lon_center").unique().to_list())
    lat_idx = {v: i for i, v in enumerate(lat_unique)}
    lon_idx = {v: i for i, v in enumerate(lon_unique)}
    mat = np.full((len(lat_unique), len(lon_unique)), np.nan)
    for row in sub.iter_rows(named=True):
        mat[lat_idx[row["lat_center"]], lon_idx[row["lon_center"]]] = row["resid_sum"]
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=DIVERGING_CMAP, norm=norm, origin="lower", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main() -> None:
    setup_matplotlib()
    resid = pl.read_parquet(RESID_PATH)

    conditions = ["raw", "厳しい(strict)", "基準(baseline, v1)", "緩い(loose)"]
    dep_vars = ["count_sighting", "count_all"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col, condition in enumerate(conditions):
        for row_i, dep_var in enumerate(dep_vars):
            sub = resid.filter((pl.col("condition") == condition) & (pl.col("dep_var") == dep_var))
            vmax = float(sub.get_column("resid_sum").abs().max())
            ax = axes[row_i, col]
            im = plot_residual_map(ax, sub, vmax)
            if row_i == 0:
                ax.set_title(condition, fontsize=10)
            if col == 0:
                ax.set_ylabel(dep_var, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7, label="残差(合計)")
    fig.suptitle("トラックB 残差空間図（共変量+年FE投入後、1kmメッシュ、赤=超過/青=不足）")
    fig.tight_layout()
    savefig(fig, "p4b_track_b_residual_maps")


if __name__ == "__main__":
    main()
