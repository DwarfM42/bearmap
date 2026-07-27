"""P4a-2: 500m/1km/2kmメッシュ候補の非ゼロセル数・件数分布を比較する。

出没データ（原記録ベース）のみを用い、共変量取得前に空間単位を決定する。
勝手にメッシュを統合せず、候補ごとの実態を報告することが目的。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from common import load_records  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[2] / "reports" / "p4a_spatial_unit_comparison.md"

SAPPORO_LAT_REF = 43.0  # メッシュ幅換算の基準緯度


def round_to_mesh(lat: float, lon: float, mesh_km: float) -> tuple[float, float]:
    lat_step = mesh_km / 111.0
    lon_step = mesh_km / (111.0 * math.cos(math.radians(SAPPORO_LAT_REF)))
    lat_bin = math.floor(lat / lat_step) * lat_step
    lon_bin = math.floor(lon / lon_step) * lon_step
    return lat_bin, lon_bin


def analyze(records: pl.DataFrame, mesh_km: float) -> dict:
    valid = records.filter(pl.col("location_precision") == "point")
    bins = [round_to_mesh(r["lat"], r["lon"], mesh_km) for r in valid.select(["lat", "lon"]).to_dicts()]
    df = pl.DataFrame({"lat_bin": [b[0] for b in bins], "lon_bin": [b[1] for b in bins]})
    counts = df.group_by(["lat_bin", "lon_bin"]).len()

    # 対象領域全体のセル総数（データが存在する範囲のバウンディングボックス内）を分母とする
    lat_min, lat_max = counts.get_column("lat_bin").min(), counts.get_column("lat_bin").max()
    lon_min, lon_max = counts.get_column("lon_bin").min(), counts.get_column("lon_bin").max()
    lat_step = mesh_km / 111.0
    lon_step = mesh_km / (111.0 * math.cos(math.radians(SAPPORO_LAT_REF)))
    n_lat_cells = round((lat_max - lat_min) / lat_step) + 1
    n_lon_cells = round((lon_max - lon_min) / lon_step) + 1
    total_cells_in_bbox = n_lat_cells * n_lon_cells

    nonzero_cells = counts.height
    vals = counts.get_column("len").to_list()
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    median = vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2

    return {
        "mesh_km": mesh_km,
        "nonzero_cells": nonzero_cells,
        "total_cells_in_bbox": total_cells_in_bbox,
        "zero_ratio_in_bbox": 1 - nonzero_cells / total_cells_in_bbox,
        "median_count_nonzero": median,
        "mean_count_nonzero": sum(vals) / n,
        "max_count": max(vals),
        "total_records": sum(vals),
    }


def main() -> None:
    records = load_records()
    results = [analyze(records, km) for km in [0.5, 1.0, 2.0]]

    lines = ["# P4a-2: 空間単位候補の比較（原記録ベース）", ""]
    lines.append("| メッシュ | 非ゼロセル数 | 対象範囲内総セル数 | 対象範囲内ゼロ比率 | 非ゼロセルの中央値 | 非ゼロセルの平均 | 最大件数セル | 総件数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['mesh_km']}km | {r['nonzero_cells']} | {r['total_cells_in_bbox']} | "
            f"{r['zero_ratio_in_bbox']:.1%} | {r['median_count_nonzero']:.1f} | "
            f"{r['mean_count_nonzero']:.2f} | {r['max_count']} | {r['total_records']} |"
        )
    lines.append("")
    lines.append(
        "注: 「対象範囲内総セル数」は出没が1件以上記録された領域のバウンディングボックスを"
        "各メッシュ幅で分割した場合の理論セル数（矩形の外接領域であり、実際の市域形状には即していない）。"
        "「ゼロ比率」はこの矩形内での粗い目安であり、市域外の海・山間部を含むため実際の可住地・生息地における"
        "ゼロ比率とは異なる点に注意。"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
