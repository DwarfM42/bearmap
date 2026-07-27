"""P4a-4: 重複判定（同一個体候補グループ化）の感度分析。

現在の試作ルール（v1: 1500m・3日・同一区）に加え、閾値を変えた
「厳しい」「緩い」の2条件を作成し、合計3条件で以下を比較する：
- 総グループ数
- 年別重複率
- 季節性（週別集計の形）
- 曜日分布
- 主要ホットスポット（1kmメッシュ、上位5セル）

いずれも「正しい重複除去」ではなく、感度分析の一系列として扱う。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from common import load_records, round_to_mesh  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[2] / "reports" / "p4a_dedup_sensitivity.md"

CONDITIONS = {
    "厳しい(strict)": {"distance_m": 300.0, "days": 1, "same_ward_required": True},
    "基準(baseline, v1)": {"distance_m": 1500.0, "days": 3, "same_ward_required": True},
    "緩い(loose)": {"distance_m": 3000.0, "days": 7, "same_ward_required": False},
}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def assign_groups(records: pl.DataFrame, distance_m: float, days: int, same_ward_required: bool) -> pl.Series:
    n = records.height
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rows = records.select(["ward", "event_date_parsed", "lat", "lon"]).to_dicts()
    if same_ward_required:
        buckets: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            buckets.setdefault(r["ward"] or "unknown", []).append(i)
    else:
        buckets = {"__all__": list(range(n))}

    for idxs in buckets.values():
        for a_pos in range(len(idxs)):
            i = idxs[a_pos]
            ri = rows[i]
            if ri["event_date_parsed"] is None or ri["lat"] is None or ri["lon"] is None:
                continue
            for b_pos in range(a_pos + 1, len(idxs)):
                j = idxs[b_pos]
                rj = rows[j]
                if rj["event_date_parsed"] is None or rj["lat"] is None or rj["lon"] is None:
                    continue
                if abs((ri["event_date_parsed"] - rj["event_date_parsed"]).days) > days:
                    continue
                if haversine_m(ri["lat"], ri["lon"], rj["lat"], rj["lon"]) <= distance_m:
                    union(i, j)

    ids = records.get_column("source_record_id").to_list()
    roots = [find(i) for i in range(n)]
    return pl.Series("group", [ids[r] for r in roots])


def representative_table(records: pl.DataFrame, group_col: pl.Series) -> pl.DataFrame:
    df = records.with_columns(group_col)
    ranked = df.with_columns(pl.col("event_date_parsed").rank(method="ordinal").over("group").alias("_r"))
    return ranked.filter(pl.col("_r") == 1).drop("_r")


def main() -> None:
    records = load_records()
    lines = ["# P4a-4: 重複判定の感度分析", "", "いずれも「正しい重複除去」ではなく感度分析の一系列として扱う。", ""]

    summaries = {}
    for name, params in CONDITIONS.items():
        print(f"computing condition: {name} ({params}) ...")
        grp = assign_groups(records, **params)
        reps = representative_table(records, grp)
        summaries[name] = reps

        n_groups = reps.height
        by_year = records.with_columns(grp).group_by("_source_year").agg(
            pl.col("group").n_unique().alias("n_groups"), pl.len().alias("n_raw")
        ).sort("_source_year").with_columns((1 - pl.col("n_groups") / pl.col("n_raw")).alias("dup_rate"))

        lines.append(f"## 条件: {name}")
        lines.append(f"- パラメータ: 距離閾値={params['distance_m']}m, 日数閾値={params['days']}日, "
                     f"同一区限定={params['same_ward_required']}")
        lines.append(f"- 総グループ数: {n_groups}（原記録{records.height}件）")
        lines.append("\n年別重複率:")
        for row in by_year.iter_rows(named=True):
            lines.append(f"- {row['_source_year']}: 原記録{row['n_raw']}件 → グループ{row['n_groups']}件 "
                         f"(重複率{row['dup_rate']:.1%})")

        weekday = reps.with_columns(pl.col("event_date_parsed").dt.weekday().alias("wd")).group_by("wd").len().sort("wd")
        lines.append("\n曜日分布（代表日ベース、月=1〜日=7）:")
        for row in weekday.iter_rows(named=True):
            lines.append(f"- 曜日{row['wd']}: {row['len']}件")

        mesh = reps.filter(pl.col("location_precision") == "point")
        bins = [round_to_mesh(r["lat"], r["lon"], 1.0) for r in mesh.select(["lat", "lon"]).to_dicts()]
        mesh_df = pl.DataFrame({"lat_bin": [b[0] for b in bins], "lon_bin": [b[1] for b in bins]})
        top_hotspots = mesh_df.group_by(["lat_bin", "lon_bin"]).len().sort("len", descending=True).head(5)
        lines.append("\n主要ホットスポット（1kmメッシュ、上位5セル、緯度経度は概算丸め値）:")
        for row in top_hotspots.iter_rows(named=True):
            lines.append(f"- ({row['lat_bin']:.4f}, {row['lon_bin']:.4f}): {row['len']}件")
        lines.append("")

    # 条件間の比較（総グループ数、代表的な年の重複率）
    lines.append("## 条件間の比較サマリ")
    lines.append("| 条件 | 総グループ数 | 2019年重複率 | 2025年重複率 |")
    lines.append("|---|---|---|---|")
    for name, params in CONDITIONS.items():
        grp = assign_groups(records, **params)
        reps = representative_table(records, grp)
        by_year = records.with_columns(grp).group_by("_source_year").agg(
            pl.col("group").n_unique().alias("n_groups"), pl.len().alias("n_raw")
        ).with_columns((1 - pl.col("n_groups") / pl.col("n_raw")).alias("dup_rate"))
        r2019 = by_year.filter(pl.col("_source_year") == 2019).get_column("dup_rate").to_list()
        r2025 = by_year.filter(pl.col("_source_year") == 2025).get_column("dup_rate").to_list()
        lines.append(
            f"| {name} | {reps.height} | {r2019[0]:.1%} | {r2025[0]:.1%} |"
        )

    lines.append("")
    lines.append(
        "**所見**: 閾値を厳しくするほどグループ数は増え（重複除去が弱まる）、緩くするほどグループ数は減る"
        "（重複除去が強まる）という単調な関係が期待通り確認できるはずである。年別重複率の絶対水準は条件により"
        "変わるが、**どの年が相対的に重複率が高いか（2019年・2025年が高いという順序）が条件を通じて安定しているか**"
        "が、この結果の頑健性を判断する鍵になる。曜日分布・ホットスポットの位置が条件によって大きく入れ替わる"
        "場合は、P3・P4での季節性・空間パターンの記述がルール依存であることを意味し、モデル化において"
        "重複判定ルールを固定パラメータとして明記する必要がある。"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
