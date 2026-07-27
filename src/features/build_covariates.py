"""P4a-1: 取得したraw共変量データを、JIS標準地域メッシュ3次メッシュ（約1km四方）
単位に集計する。

対象グリッド：出没データのバウンディングボックス（P2_GATE.md/P4a-2で確認した範囲）を
JIS 3次メッシュで埋め尽くした全セル（出没が0件のセルも含む）。

出力: data/processed/sapporo_covariates_1km.parquet
列: mesh_code, lat_center, lon_center,
    population_2020（2020年国勢調査、非補間・全年同一値。仮定はDATA_SOURCES.md参照）,
    road_len_major_m（主要道路の総延長, メートル）,
    dist_to_road_m（メッシュ中心から最も近い主要道路までの距離）,
    landuse_residential_ratio（宅地系土地利用の被覆率、建物密度の代理変数）,
    landuse_forest_ratio（森林・樹林地の被覆率）,
    dist_to_forest_edge_m（森林/非森林境界までの距離）,
    dist_to_river_m（河川までの距離）,
    elevation_mean_m, slope_mean_deg（DEMより算出）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import osmium
import polars as pl
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from common import jis_3rd_mesh_bounds, jis_3rd_mesh_center, jis_3rd_mesh_code  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "sapporo_covariates_1km.parquet"

BBOX = (42.86, 141.04, 43.18, 141.50)  # south, west, north, east — P2_GATE/P4a-2と同じ範囲
PBF_PATH = RAW_DIR / "osm_pbf" / "hokkaido-latest.osm.pbf"

MAJOR_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
}

DEM_ZOOM = 14
DEM_DIR = RAW_DIR / "dem_sapporo"
DEM_PIXEL_SIZE_M = 10.0  # GSI「dem」レイヤは10mメッシュとして提供されている（公称値、厳密な緯度補正はしない近似）


def _latlon_to_tile_frac(lat: float, lon: float, z: int) -> tuple[float, float]:
    lat_rad = np.radians(lat)
    n = 2**z
    x = (lon + 180) / 360 * n
    y = (1 - np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / np.pi) / 2 * n
    return x, y


def load_dem_tiles() -> dict[tuple[int, int], np.ndarray]:
    tiles = {}
    if not DEM_DIR.exists():
        return tiles
    for p in DEM_DIR.glob("*.txt"):
        x_str, y_str = p.stem.split("_")
        x, y = int(x_str), int(y_str)
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            vals = []
            for v in line.split(","):
                v = v.strip()
                if v in ("e", "", "-"):
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(np.nan)
            if vals:
                rows.append(vals)
        if rows:
            tiles[(x, y)] = np.array(rows, dtype=float)
    return tiles


def sample_elevation_slope(
    lat: float, lon: float, mesh_bounds: tuple[float, float, float, float], tiles: dict[tuple[int, int], np.ndarray]
) -> tuple[float | None, float | None]:
    """メッシュセルの範囲に対応するタイル内ピクセル範囲を取り出し、平均標高と平均傾斜を返す。
    セルがタイル境界をまたぐ場合は、セル中心を含むタイルのみを使う近似。"""
    if not tiles:
        return None, None
    xf, yf = _latlon_to_tile_frac(lat, lon, DEM_ZOOM)
    tile_x, tile_y = int(xf), int(yf)
    arr = tiles.get((tile_x, tile_y))
    if arr is None or arr.size == 0:
        return None, None

    s, w, n, e = mesh_bounds
    # セル四隅のタイル内ピクセル座標（256px/タイル）を求める
    px_list, py_list = [], []
    for lat_c, lon_c in [(s, w), (s, e), (n, w), (n, e)]:
        xf2, yf2 = _latlon_to_tile_frac(lat_c, lon_c, DEM_ZOOM)
        px_list.append((xf2 - tile_x) * 256)
        py_list.append((yf2 - tile_y) * 256)
    col_min, col_max = max(0, int(min(px_list))), min(255, int(max(px_list)))
    row_min, row_max = max(0, int(min(py_list))), min(255, int(max(py_list)))
    if col_max < col_min or row_max < row_min or row_max >= arr.shape[0] or col_max >= arr.shape[1]:
        row_min, row_max = max(0, row_min), min(arr.shape[0] - 1, max(row_min, row_max))
        col_min, col_max = max(0, col_min), min(arr.shape[1] - 1, max(col_min, col_max))

    sub = arr[row_min : row_max + 1, col_min : col_max + 1]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return None, None
    elevation_mean = float(np.nanmean(sub))

    gy, gx = np.gradient(arr)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2) / DEM_PIXEL_SIZE_M)
    slope_deg_full = np.degrees(slope_rad)
    sub_slope = slope_deg_full[row_min : row_max + 1, col_min : col_max + 1]
    slope_mean = float(np.nanmean(sub_slope)) if not np.all(np.isnan(sub_slope)) else None
    return elevation_mean, slope_mean


class _CovariateHandler(osmium.SimpleHandler):
    """hokkaido-latest.osm.pbf を1回走査し、分析範囲（BBOXに少しでもかかるway）の
    道路・土地利用（landuse=*, natural=wood）・河川ラインを抽出する。"""

    def __init__(self, bbox: tuple[float, float, float, float]):
        super().__init__()
        self.s, self.w, self.n, self.e = bbox
        self.road_lines: list[LineString] = []
        self.road_tags: list[str] = []
        self.landuse_polys: list[Polygon] = []
        self.landuse_tags: list[str] = []
        self.water_lines: list[LineString] = []

    def _in_bbox(self, coords: list[tuple[float, float]]) -> bool:
        return any(self.w <= lon <= self.e and self.s <= lat <= self.n for lon, lat in coords)

    def way(self, w):  # noqa: N802 (osmium callback naming)
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:
            return
        if len(coords) < 2 or not self._in_bbox(coords):
            return

        highway = w.tags.get("highway")
        if highway in MAJOR_HIGHWAY_TYPES:
            self.road_lines.append(LineString(coords))
            self.road_tags.append(highway)
            return

        landuse = w.tags.get("landuse")
        natural = w.tags.get("natural")
        if (landuse or natural == "wood") and len(coords) >= 4:
            ring = coords if coords[0] == coords[-1] else coords + [coords[0]]
            try:
                poly = Polygon(ring)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.area > 0:
                    self.landuse_polys.append(poly)
                    self.landuse_tags.append(landuse or natural)
            except Exception:
                pass
            return

        if "waterway" in w.tags:
            self.water_lines.append(LineString(coords))


def load_osm_layers() -> _CovariateHandler:
    handler = _CovariateHandler(BBOX)
    handler.apply_file(str(PBF_PATH), locations=True)
    return handler


def build_grid() -> list[str]:
    """BBOXを覆う3次メッシュコードの一覧を作る。"""
    s, w, n, e = BBOX
    codes = set()
    lat = s
    while lat <= n:
        lon = w
        while lon <= e:
            codes.add(jis_3rd_mesh_code(lat, lon))
            lon += 45 / 3600 / 2  # 半セル刻みで走査し取りこぼしを防ぐ
        lat += (0.5 / 60) / 2
    return sorted(codes)


def load_population() -> pl.DataFrame:
    txt_path = RAW_DIR / "population_mesh" / "tblT001141H01.txt"
    with open(txt_path, encoding="shift_jis") as f:
        lines = f.read().splitlines()
    header = lines[0].split(",")
    key_idx = header.index("KEY_CODE")
    pop_idx = header.index("T001141001")
    rows = []
    for line in lines[2:]:
        parts = line.split(",")
        if len(parts) <= max(key_idx, pop_idx):
            continue
        key = parts[key_idx].strip()
        pop_raw = parts[pop_idx].strip()
        if not key or not pop_raw or pop_raw in ("*", "-"):
            continue
        try:
            pop = float(pop_raw)
        except ValueError:
            continue
        rows.append((key[:8], pop))  # 先頭8桁 = 親の3次メッシュコード
    df = pl.DataFrame(rows, schema=["mesh_code", "population_500m"], orient="row")
    return df.group_by("mesh_code").agg(pl.col("population_500m").sum().alias("population_2020"))


# load_osm_layers()（上のosmiumハンドラ）が道路・土地利用・河川を1回のPBF走査でまとめて返す。
# 個別のload_roads/load_landuse/load_waterwayは廃止し、main()でハンドラの属性を直接参照する。


RESIDENTIAL_TAGS = {"residential", "commercial", "industrial", "retail", "construction"}
FOREST_TAGS = {"forest", "wood"}


def main() -> None:
    codes = build_grid()
    grid = pl.DataFrame(
        {
            "mesh_code": codes,
        }
    ).with_columns(
        pl.col("mesh_code")
        .map_elements(lambda c: jis_3rd_mesh_center(c)[0], return_dtype=pl.Float64)
        .alias("lat_center"),
        pl.col("mesh_code")
        .map_elements(lambda c: jis_3rd_mesh_center(c)[1], return_dtype=pl.Float64)
        .alias("lon_center"),
    )
    print(f"grid cells: {grid.height}")

    # --- 人口 ---
    pop = load_population()
    grid = grid.join(pop, on="mesh_code", how="left").with_columns(pl.col("population_2020").fill_null(0.0))
    print(f"population joined: {grid.filter(pl.col('population_2020') > 0).height} nonzero cells")

    # --- OSMレイヤ（道路・土地利用・河川）をPBFから1回の走査で取得 ---
    print("parsing OSM PBF (roads, landuse, waterway)...")
    osm = load_osm_layers()
    road_lines, road_tags = osm.road_lines, osm.road_tags
    landuse_polys, landuse_tags = osm.landuse_polys, osm.landuse_tags
    water_lines = osm.water_lines

    # --- 道路 ---
    if road_lines:
        road_union = unary_union(road_lines)
        road_tree = STRtree(road_lines)
        lens, dists = [], []
        for lat, lon in zip(grid.get_column("lat_center").to_list(), grid.get_column("lon_center").to_list()):
            pt = Point(lon, lat)
            s, w, n, e = jis_3rd_mesh_bounds(jis_3rd_mesh_code(lat, lon))
            cell = Polygon([(w, s), (e, s), (e, n), (w, n)])
            idxs = road_tree.query(cell)
            length_deg = sum(road_lines[i].intersection(cell).length for i in idxs)
            # 度単位の長さをメートルに概算変換（緯度43度基準）
            m_per_deg = 111000.0
            lens.append(length_deg * m_per_deg)
            nearest_idx = road_tree.nearest(pt)
            dists.append(pt.distance(road_lines[nearest_idx]) * m_per_deg)
        grid = grid.with_columns(pl.Series("road_len_major_m", lens), pl.Series("dist_to_road_m", dists))
    else:
        grid = grid.with_columns(pl.lit(None).alias("road_len_major_m"), pl.lit(None).alias("dist_to_road_m"))
    print(f"roads: {len(road_lines)} ways loaded")

    # --- 土地利用 ---
    if landuse_polys:
        residential_polys = [p for p, t in zip(landuse_polys, landuse_tags) if t in RESIDENTIAL_TAGS]
        forest_polys = [p for p, t in zip(landuse_polys, landuse_tags) if t in FOREST_TAGS]
        residential_union = unary_union(residential_polys) if residential_polys else None
        forest_union = unary_union(forest_polys) if forest_polys else None
        forest_tree = STRtree(forest_polys) if forest_polys else None

        res_ratios, forest_ratios, forest_dists = [], [], []
        for lat, lon in zip(grid.get_column("lat_center").to_list(), grid.get_column("lon_center").to_list()):
            s, w, n, e = jis_3rd_mesh_bounds(jis_3rd_mesh_code(lat, lon))
            cell = Polygon([(w, s), (e, s), (e, n), (w, n)])
            cell_area = cell.area
            res_ratio = (residential_union.intersection(cell).area / cell_area) if residential_union else 0.0
            forest_ratio = (forest_union.intersection(cell).area / cell_area) if forest_union else 0.0
            res_ratios.append(res_ratio)
            forest_ratios.append(forest_ratio)
            if forest_tree is not None and len(forest_polys) > 0:
                pt = Point(lon, lat)
                nearest_idx = forest_tree.nearest(pt)
                d = pt.distance(forest_polys[nearest_idx]) * 111000.0
                forest_dists.append(d)
            else:
                forest_dists.append(None)
        grid = grid.with_columns(
            pl.Series("landuse_residential_ratio", res_ratios),
            pl.Series("landuse_forest_ratio", forest_ratios),
            pl.Series("dist_to_forest_edge_m", forest_dists),
        )
    else:
        grid = grid.with_columns(
            pl.lit(None).alias("landuse_residential_ratio"),
            pl.lit(None).alias("landuse_forest_ratio"),
            pl.lit(None).alias("dist_to_forest_edge_m"),
        )
    print(f"landuse: {len(landuse_polys)} polygons loaded")

    # --- 河川 ---
    if water_lines:
        water_tree = STRtree(water_lines)
        river_dists = []
        for lat, lon in zip(grid.get_column("lat_center").to_list(), grid.get_column("lon_center").to_list()):
            pt = Point(lon, lat)
            nearest_idx = water_tree.nearest(pt)
            river_dists.append(pt.distance(water_lines[nearest_idx]) * 111000.0)
        grid = grid.with_columns(pl.Series("dist_to_river_m", river_dists))
    else:
        grid = grid.with_columns(pl.lit(None).alias("dist_to_river_m"))
    print(f"waterway: {len(water_lines)} ways loaded")

    # --- 標高・傾斜（DEM） ---
    dem_tiles = load_dem_tiles()
    if dem_tiles:
        elevations, slopes = [], []
        for mesh_code, lat, lon in zip(
            grid.get_column("mesh_code").to_list(),
            grid.get_column("lat_center").to_list(),
            grid.get_column("lon_center").to_list(),
        ):
            bounds = jis_3rd_mesh_bounds(mesh_code)
            elev, slope = sample_elevation_slope(lat, lon, bounds, dem_tiles)
            elevations.append(elev)
            slopes.append(slope)
        grid = grid.with_columns(pl.Series("elevation_mean_m", elevations), pl.Series("slope_mean_deg", slopes))
    else:
        grid = grid.with_columns(pl.lit(None).alias("elevation_mean_m"), pl.lit(None).alias("slope_mean_deg"))
    print(f"DEM: {len(dem_tiles)} tiles loaded")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.write_parquet(OUT_PATH)
    print(f"written {grid.height} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
