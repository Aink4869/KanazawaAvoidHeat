"""日陰グリッド・DEM 標高グリッドと OSM 歩道ネットワークから、複数要素のコストで最短経路を求める。"""
from __future__ import annotations

import errno
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import LineString


@dataclass
class RouteFactors:
    """ルートのコストに入れる要素。チェックを外した項は 0 扱い。"""

    use_distance: bool = True
    use_time: bool = True
    use_slope: bool = True
    use_shadow: bool = True
    include_tree_shade: bool = False
    weight_distance: float = 1.0
    weight_time: float = 1.0
    weight_slope: float = 3.0
    weight_shadow: float = 4.0
    walk_speed_m_s: float = 1.2


SHADOW_MODEL_FLAT = "flat"
SHADOW_MODEL_GEOMETRY = "geometry"
SHADOW_MODEL_GEOMETRY_TREES = "geometry_trees"

_SHADOW_MODEL_FILES = {
    SHADOW_MODEL_FLAT: "summer_shadow_{year}",
    SHADOW_MODEL_GEOMETRY: "summer_shadow_geometry_{year}",
    SHADOW_MODEL_GEOMETRY_TREES: "summer_shadow_geometry_trees_{year}",
}


@dataclass
class ShadowGrid:
    masks: np.ndarray  # (T, ny, nx) uint8
    elev_m: np.ndarray | None  # (ny, nx) セル中心標高 [m]、無ければ None
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    nx: int
    ny: int
    year: int
    date_start: str
    date_end: str
    hours: list[int]
    model: str = SHADOW_MODEL_FLAT

    @property
    def transform(self):
        return from_bounds(self.xmin, self.ymin, self.xmax, self.ymax, self.nx, self.ny)


def resolve_shadow_model(data_dir: Path, year: int, model: str | None = None) -> str:
    """利用可能な日陰モデルを決定。model 未指定時は環境変数 SHADOW_MODEL → 幾何 NPZ 優先。"""
    if model is not None:
        chosen = model.strip().lower()
    else:
        chosen = os.environ.get("SHADOW_MODEL", SHADOW_MODEL_GEOMETRY).strip().lower()
    if chosen in ("auto", "default", ""):
        chosen = SHADOW_MODEL_GEOMETRY
    if chosen not in _SHADOW_MODEL_FILES:
        raise ValueError(
            f"SHADOW_MODEL は {list(_SHADOW_MODEL_FILES)} のいずれかにしてください（指定: {chosen}）"
        )
    if chosen == SHADOW_MODEL_FLAT:
        return chosen
    stem = _SHADOW_MODEL_FILES[chosen].format(year=year)
    if (data_dir / f"{stem}.npz").is_file():
        return chosen
    if chosen == SHADOW_MODEL_GEOMETRY:
        trees_stem = _SHADOW_MODEL_FILES[SHADOW_MODEL_GEOMETRY_TREES].format(year=year)
        if (data_dir / f"{trees_stem}.npz").is_file():
            return SHADOW_MODEL_GEOMETRY_TREES
    flat_path = data_dir / f"{_SHADOW_MODEL_FILES[SHADOW_MODEL_FLAT].format(year=year)}.npz"
    if flat_path.is_file():
        return SHADOW_MODEL_FLAT
    raise FileNotFoundError(
        f"日陰 NPZ がありません。compute_geometry_shadow_npz.py または compute_summer_shadows.py を実行してください "
        f"（data_dir={data_dir}, year={year}, requested={chosen}）"
    )


def shadow_grid_paths(data_dir: Path, year: int, model: str) -> tuple[Path, Path]:
    stem = _SHADOW_MODEL_FILES[model].format(year=year)
    return data_dir / f"{stem}.npz", data_dir / f"{stem}_meta.json"


def load_shadow_grid(data_dir: Path, year: int = 2026, model: str | None = None) -> ShadowGrid:
    resolved = resolve_shadow_model(data_dir, year, model)
    path, meta_path = shadow_grid_paths(data_dir, year, resolved)
    if not path.is_file():
        scripts = {
            SHADOW_MODEL_FLAT: "compute_summer_shadows.py",
            SHADOW_MODEL_GEOMETRY: "compute_geometry_shadow_npz.py",
            SHADOW_MODEL_GEOMETRY_TREES: "compute_geometry_shadow_npz.py --with-trees",
        }
        raise FileNotFoundError(f"先に {scripts[resolved]} を実行してください: {path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(path)
    elev = z["elev_m"] if "elev_m" in z.files else None
    return ShadowGrid(
        masks=z["masks"],
        elev_m=elev,
        xmin=float(z["xmin"]),
        xmax=float(z["xmax"]),
        ymin=float(z["ymin"]),
        ymax=float(z["ymax"]),
        nx=int(z["nx"]),
        ny=int(z["ny"]),
        year=int(meta["year"]),
        date_start=meta["date_start"],
        date_end=meta.get("date_end", meta["date_start"]),
        hours=list(meta["hours_local"]),
        model=str(meta.get("model", resolved)),
    )


def time_index(grid: ShadowGrid, when: date, hour_local: int) -> int:
    if hour_local not in grid.hours:
        hour_local = min(grid.hours, key=lambda h: abs(h - hour_local))
    try:
        base = date.fromisoformat(grid.date_start)
        end = date.fromisoformat(grid.date_end)
    except Exception:
        base = end = when
    # 任意の日付入力を許容し、キャッシュデータの範囲内に安全にフォールバック
    if when < base:
        when = base
    elif when > end:
        when = end
    di = (when - base).days
    hi = grid.hours.index(hour_local)
    ti = di * len(grid.hours) + hi
    ti = max(0, min(ti, grid.masks.shape[0] - 1))
    return ti


def sample_edge_shadow_fraction(
    mask: np.ndarray,
    transform,
    ny: int,
    nx: int,
    line: LineString,
    n_samples: int = 8,
) -> float:
    if line is None or line.is_empty:
        return 0.0
    vals: list[float] = []
    for i in range(n_samples):
        pt = line.interpolate((i + 0.5) / n_samples, normalized=True)
        try:
            r, c = rowcol(transform, pt.x, pt.y)
        except Exception:
            continue
        if 0 <= r < ny and 0 <= c < nx:
            vals.append(float(mask[r, c]))
    return float(np.mean(vals)) if vals else 0.0


def _node_elev(
    G: nx.MultiDiGraph,
    n: int,
    elev_m: np.ndarray | None,
    tf,
    ny: int,
    nx: int,
) -> float:
    """ノード位置の標高。グリッド外・欠損は nan。"""
    if elev_m is None:
        return float("nan")
    nd = G.nodes[n]
    x, y = float(nd["x"]), float(nd["y"])
    try:
        r, c = rowcol(tf, x, y)
    except Exception:
        return float("nan")
    if 0 <= r < ny and 0 <= c < nx:
        v = float(elev_m[r, c])
        if np.isfinite(v):
            return v
    return float("nan")


def prepare_graph_nav_costs(
    G_proj: nx.MultiDiGraph,
    mask: np.ndarray,
    grid: ShadowGrid,
    factors: RouteFactors,
) -> nx.MultiDiGraph:
    """
    辺コスト（加算モデル）::

        cost = ε·L
             + w_d·L                    （距離、ON 時）
             + w_t·(L / v_walk)         （時間相当、ON 時）
             + w_s·L·|勾配|            （勾配、ON かつ DEM あり）
             + w_sh·L·日陰割合         （日陰、ON 時）

    勾配はノード標高差 / 辺長（おおよそ m/m）。ε は全要素オフでも連結性を保つ微小項。
    """
    if not any(
        [
            factors.use_distance,
            factors.use_time,
            factors.use_slope,
            factors.use_shadow,
        ]
    ):
        raise ValueError("少なくとも1つの要素（距離・時間・勾配・日陰）を有効にしてください")

    tf = grid.transform
    ny, nx = grid.ny, grid.nx
    eps = 1e-9

    for _u, _v, _k, data in G_proj.edges(keys=True, data=True):
        geom = data.get("geometry")
        length = float(data.get("length", 1.0))
        if length < 1e-6:
            length = 1e-6

        if geom is not None:
            frac = sample_edge_shadow_fraction(mask, tf, ny, nx, geom)
        else:
            n1, n2 = G_proj.nodes[_u], G_proj.nodes[_v]
            ls = LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])])
            frac = sample_edge_shadow_fraction(mask, tf, ny, nx, ls)

        zu = _node_elev(G_proj, _u, grid.elev_m, tf, ny, nx)
        zv = _node_elev(G_proj, _v, grid.elev_m, tf, ny, nx)
        if (
            factors.use_slope
            and grid.elev_m is not None
            and np.isfinite(zu)
            and np.isfinite(zv)
        ):
            grade = abs(float(zv - zu) / length)
        else:
            grade = 0.0

        term_d = factors.weight_distance * length if factors.use_distance else 0.0
        term_t = (
            factors.weight_time * (length / factors.walk_speed_m_s)
            if factors.use_time
            else 0.0
        )
        term_s = factors.weight_slope * length * grade if factors.use_slope else 0.0
        term_sh = factors.weight_shadow * length * frac if factors.use_shadow else 0.0

        nav_cost = eps * length + term_d + term_t + term_s + term_sh
        data["shadow_fraction"] = frac
        data["slope_grade_abs"] = grade
        data["nav_cost"] = max(1e-12, nav_cost)

    return G_proj


def nearest_node(G: nx.MultiDiGraph, x: float, y: float):
    return ox.distance.nearest_nodes(G, x, y)


def _path_edge_attr(G: nx.MultiDiGraph, path: list, attr: str) -> list[float]:
    vals = []
    for u, v in zip(path[:-1], path[1:]):
        if u not in G or v not in G[u]:
            continue
        best = min((float(d.get(attr, 0.0)) for d in G[u][v].values()), default=0.0)
        vals.append(best)
    return vals


def resolve_route_mask(
    grid: ShadowGrid,
    when: date,
    hour_local: int,
    *,
    data_dir: Path | None = None,
    allow_flat_fallback: bool = True,
) -> tuple[np.ndarray, ShadowGrid, int, bool]:
    """ルート用マスク。幾何 NPZ の日付範囲外なら平坦地 NPZ にフォールバック可能。"""
    try:
        ti = time_index(grid, when, hour_local)
        return grid.masks[ti], grid, ti, False
    except ValueError as e:
        if allow_flat_fallback and grid.model != SHADOW_MODEL_FLAT and data_dir is not None:
            try:
                flat = load_shadow_grid(data_dir, grid.year, SHADOW_MODEL_FLAT)
                ti = time_index(flat, when, hour_local)
                return flat.masks[ti], flat, ti, True
            except (ValueError, FileNotFoundError):
                pass
        raise e


def route_shadow_aware(
    grid: ShadowGrid,
    when: date,
    hour_local: int,
    orig_xy: tuple[float, float],
    dest_xy: tuple[float, float],
    *,
    factors: RouteFactors | None = None,
    network_type: str = "walk",
    data_dir: Path | None = None,
    allow_flat_fallback: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], dict]:
    if factors is None:
        factors = RouteFactors()

    shadow_fallback = False
    try:
        mask, active_grid, ti, shadow_fallback = resolve_route_mask(
            grid,
            when,
            hour_local,
            data_dir=data_dir,
            allow_flat_fallback=allow_flat_fallback,
        )
    except ValueError:
        raise

    from nav_tree_shade import apply_tree_residual

    mask, residual, tree_info = apply_tree_residual(
        mask,
        active_grid,
        when,
        hour_local,
        include_tree_shade=factors.include_tree_shade,
        root=Path(__file__).resolve().parent if data_dir is None else Path(data_dir).parent,
    )

    to_wgs = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    olon, olat = to_wgs.transform(orig_xy[0], orig_xy[1])
    dlon, dlat = to_wgs.transform(dest_xy[0], dest_xy[1])
    pad = 0.008
    lat_max, lat_min = max(olat, dlat) + pad, min(olat, dlat) - pad
    lon_max, lon_min = max(olon, dlon) + pad, min(olon, dlon) - pad

    try:
        G = ox.graph_from_bbox(
            bbox=(lon_min, lat_min, lon_max, lat_max),
            network_type=network_type,
        )
    except OSError as e:
        # Windows 環境で OSMnx キャッシュ書き込みが失敗する場合があるため、
        # そのときだけキャッシュ無効で再取得する。
        if e.errno == errno.EINVAL and "cache" in str(e).lower():
            old_cache = ox.settings.use_cache
            ox.settings.use_cache = False
            try:
                G = ox.graph_from_bbox(
                    bbox=(lon_min, lat_min, lon_max, lat_max),
                    network_type=network_type,
                )
            finally:
                ox.settings.use_cache = old_cache
        else:
            raise
    Gp = ox.project_graph(G, to_crs="EPSG:6670")

    on = nearest_node(Gp, orig_xy[0], orig_xy[1])
    dn = nearest_node(Gp, dest_xy[0], dest_xy[1])

    Gp = prepare_graph_nav_costs(Gp, mask, active_grid, factors)
    path = nx.shortest_path(Gp, on, dn, weight="nav_cost")
    coords_xy = [(float(Gp.nodes[n]["x"]), float(Gp.nodes[n]["y"])) for n in path]

    lengths = _path_edge_attr(Gp, path, "length")
    fracs = _path_edge_attr(Gp, path, "shadow_fraction")
    grades = _path_edge_attr(Gp, path, "slope_grade_abs")
    nav_lens = _path_edge_attr(Gp, path, "nav_cost")

    res_fracs: list[float] = []
    if tree_info.get("tree_shade_applied") and np.any(residual):
        tf = active_grid.transform
        ny_g, nx_g = active_grid.ny, active_grid.nx
        for u, v in zip(path[:-1], path[1:]):
            if u not in Gp or v not in Gp[u]:
                continue
            data = min(Gp[u][v].values(), key=lambda d: float(d.get("nav_cost", 1e18)))
            geom = data.get("geometry")
            if geom is None:
                n1, n2 = Gp.nodes[u], Gp.nodes[v]
                geom = LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])])
            res_fracs.append(sample_edge_shadow_fraction(residual, tf, ny_g, nx_g, geom))

    stats = {
        "timestep_index": int(ti),
        "shadow_model": (
            f"{active_grid.model}+tree_residual" if tree_info.get("tree_shade_applied") else active_grid.model
        ),
        "shadow_fallback_to_flat": shadow_fallback,
        "path_edges": len(path) - 1,
        "path_length_geometric_m": float(sum(lengths)),
        "path_cost_total": float(sum(nav_lens)),
        "mean_shadow_fraction_on_path": float(np.mean(fracs)) if fracs else 0.0,
        "mean_tree_residual_fraction_on_path": float(np.mean(res_fracs)) if res_fracs else 0.0,
        "mean_slope_grade_abs": float(np.mean(grades)) if grades else 0.0,
        "approx_time_s": float(sum(lengths)) / factors.walk_speed_m_s,
        "factors": {
            "use_distance": factors.use_distance,
            "use_time": factors.use_time,
            "use_slope": factors.use_slope,
            "use_shadow": factors.use_shadow,
            "include_tree_shade": factors.include_tree_shade,
            "weight_distance": factors.weight_distance,
            "weight_time": factors.weight_time,
            "weight_slope": factors.weight_slope,
            "weight_shadow": factors.weight_shadow,
            "walk_speed_m_s": factors.walk_speed_m_s,
        },
        "elev_grid_available": active_grid.elev_m is not None,
        "tree_shade": tree_info,
    }

    to_ll = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    coords_ll = []
    for x, y in coords_xy:
        lo, la = to_ll.transform(x, y)
        coords_ll.append((la, lo))
    return coords_xy, coords_ll, stats
