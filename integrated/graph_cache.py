"""OSM 歩道グラフ・コスト付きグラフのメモリ／ディスクキャッシュ。"""
from __future__ import annotations

import errno
import re
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox

REPO = Path(__file__).resolve().parent.parent
DISK_CACHE_DIR = REPO / "data" / "osm_walk_cache"

# bbox: (lon_min, lat_min, lon_max, lat_max) 丸めキー
_walk_graph_cache: dict[tuple[float, float, float, float], nx.MultiDiGraph] = {}
_prepared_cache: dict[tuple, nx.MultiDiGraph] = {}
_undirected_cache: dict[int, nx.Graph] = {}
_cs_nodes_cache: dict[tuple[float, float, float, float], list] = {}
_last_graph_source: str = "memory"

# ディスクキャッシュの一覧メタデータ (lon_min, lat_min, lon_max, lat_max, filepath)
_cached_disk_files: list[tuple[float, float, float, float, Path]] = []


def _scan_disk_cache() -> None:
    global _cached_disk_files
    if not DISK_CACHE_DIR.is_dir():
        return
    results = []
    pattern = re.compile(
        r"^walk_([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)\.graphml$"
    )
    for p in DISK_CACHE_DIR.glob("*.graphml"):
        m = pattern.match(p.name)
        if m:
            lomin, lamin, lomax, lamax = map(float, m.groups())
            results.append((lomin, lamin, lomax, lamax, p))
    # 広い bbox を優先
    results.sort(
        key=lambda item: (item[2] - item[0]) * (item[3] - item[1]), reverse=True
    )
    _cached_disk_files = results


_scan_disk_cache()


def _bbox_key(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> tuple[float, float, float, float]:
    return (
        round(lon_min, 4),
        round(lat_min, 4),
        round(lon_max, 4),
        round(lat_max, 4),
    )


def _disk_path(key: tuple[float, float, float, float]) -> Path:
    name = "walk_" + "_".join(f"{v:.4f}" for v in key) + ".graphml"
    return DISK_CACHE_DIR / name


def _find_covering_disk_graph(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> Path | None:
    """要求された bbox を内包する既存の .graphml キャッシュファイルを探す。"""
    for lomin, lamin, lomax, lamax, p in _cached_disk_files:
        if lomin <= lon_min and lamin <= lat_min and lomax >= lon_max and lamax >= lat_max:
            return p
    if _cached_disk_files:
        return _cached_disk_files[0][4]
    return None


def last_graph_load_source() -> str:
    """直近のグラフ取得元: memory / disk / network。"""
    return _last_graph_source


def _fetch_walk_graph(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    network_type: str = "walk",
) -> nx.MultiDiGraph:
    try:
        G = ox.graph_from_bbox(
            bbox=(lon_min, lat_min, lon_max, lat_max),
            network_type=network_type,
        )
    except OSError as e:
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
    return G


def get_walk_graph_projected(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    network_type: str = "walk",
) -> nx.MultiDiGraph:
    """EPSG:6670 投影済みグラフ（メモリ → ディスク完全一致 → ディスク内包フォールバック → Overpass の順で取得）。"""
    global _last_graph_source

    key = _bbox_key(lon_min, lat_min, lon_max, lat_max)
    if key in _walk_graph_cache:
        _last_graph_source = "memory"
        return _walk_graph_cache[key]

    disk = _disk_path(key)
    if disk.is_file():
        Gp = ox.load_graphml(disk)
        _walk_graph_cache[key] = Gp
        _last_graph_source = "disk"
        return Gp

    covering_disk = _find_covering_disk_graph(lon_min, lat_min, lon_max, lat_max)
    if covering_disk is not None and covering_disk.is_file():
        Gp = ox.load_graphml(covering_disk)
        _walk_graph_cache[key] = Gp
        _last_graph_source = "disk_fallback"
        return Gp

    try:
        G = _fetch_walk_graph(lon_min, lat_min, lon_max, lat_max, network_type=network_type)
        Gp = ox.project_graph(G, to_crs="EPSG:6670")
        _walk_graph_cache[key] = Gp
        _last_graph_source = "network"

        try:
            DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(Gp, disk)
            _scan_disk_cache()
        except OSError:
            pass

        return Gp
    except Exception as net_err:
        if _cached_disk_files:
            fallback_path = _cached_disk_files[0][4]
            Gp = ox.load_graphml(fallback_path)
            _walk_graph_cache[key] = Gp
            _last_graph_source = "disk_emergency_fallback"
            return Gp
        raise net_err


def warmup_bbox(
    lon_min: float = 136.642,
    lat_min: float = 36.548,
    lon_max: float = 136.678,
    lat_max: float = 36.582,
    *,
    network_type: str = "walk",
) -> None:
    """起動時プリロード用（金沢中心の典型 bbox）。"""
    get_walk_graph_projected(lon_min, lat_min, lon_max, lat_max, network_type=network_type)


def get_cs_snap_nodes(
    bbox_key: tuple[float, float, float, float],
    Gp: nx.MultiDiGraph,
    sites_xy: list[tuple[float, float]],
) -> list:
    """CS 座標をグラフ上の最近傍ノードへ一括スナップ（bbox 単位でキャッシュ）。"""
    if bbox_key in _cs_nodes_cache:
        return _cs_nodes_cache[bbox_key]

    xs = np.array([p[0] for p in sites_xy], dtype=np.float64)
    ys = np.array([p[1] for p in sites_xy], dtype=np.float64)
    nodes = ox.distance.nearest_nodes(Gp, xs, ys)
    out = [int(n) for n in np.atleast_1d(nodes)]

    _cs_nodes_cache[bbox_key] = out
    if len(_cs_nodes_cache) > 12:
        _cs_nodes_cache.pop(next(iter(_cs_nodes_cache)))
    return out


def cache_get_prepared(key: tuple) -> nx.MultiDiGraph | None:
    return _prepared_cache.get(key)


def cache_put_prepared(key: tuple, Gp: nx.MultiDiGraph) -> None:
    _undirected_cache.pop(id(Gp), None)
    if len(_prepared_cache) > 8:
        old = next(iter(_prepared_cache))
        _prepared_cache.pop(old)
    _prepared_cache[key] = Gp


def prepared_cache_key(
    bbox_key: tuple[float, float, float, float],
    timestep_index: int,
    factors_tuple: tuple[Any, ...],
) -> tuple:
    return (bbox_key, timestep_index, factors_tuple)


def get_undirected_cost_graph(Gp: nx.MultiDiGraph) -> nx.Graph:
    """双方向の nav_cost を1本の無向辺にまとめ（全源ダイクストラ用）。"""
    gid = id(Gp)
    if gid in _undirected_cache:
        return _undirected_cache[gid]

    Gu = nx.Graph()
    for u, v, _k, data in Gp.edges(keys=True, data=True):
        w = float(data.get("nav_cost", 1.0))
        if Gu.has_edge(u, v):
            if w < Gu[u][v]["weight"]:
                Gu[u][v]["weight"] = w
        else:
            Gu.add_edge(u, v, weight=w)

    _undirected_cache[gid] = Gu
    if len(_undirected_cache) > 16:
        _undirected_cache.pop(next(iter(_undirected_cache)))
    return Gu
