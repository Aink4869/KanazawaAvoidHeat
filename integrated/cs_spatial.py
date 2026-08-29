"""クーリングシェルター位置（立ち寄り経路用）。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
CS_GEOJSON = REPO / "ジオコーディング" / "cooling_shelters.geojson"

_cs_catalog: list[CoolingShelterSite] | None = None
_cs_xy: np.ndarray | None = None
_cs_tree: cKDTree | None = None


@dataclass(frozen=True)
class CoolingShelterSite:
    cs_id: int
    name: str
    address: str
    x: float
    y: float
    lon: float
    lat: float


def _load_cs_geojson() -> dict:
    text = CS_GEOJSON.read_text(encoding="utf-8")
    text = re.sub(r"\bNaN\b", "null", text)
    return json.loads(text)


def load_cs_catalog() -> list[CoolingShelterSite]:
    global _cs_catalog, _cs_xy, _cs_tree
    if _cs_catalog is not None:
        return _cs_catalog
    if not CS_GEOJSON.is_file():
        raise FileNotFoundError(f"GeoJSON がありません: {CS_GEOJSON}")
    geo = _load_cs_geojson()
    tf = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    sites: list[CoolingShelterSite] = []
    for feat in geo.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        x, y = tf.transform(lon, lat)
        p = feat.get("properties") or {}
        sites.append(
            CoolingShelterSite(
                cs_id=int(p.get("cs_id", len(sites) + 1)),
                name=str(p.get("cs_name", "")),
                address=str(p.get("address", "")),
                x=float(x),
                y=float(y),
                lon=float(lon),
                lat=float(lat),
            )
        )
    if not sites:
        raise ValueError("クーリングシェルターの座標がありません")
    _cs_catalog = sites
    _cs_xy = np.asarray([(s.x, s.y) for s in sites], dtype=np.float64)
    _cs_tree = cKDTree(_cs_xy)
    return _cs_catalog


def get_cs_kdtree() -> tuple[cKDTree, np.ndarray]:
    load_cs_catalog()
    assert _cs_tree is not None and _cs_xy is not None
    return _cs_tree, _cs_xy


def get_cs_by_id(cs_id: int) -> CoolingShelterSite | None:
    for s in load_cs_catalog():
        if s.cs_id == cs_id:
            return s
    return None


def find_cs_near_xy(x: float, y: float, *, max_dist_m: float = 45.0) -> CoolingShelterSite | None:
    tree, _ = get_cs_kdtree()
    d, i = tree.query([x, y])
    if float(d) <= max_dist_m:
        return load_cs_catalog()[int(i)]
    return None


def candidate_visit_cs_indices(
    orig_xy: tuple[float, float],
    dest_xy: tuple[float, float],
    *,
    max_candidates: int = 8,
) -> list[int]:
    """起点・終点の折り返し距離が短い CS を立ち寄り候補にする。"""
    _, cs_xy = get_cs_kdtree()
    ox, oy = orig_xy
    dx, dy = dest_xy
    detour = (
        np.hypot(cs_xy[:, 0] - ox, cs_xy[:, 1] - oy)
        + np.hypot(cs_xy[:, 0] - dx, cs_xy[:, 1] - dy)
    )
    order = np.argsort(detour)
    return [int(i) for i in order[:max_candidates]]
