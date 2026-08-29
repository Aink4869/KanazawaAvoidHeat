"""PLATEAU LOD2 面ポリゴンから日影（2D 投影）を構築。"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import shadow_geometry as sg
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from plateau_lod2 import Lod2Building


def surface_ring_shadow(
    ring_xyz: np.ndarray,
    sun_unit: np.ndarray,
    terrain: sg.TerrainFn,
) -> Polygon | MultiPolygon:
    """
    3D 面の外周リング各頂点から光線を地形へ延ばし、着地点を順に結んだ影ポリゴン。
    傾斜屋根・立面（PLATEAU LOD2）向け。
    """
    if ring_xyz.shape[0] < 3:
        return Polygon()
    ring = ring_xyz
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        return Polygon()

    pts: list[tuple[float, float]] = []
    for x, y, z in ring:
        hit = sg.sun_ray_intersect_terrain(float(x), float(y), float(z), sun_unit, terrain)
        if hit is None:
            return Polygon()
        pts.append(hit)
    if len(pts) < 3:
        return Polygon()
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-8:
            return Polygon()
        return poly
    except Exception:
        return Polygon()


def lod2_building_shadow_union(
    building: Lod2Building,
    sun_unit: np.ndarray,
    terrain: sg.TerrainFn,
    study,
) -> Polygon | MultiPolygon:
    parts = []
    for ring in building.surface_rings:
        sh = surface_ring_shadow(ring, sun_unit, terrain)
        if sh.is_empty:
            continue
        if sh.geom_type == "MultiPolygon":
            parts.extend(list(sh.geoms))
        else:
            parts.append(sh)
    if not parts:
        return Polygon()
    u = unary_union(parts).intersection(study)
    return u if not u.is_empty else Polygon()


def lod2_shade_union_all(
    buildings: Iterable[Lod2Building],
    sun_unit: np.ndarray,
    terrain: sg.TerrainFn,
    study,
) -> Polygon | MultiPolygon:
    parts = []
    for b in buildings:
        sh = lod2_building_shadow_union(b, sun_unit, terrain, study)
        if sh.is_empty:
            continue
        if sh.geom_type == "MultiPolygon":
            parts.extend(list(sh.geoms))
        else:
            parts.append(sh)
    if not parts:
        return Polygon()
    return unary_union(parts)


def _lod0_footprint_has_lod2(
    footprint: Polygon,
    lod2_fps: list[Polygon],
    tree: STRtree | None,
    *,
    iou_thr: float = 0.15,
) -> bool:
    """LOD0 footprint が LOD2 建物と十分重なるか。"""
    if tree is None or not lod2_fps or footprint.is_empty:
        return False
    try:
        cand = tree.query(footprint)
    except Exception:
        return False
    for idx in cand:
        other = lod2_fps[int(idx)]
        if other is None or other.is_empty:
            continue
        inter = footprint.intersection(other).area
        if inter <= 0:
            continue
        union = footprint.union(other).area
        if union > 0 and inter / union >= iou_thr:
            return True
        # 包含・大半重なりもヒットとみなす
        if inter >= 0.5 * min(footprint.area, other.area):
            return True
    return False


def hybrid_lod2_or_lod0_shade_union(
    lod2_buildings: list[Lod2Building],
    lod0_buildings,
    sun_unit: np.ndarray,
    terrain: sg.TerrainFn,
    study,
    *,
    include_flat_roof_fallback: bool = True,
    subdivisions_per_edge: int = 1,
) -> tuple[Polygon | MultiPolygon, dict]:
    """
    LOD2 壁面・屋根面がある建物は面投影、無い建物は従来の
    LOD0 フットプリント＋高さ（鉛直壁＋任意で水平屋根）＋DEM にフォールバック。
    """
    parts: list = []
    n_lod2 = 0
    for b in lod2_buildings:
        sh = lod2_building_shadow_union(b, sun_unit, terrain, study)
        if sh.is_empty:
            continue
        n_lod2 += 1
        if sh.geom_type == "MultiPolygon":
            parts.extend(list(sh.geoms))
        else:
            parts.append(sh)

    lod2_fps = [b.footprint_xy for b in lod2_buildings if b.footprint_xy is not None and not b.footprint_xy.is_empty]
    tree = STRtree(lod2_fps) if lod2_fps else None

    n_fallback = 0
    n_skipped = 0
    for bf in lod0_buildings:
        fp = bf.poly_xy
        if fp is None or fp.is_empty or not fp.intersects(study):
            continue
        if _lod0_footprint_has_lod2(fp, lod2_fps, tree):
            n_skipped += 1
            continue
        try:
            xs, ys = fp.exterior.xy
        except Exception:
            continue
        ring = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
        if ring.shape[0] < 4:
            continue
        zf = np.array([terrain(float(x), float(y)) for x, y in ring], dtype=np.float64)
        sh = sg.footprint_building_shadow(
            ring,
            zf,
            float(bf.height_m),
            sun_unit,
            terrain,
            subdivisions_per_edge=subdivisions_per_edge,
            include_flat_roof=include_flat_roof_fallback,
        )
        if sh.is_empty:
            continue
        n_fallback += 1
        if sh.geom_type == "MultiPolygon":
            parts.extend(list(sh.geoms))
        else:
            parts.append(sh)

    stats = {
        "lod2_buildings_with_shade": n_lod2,
        "lod0_fallback_buildings": n_fallback,
        "lod0_skipped_covered_by_lod2": n_skipped,
        "lod2_loaded": len(lod2_buildings),
        "lod0_loaded": len(lod0_buildings) if hasattr(lod0_buildings, "__len__") else -1,
    }
    if not parts:
        return Polygon(), stats
    u = unary_union(parts).intersection(study)
    return (u if not u.is_empty else Polygon()), stats
