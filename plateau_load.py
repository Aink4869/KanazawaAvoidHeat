"""PLATEAU GML から解析バウンディングボックス内の建物 footprint（平面直角）を読み込む共通モジュール。"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


@dataclass
class BuildingFootprint:
    poly_xy: Polygon
    height_m: float


def _parse_poslist(text: str) -> np.ndarray:
    vals = np.array([float(x) for x in text.split()], dtype=np.float64)
    return vals.reshape(-1, 3)


def iter_buildings_gml(path: Path):
    for event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "Building":
            continue
        height = None
        for child in elem:
            if _local(child.tag) == "measuredHeight" and child.text:
                height = float(child.text.strip())
                break
        if height is None or height <= 0:
            elem.clear()
            continue
        lod0 = None
        for child in elem:
            if _local(child.tag) == "lod0FootPrint":
                lod0 = child
                break
        if lod0 is None:
            elem.clear()
            continue
        pos_text = None
        for node in lod0.iter():
            if _local(node.tag) == "posList" and node.text:
                pos_text = node.text
                break
        elem.clear()
        if not pos_text:
            continue
        try:
            pts = _parse_poslist(pos_text)
        except ValueError:
            continue
        if len(pts) < 3:
            continue
        yield pts, height


def iter_roads_gml(path: Path):
    """tran:Road の各 cityObject について、lod1 の全 posList から緯度経度ポリゴンを返す。"""
    for _event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "Road":
            continue
        polys_ll: list[Polygon] = []
        for node in elem.iter():
            if _local(node.tag) != "posList" or not node.text:
                continue
            try:
                pts = _parse_poslist(node.text)
            except ValueError:
                continue
            if len(pts) < 3:
                continue
            try:
                polys_ll.append(Polygon([(p[1], p[0]) for p in pts]))
            except Exception:
                continue
        elem.clear()
        if not polys_ll:
            continue
        try:
            g = unary_union(polys_ll)
        except Exception:
            continue
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            yield g
        elif g.geom_type == "MultiPolygon":
            for pg in g.geoms:
                if pg.geom_type == "Polygon" and not pg.is_empty:
                    yield pg


def load_roads_projected(
    data_dir: Path,
    bbox_ll: Polygon,
    transformer: Transformer,
) -> list[Polygon]:
    """EPSG:6697 相当の緯経度 bbox と交わる道路面を EPSG:6670 のポリゴンに変換。"""
    out: list[Polygon] = []
    for gml in sorted(data_dir.glob("*_tran_*_op.gml")):
        for poly_ll in iter_roads_gml(gml):
            if not poly_ll.intersects(bbox_ll):
                continue
            try:
                coords = np.asarray(poly_ll.exterior.coords, dtype=np.float64)
                xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
            except Exception:
                continue
            try:
                p = Polygon(np.column_stack([xs, ys]))
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty or p.area <= 0:
                    continue
                out.append(p)
            except Exception:
                continue
    return out


def load_buildings_projected(
    data_dir: Path,
    bbox_ll: Polygon,
    transformer: Transformer,
) -> list[BuildingFootprint]:
    """EPSG:6697 の bbox と交わる建物を EPSG:6670 のポリゴンに変換。"""
    out: list[BuildingFootprint] = []
    for gml in sorted(data_dir.glob("*_bldg_*_op.gml")):
        for pts, h in iter_buildings_gml(gml):
            poly_ll = Polygon([(p[1], p[0]) for p in pts])
            if not poly_ll.intersects(bbox_ll):
                continue
            lon = pts[:, 1]
            lat = pts[:, 0]
            xs, ys = transformer.transform(lon, lat)
            ring = np.column_stack([xs, ys])
            try:
                p = Polygon(ring)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty or p.area <= 0:
                    continue
                out.append(BuildingFootprint(poly_xy=p, height_m=float(h)))
            except Exception:
                continue
    return out


def kanazawa_study_bbox(buf_deg: float = 0.0022) -> tuple[Polygon, float, float, float, float]:
    """近江町〜兼六園周辺の (bbox_ll, lat_min, lat_max, lon_min, lon_max)。"""
    omicho_lat, omicho_lon = 36.57125, 136.65694
    kenrokuen_lat, kenrokuen_lon = 36.56204, 136.66245
    lat_min = min(omicho_lat, kenrokuen_lat) - buf_deg
    lat_max = max(omicho_lat, kenrokuen_lat) + buf_deg
    lon_min = min(omicho_lon, kenrokuen_lon) - buf_deg
    lon_max = max(omicho_lon, kenrokuen_lon) + buf_deg
    return box(lon_min, lat_min, lon_max, lat_max), lat_min, lat_max, lon_min, lon_max
