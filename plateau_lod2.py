"""PLATEAU Building lod2Solid 面（Wall/Roof 等）を EPSG:6670 の 3D リングとして読み込む。"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box

from plateau_load import kanazawa_study_bbox

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
GML_ID = "{http://www.opengis.net/gml}id"


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_poslist_llh(text: str) -> np.ndarray:
    vals = np.array([float(x) for x in text.split()], dtype=np.float64)
    return vals.reshape(-1, 3)  # lat, lon, height


@dataclass
class Lod2Building:
    """1 棟ぶんの LOD2 面リング（EPSG:6670, shape (n,3)）。"""
    surface_rings: list[np.ndarray]
    footprint_xy: Polygon | None = None


def _polygons_by_id(building_elem) -> dict[str, np.ndarray]:
    """Building 要素内の gml:Polygon id → lat/lon/h リング。"""
    out: dict[str, np.ndarray] = {}
    for node in building_elem.iter():
        if _local(node.tag) != "Polygon":
            continue
        gid = node.attrib.get(GML_ID)
        if not gid:
            continue
        for sub in node.iter():
            if _local(sub.tag) == "posList" and sub.text:
                try:
                    out[gid] = _parse_poslist_llh(sub.text)
                except ValueError:
                    pass
                break
    return out


def _lod2_hrefs(building_elem) -> list[str]:
    hrefs: list[str] = []
    for node in building_elem.iter():
        if _local(node.tag) != "lod2Solid":
            continue
        for sm in node.iter():
            if _local(sm.tag) != "surfaceMember":
                continue
            href = sm.attrib.get(XLINK_HREF)
            if href:
                hrefs.append(href.lstrip("#"))
    return hrefs


def _footprint_ll(building_elem) -> Polygon | None:
    for node in building_elem.iter():
        if _local(node.tag) != "lod0FootPrint":
            continue
        for sub in node.iter():
            if _local(sub.tag) == "posList" and sub.text:
                pts = _parse_poslist_llh(sub.text)
                return Polygon([(p[1], p[0]) for p in pts])
    return None


def iter_lod2_buildings_gml(path: Path):
    """GML 1 ファイルから LOD2 面を持つ建物を yield。"""
    for _event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "Building":
            continue
        hrefs = _lod2_hrefs(elem)
        if not hrefs:
            elem.clear()
            continue
        poly_map = _polygons_by_id(elem)
        fp = _footprint_ll(elem)
        elem.clear()
        if not poly_map:
            continue
        rings_llh = [poly_map[h] for h in hrefs if h in poly_map]
        if not rings_llh:
            continue
        yield rings_llh, fp


def load_lod2_buildings_projected(
    data_dir: Path,
    bbox_ll: Polygon,
    transformer: Transformer,
) -> list[Lod2Building]:
    """解析域と交わる建物の LOD2 面を EPSG:6670 に変換して返す。"""
    out: list[Lod2Building] = []
    for gml in sorted(data_dir.glob("*_bldg_*_op.gml")):
        for rings_llh, fp in iter_lod2_buildings_gml(gml):
            if fp is not None and not fp.intersects(bbox_ll):
                continue
            if fp is None:
                # footprint が無い場合は最初の面の範囲で粗く判定
                pts = rings_llh[0]
                if not box(pts[:, 1].min(), pts[:, 0].min(), pts[:, 1].max(), pts[:, 0].max()).intersects(bbox_ll):
                    continue
            surface_rings: list[np.ndarray] = []
            for ring in rings_llh:
                if ring.shape[0] < 3:
                    continue
                lon = ring[:, 1]
                lat = ring[:, 0]
                z = ring[:, 2]
                xs, ys = transformer.transform(lon, lat)
                xyz = np.column_stack([xs, ys, z])
                surface_rings.append(xyz)
            fp_xy: Polygon | None = None
            if fp is not None and not fp.is_empty:
                try:
                    xs_f, ys_f = transformer.transform(
                        np.asarray(fp.exterior.xy[0], dtype=np.float64),
                        np.asarray(fp.exterior.xy[1], dtype=np.float64),
                    )
                    fp_xy = Polygon(np.column_stack([xs_f, ys_f]))
                    if not fp_xy.is_valid:
                        fp_xy = fp_xy.buffer(0)
                except Exception:
                    fp_xy = None
            if surface_rings:
                out.append(Lod2Building(surface_rings=surface_rings, footprint_xy=fp_xy))
    return out
