"""PLATEAU DEM（TIN）から解析域内の頂点群を収集（経緯度・標高）。"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def iter_dem_triangles_lonlat(path: Path):
    for event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "Triangle":
            continue
        pos_text = None
        for node in elem.iter():
            if _local(node.tag) == "posList" and node.text:
                pos_text = node.text
                break
        elem.clear()
        if not pos_text:
            continue
        try:
            arr = np.array([float(x) for x in pos_text.split()], dtype=np.float64).reshape(-1, 3)
        except ValueError:
            continue
        if len(arr) < 3:
            continue
        yield arr[0], arr[1], arr[2]


def load_dem_vertices_ll(
    dem_paths: list[Path],
    bbox_ll: Polygon,
    pad_deg: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    bbox_buf = bbox_ll.buffer(pad_deg)
    lons: list[float] = []
    lats: list[float] = []
    zs: list[float] = []
    key_to_i: dict[tuple[int, int], int] = {}

    def vertex_index(lat: float, lon: float, z: float) -> int:
        key = (int(round(lat * 1e7)), int(round(lon * 1e7)))
        if key in key_to_i:
            return key_to_i[key]
        idx = len(lons)
        key_to_i[key] = idx
        lons.append(float(lon))
        lats.append(float(lat))
        zs.append(float(z))
        return idx

    n_tris = 0
    for path in dem_paths:
        for a, b, c in iter_dem_triangles_lonlat(path):
            tri_ll = Polygon([(a[1], a[0]), (b[1], b[0]), (c[1], c[0])])
            if not tri_ll.intersects(bbox_buf):
                continue
            vertex_index(a[0], a[1], a[2])
            vertex_index(b[0], b[1], b[2])
            vertex_index(c[0], c[1], c[2])
            n_tris += 1

    if n_tris < 10 or len(lons) < 10:
        return None, None, None
    return (
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
        np.asarray(zs, dtype=np.float64),
    )
