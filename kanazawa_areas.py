# -*- coding: utf-8 -*-
"""
金沢・道路日陰比較用のエリア定義。

近江町市場: 指定住所の凸包＋バッファ（円形ではない）。
金沢城公園: 指定住所の凸包＋バッファ。
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer
from rasterio import features
from shapely.geometry import MultiPoint, Polygon, mapping
from shapely.ops import transform as shp_transform

# 近江町市場周辺 — 代表住所（緯度, 経度）
OMICHO_MARKET_ANCHORS_LL: list[tuple[float, float, str]] = [
    (36.57106, 136.65726, "金沢市上近江町"),
    (36.57061, 136.66135, "金沢市尾張町"),
    (36.56781, 136.66498, "金沢市兼六元町11"),
    (36.57002, 136.65456, "金沢市高岡町3丁目"),
]
OMICHO_MARKET_BUFFER_M = 120.0

# 金沢城公園全体 — 代表住所（緯度, 経度）
CASTLE_PARK_ANCHORS_LL: list[tuple[float, float, str]] = [
    (36.56979, 136.65750, "金沢市西町4番"),
    (36.56825, 136.66182, "金沢市丸の内6-1"),
    (36.56576, 136.66351, "金沢市兼六元町1-1"),
    (36.55872, 136.66032, "金沢市本多町3丁目"),
    (36.56255, 136.65579, "金沢市広坂2丁目2"),
    (36.56741, 136.65553, "金沢市尾山町"),
]
CASTLE_PARK_BUFFER_M = 120.0


def _anchors_to_6670(anchors: list[tuple[float, float, str]], tr: Transformer) -> list[tuple[float, float]]:
    return [tr.transform(lon, lat) for lat, lon, _ in anchors]


def _polygon_from_anchors(anchors: list[tuple[float, float, str]], buffer_m: float,
                          tr: Transformer | None = None) -> Polygon:
    if tr is None:
        tr = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    pts = _anchors_to_6670(anchors, tr)
    return MultiPoint(pts).convex_hull.buffer(buffer_m)


def omicho_market_polygon_6670(tr: Transformer | None = None) -> Polygon:
    return _polygon_from_anchors(OMICHO_MARKET_ANCHORS_LL, OMICHO_MARKET_BUFFER_M, tr)


def castle_park_polygon_6670(tr: Transformer | None = None) -> Polygon:
    return _polygon_from_anchors(CASTLE_PARK_ANCHORS_LL, CASTLE_PARK_BUFFER_M, tr)


def castle_park_polygon_ll() -> Polygon:
    pts = [(lon, lat) for lat, lon, _ in CASTLE_PARK_ANCHORS_LL]
    hull = MultiPoint(pts).convex_hull
    tr_fwd = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    tr_back = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    hull_6670 = shp_transform(lambda x, y, z=None: tr_fwd.transform(x, y), hull)
    buffered = hull_6670.buffer(CASTLE_PARK_BUFFER_M)
    return shp_transform(lambda x, y, z=None: tr_back.transform(x, y), buffered)


def _mask_raster(poly: Polygon, transform, ny: int, nx: int) -> np.ndarray:
    return features.rasterize(
        [(poly, 1)], out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8,
    ) > 0


def omicho_mask_raster(transform, ny: int, nx: int, tr: Transformer | None = None) -> np.ndarray:
    return _mask_raster(omicho_market_polygon_6670(tr), transform, ny, nx)


def castle_mask_raster(transform, ny: int, nx: int, tr: Transformer | None = None) -> np.ndarray:
    return _mask_raster(castle_park_polygon_6670(tr), transform, ny, nx)


def _mask_from_xy(poly: Polygon, X, Y) -> np.ndarray:
    from shapely import contains, points

    pts = points(X.ravel(), Y.ravel())
    return contains(poly, pts).reshape(X.shape)


def omicho_mask_from_xy(X, Y, tr: Transformer | None = None) -> np.ndarray:
    return _mask_from_xy(omicho_market_polygon_6670(tr), X, Y)


def castle_mask_from_xy(X, Y, tr: Transformer | None = None) -> np.ndarray:
    return _mask_from_xy(castle_park_polygon_6670(tr), X, Y)


def _area_meta(poly: Polygon, anchors: list, buffer_m: float) -> dict:
    return {
        "type": "polygon_convex_hull_buffer",
        "buffer_m": buffer_m,
        "anchors": [{"name": n, "lat": lat, "lon": lon} for lat, lon, n in anchors],
        "bounds_6670": {
            "xmin": poly.bounds[0], "ymin": poly.bounds[1],
            "xmax": poly.bounds[2], "ymax": poly.bounds[3],
        },
        "geojson_6670": mapping(poly),
    }


def area_meta_json() -> dict:
    tr = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    return {
        "omicho_market": _area_meta(
            omicho_market_polygon_6670(tr), OMICHO_MARKET_ANCHORS_LL, OMICHO_MARKET_BUFFER_M,
        ),
        "castle_park": _area_meta(
            castle_park_polygon_6670(tr), CASTLE_PARK_ANCHORS_LL, CASTLE_PARK_BUFFER_M,
        ),
    }
