# -*- coding: utf-8 -*-
"""軽量ナビ日陰に、高精度（樹木あり）だけが持つ日陰を足す。

nav_mask = lightweight ∪ (HP_trees \\ lightweight)

樹冠は航空校正と同じ nendophoto2024 ExG（なければ GSI）、樹高は中央値校正 H≈14.9 m。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from rasterio.transform import from_bounds

from geometry_shadow_engine import _tree_shadow
from plateau_load import kanazawa_study_bbox, load_buildings_projected

ROOT = Path(__file__).resolve().parent
AERIAL_TIF = ROOT / "成果物" / "11_残差樹木日陰" / "13_航空写真" / "航空写真_nendophoto2024_z18_研究範囲.tif"
TREE_HEIGHT_M = 14.9
CANOPY_THR_AERIAL = 0.20
REF_LAT = (36.57125 + 36.56204) / 2.0
REF_LON = (136.65694 + 136.66245) / 2.0
JST = timezone(timedelta(hours=9))

_ENGINE: TreeShadeEngine | None = None
_ENGINE_KEY: tuple | None = None
_MASK_MEM: dict[tuple, np.ndarray] = {}


@dataclass
class TreeShadeEngine:
    canopy: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    cellx: float
    celly: float
    tree_height_m: float
    source: str
    cache_dir: Path

    def mask_at(self, when: date, hour_local: int) -> np.ndarray:
        key = (when.isoformat(), int(hour_local), self.canopy.shape, round(self.tree_height_m, 2))
        if key in _MASK_MEM:
            return _MASK_MEM[key]
        disk = self.cache_dir / f"tree_{when.isoformat()}_{hour_local:02d}_h{self.tree_height_m:.1f}.npy"
        if disk.is_file():
            arr = np.load(disk)
            _MASK_MEM[key] = arr
            return arr
        az, elev = _solar(when, hour_local)
        sh = _tree_shadow(
            self.canopy, self.X, self.Y, az, elev, self.tree_height_m, self.cellx, self.celly
        )
        arr = sh.astype(np.uint8)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(disk, arr)
        _MASK_MEM[key] = arr
        return arr


def _solar(when: date, hour_local: int) -> tuple[float, float]:
    ts = pd.Timestamp(datetime(when.year, when.month, when.day, hour_local, tzinfo=JST))
    sol = pvlib.solarposition.get_solarposition(ts, REF_LAT, REF_LON)
    az = float(sol["azimuth"].iloc[0])
    elev = 90.0 - float(sol["apparent_zenith"].iloc[0])
    return az, elev


def _grid_xy(xmin: float, xmax: float, ymin: float, ymax: float, nx: int, ny: int):
    tf = from_bounds(xmin, xmax, ymin, ymax, nx, ny)
    cellx = (xmax - xmin) / nx
    celly = (ymax - ymin) / ny
    xs = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * cellx
    # 行0が北（ymax）側（rasterio north-up と同じ）
    ys = ymax - (np.arange(ny, dtype=np.float64) + 0.5) * celly
    X, Y = np.meshgrid(xs, ys)
    return tf, X, Y, cellx, celly


def _building_mask(tf, ny: int, nx: int) -> np.ndarray:
    from rasterio import features
    from rasterio.enums import MergeAlg

    bbox_ll, _, _, _, _ = kanazawa_study_bbox()
    transformer = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
    buildings = load_buildings_projected(ROOT / "plateaudata", bbox_ll, transformer)
    shapes = []
    for bf in buildings:
        try:
            shapes.append((bf.poly_xy, 1))
        except Exception:
            continue
    if not shapes:
        return np.zeros((ny, nx), dtype=bool)
    return (
        features.rasterize(
            shapes,
            out_shape=(ny, nx),
            transform=tf,
            fill=0,
            dtype=np.uint8,
            merge_alg=MergeAlg.add,
            all_touched=True,
        )
        > 0
    )


def _canopy_aerial(X: np.ndarray, Y: np.ndarray, bmask: np.ndarray) -> tuple[np.ndarray, str]:
    from export_aerial_calibrated_tree_shade_maps import canopy_from_aerial, load_aerial_arrays

    mosaic, mx0, my0, mpp = load_aerial_arrays()
    canopy, _, _ = canopy_from_aerial(
        X, Y, mosaic, mx0, my0, mpp, bmask, thr=CANOPY_THR_AERIAL
    )
    return canopy, "nendophoto2024_ExG_H14.9m"


def _canopy_gsi(X: np.ndarray, Y: np.ndarray, bmask: np.ndarray) -> tuple[np.ndarray, str]:
    from geometry_shadow_engine import ORTHO_Z, _fetch_mosaic, _tree_fraction_grid

    to_ll = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    los, las = [], []
    for x, y in (
        (float(X.min()), float(Y.min())),
        (float(X.min()), float(Y.max())),
        (float(X.max()), float(Y.min())),
        (float(X.max()), float(Y.max())),
    ):
        lo, la = to_ll.transform(x, y)
        los.append(lo)
        las.append(la)
    mosaic, mx0, my0, mpp = _fetch_mosaic(
        min(las) - 0.001,
        max(las) + 0.001,
        min(los) - 0.001,
        max(los) + 0.001,
        ORTHO_Z,
        ROOT / "cache" / "gsi_ortho",
    )
    to_merc = Transformer.from_crs("EPSG:6670", "EPSG:3857", always_xy=True)
    frac = _tree_fraction_grid(X, Y, mosaic, mx0, my0, mpp, to_merc)
    canopy = (frac >= 0.35) & ~bmask
    return canopy, "gsi_seamlessphoto_H14.9m"


def get_tree_engine(grid, *, root: Path | None = None, tree_height_m: float = TREE_HEIGHT_M) -> TreeShadeEngine | None:
    global _ENGINE, _ENGINE_KEY
    root = root or ROOT
    key = (round(grid.xmin, 3), round(grid.xmax, 3), round(grid.ymin, 3), round(grid.ymax, 3), grid.nx, grid.ny, round(tree_height_m, 2))
    if _ENGINE is not None and _ENGINE_KEY == key:
        return _ENGINE
    tf, X, Y, cellx, celly = _grid_xy(grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.nx, grid.ny)
    bmask = _building_mask(tf, grid.ny, grid.nx)
    source = "none"
    canopy = np.zeros((grid.ny, grid.nx), dtype=bool)
    if AERIAL_TIF.is_file():
        try:
            canopy, source = _canopy_aerial(X, Y, bmask)
        except Exception:
            canopy, source = _canopy_gsi(X, Y, bmask)
    else:
        try:
            canopy, source = _canopy_gsi(X, Y, bmask)
        except Exception:
            return None
    if not np.any(canopy):
        return None
    eng = TreeShadeEngine(
        canopy=canopy,
        X=X,
        Y=Y,
        cellx=cellx,
        celly=celly,
        tree_height_m=float(tree_height_m),
        source=source,
        cache_dir=root / "data" / "cache" / "tree_residual",
    )
    _ENGINE = eng
    _ENGINE_KEY = key
    return eng


def combine_light_and_hp_trees(light: np.ndarray, tree: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """軽量 ∪（高精度樹木のうち軽量に無いセル）。"""
    t = (np.asarray(tree) > 0).astype(np.uint8)
    L = (np.asarray(light) > 0).astype(np.uint8)
    residual = (t == 1) & (L == 0)
    combined = np.maximum(L, t)
    return combined.astype(np.uint8), residual.astype(np.uint8)


def apply_tree_residual(
    light_mask: np.ndarray,
    grid,
    when: date,
    hour_local: int,
    *,
    include_tree_shade: bool = True,
    root: Path | None = None,
    tree_height_m: float = TREE_HEIGHT_M,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns
    -------
    combined, residual, info
    """
    empty = np.zeros_like(light_mask, dtype=np.uint8)
    info = {
        "include_tree_shade": bool(include_tree_shade),
        "tree_shade_applied": False,
        "tree_shade_source": None,
        "tree_height_m": tree_height_m,
        "tree_residual_cell_fraction": 0.0,
        "combined_shadow_cell_fraction": float(np.mean(light_mask > 0)) if light_mask.size else 0.0,
        "lightweight_shadow_cell_fraction": float(np.mean(light_mask > 0)) if light_mask.size else 0.0,
    }
    if not include_tree_shade:
        return light_mask.astype(np.uint8), empty, info
    eng = get_tree_engine(grid, root=root, tree_height_m=tree_height_m)
    if eng is None:
        info["tree_shade_source"] = "unavailable"
        return light_mask.astype(np.uint8), empty, info
    tree = eng.mask_at(when, hour_local)
    if tree.shape != light_mask.shape:
        info["tree_shade_source"] = f"shape_mismatch:{tree.shape}"
        return light_mask.astype(np.uint8), empty, info
    combined, residual = combine_light_and_hp_trees(light_mask, tree)
    info.update(
        {
            "tree_shade_applied": True,
            "tree_shade_source": eng.source,
            "tree_residual_cell_fraction": float(np.mean(residual > 0)),
            "combined_shadow_cell_fraction": float(np.mean(combined > 0)),
        }
    )
    return combined, residual, info
