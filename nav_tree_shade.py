# -*- coding: utf-8 -*-
"""軽量ナビ日陰に、高精度（樹木あり）だけが持つ日陰を足す。

nav_mask = lightweight ∪ (HP_trees \\ lightweight)

樹冠は航空校正と同じ nendophoto2024 ExG（なければ GSI）、樹高は中央値校正 H≈14.9 m。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parent
AERIAL_TIF = ROOT / "成果物" / "11_残差樹木日陰" / "13_航空写真" / "航空写真_nendophoto2024_z18_研究範囲.tif"
TREE_HEIGHT_M = 14.9
CANOPY_THR_AERIAL = 0.20
ORTHO_Z = 18
R_E = 20037508.342789244
REF_LAT = (36.57125 + 36.56204) / 2.0
REF_LON = (136.65694 + 136.66245) / 2.0
JST = timezone(timedelta(hours=9))

_ENGINE: TreeShadeEngine | None = None
_ENGINE_KEY: tuple | None = None
_MASK_MEM: dict[tuple, np.ndarray] = {}


def _tree_shadow(canopy, X, Y, az, elev, h, cellx, celly):
    if elev <= 0 or not canopy.any():
        return np.zeros(canopy.shape, bool)
    dx, dy = -math.sin(math.radians(az)), -math.cos(math.radians(az))
    L = h / math.tan(math.radians(elev))
    sh = canopy.copy()
    r_idx, c_idx = np.where(canopy)
    xs, ys = X[r_idx, c_idx], Y[r_idx, c_idx]
    step = min(cellx, celly) * 0.5
    nstep = int(L / step) + 1
    ny, nx = canopy.shape
    gxmin, gxmax = float(X.min()), float(X.max())
    gymin, gymax = float(Y.min()), float(Y.max())
    for s in range(1, nstep + 1):
        t = s * step
        if t > L:
            break
        px = xs + t * dx
        py = ys + t * dy
        col2 = ((px - gxmin) / (gxmax - gxmin) * nx).astype(int)
        row2 = ((gymax - py) / (gymax - gymin) * ny).astype(int)
        ok = (col2 >= 0) & (col2 < nx) & (row2 >= 0) & (row2 < ny)
        sh[row2[ok], col2[ok]] = True
    return sh


def _lonlat_to_tile(lon, lat, z):
    n = 2**z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return xt, yt


def _fetch_mosaic(lat_min, lat_max, lon_min, lon_max, z, cache: Path):
    from PIL import Image
    import time
    import urllib.request

    cache.mkdir(parents=True, exist_ok=True)
    n = 2**z
    x0f, y0f = _lonlat_to_tile(lon_min, lat_max, z)
    x1f, y1f = _lonlat_to_tile(lon_max, lat_min, z)
    xt0, xt1 = int(math.floor(x0f)), int(math.floor(x1f))
    yt0, yt1 = int(math.floor(y0f)), int(math.floor(y1f))
    mosaic = Image.new("RGB", ((xt1 - xt0 + 1) * 256, (yt1 - yt0 + 1) * 256))
    for j, yt in enumerate(range(yt0, yt1 + 1)):
        for i, xt in enumerate(range(xt0, xt1 + 1)):
            fp = cache / f"{z}_{xt}_{yt}.jpg"
            if not fp.is_file():
                url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{xt}/{yt}.jpg"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                try:
                    fp.write_bytes(urllib.request.urlopen(req, timeout=25).read())
                    time.sleep(0.04)
                except Exception:
                    continue
            try:
                mosaic.paste(Image.open(fp).convert("RGB"), (i * 256, j * 256))
            except Exception:
                pass
    mx0, my0 = (-R_E + xt0 / n * 2 * R_E), (R_E - yt0 / n * 2 * R_E)
    mpp = (2 * R_E / n) / 256.0
    return np.asarray(mosaic), mx0, my0, mpp


def _tree_fraction_grid(X, Y, mosaic, mx0, my0, mpp, to_merc, half=10):
    H, W = mosaic.shape[:2]
    Rr = mosaic[:, :, 0].astype(np.float32)
    Gg = mosaic[:, :, 1].astype(np.float32)
    Bb = mosaic[:, :, 2].astype(np.float32)
    exg = 2 * Gg - Rr - Bb
    bright = (Rr + Gg + Bb) / 3
    veg = (exg > 38) & (Gg > Rr + 20) & (Gg >= Bb + 4) & (bright > 45) & (bright < 165)
    k = 2 * half + 1
    veg_sm = uniform_filter(veg.astype(np.float64), size=k)
    mxx, myy = to_merc.transform(X.ravel(), Y.ravel())
    px = ((mxx - mx0) / mpp).astype(np.int32)
    py = ((my0 - myy) / mpp).astype(np.int32)
    tree_frac = np.zeros(X.size, dtype=np.float64)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    tree_frac[ok] = veg_sm[py[ok], px[ok]]
    return tree_frac.reshape(X.shape)


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
    ys = ymax - (np.arange(ny, dtype=np.float64) + 0.5) * celly
    X, Y = np.meshgrid(xs, ys)
    return tf, X, Y, cellx, celly


def _building_mask(tf, ny: int, nx: int) -> np.ndarray:
    try:
        from plateau_load import kanazawa_study_bbox, load_buildings_projected
        from rasterio import features
        from rasterio.enums import MergeAlg

        bbox_ll, _, _, _, _ = kanazawa_study_bbox()
        transformer = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
        data_dir = ROOT / "plateaudata"
        if not data_dir.exists():
            return np.zeros((ny, nx), dtype=bool)
        buildings = load_buildings_projected(data_dir, bbox_ll, transformer)
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
    except Exception:
        return np.zeros((ny, nx), dtype=bool)


def _canopy_aerial(X: np.ndarray, Y: np.ndarray, bmask: np.ndarray) -> tuple[np.ndarray, str]:
    from export_aerial_calibrated_tree_shade_maps import canopy_from_aerial, load_aerial_arrays

    mosaic, mx0, my0, mpp = load_aerial_arrays()
    canopy, _, _ = canopy_from_aerial(
        X, Y, mosaic, mx0, my0, mpp, bmask, thr=CANOPY_THR_AERIAL
    )
    return canopy, "nendophoto2024_ExG_H14.9m"


def _canopy_gsi(X: np.ndarray, Y: np.ndarray, bmask: np.ndarray) -> tuple[np.ndarray, str]:
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
    source = "none"
    canopy = np.zeros((grid.ny, grid.nx), dtype=bool)

    # 1. 最優先: 同梱の事前計算キャノピーマスク (.npy)
    canopy_candidates = [
        root / "data" / "canopy_mask_base.npy",
        Path(__file__).resolve().parent / "data" / "canopy_mask_base.npy",
        Path.cwd() / "data" / "canopy_mask_base.npy",
    ]
    for cp in canopy_candidates:
        if cp.is_file():
            try:
                arr = np.load(cp)
                if arr.shape == (grid.ny, grid.nx):
                    canopy = arr > 0
                    source = "canopy_mask_base_H14.9m"
                    break
            except Exception:
                pass

    # 2. 航空写真 TIF がある場合
    if not np.any(canopy):
        bmask = _building_mask(tf, grid.ny, grid.nx)
        if AERIAL_TIF.is_file():
            try:
                canopy, source = _canopy_aerial(X, Y, bmask)
            except Exception:
                pass

    # 3. キャノピーが見つからない場合、全域ゼロ配列で安全にフォールバック
    if not np.any(canopy):
        canopy = np.zeros((grid.ny, grid.nx), dtype=bool)
        source = "empty_fallback"

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
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    include_tree_shade=False:
      combined=light_mask, residual=0

    include_tree_shade=True:
      combined=light_mask ∪ (tree \\ light_mask), residual=tree \\ light_mask
    """
    ny, nx = light_mask.shape
    empty_res = np.zeros((ny, nx), dtype=np.uint8)
    if not include_tree_shade:
        return (
            (np.asarray(light_mask) > 0).astype(np.uint8),
            empty_res,
            {"tree_shade_applied": False, "source": "building_only"},
        )
    eng = get_tree_engine(grid, root=root)
    if eng is None:
        return (
            (np.asarray(light_mask) > 0).astype(np.uint8),
            empty_res,
            {"tree_shade_applied": False, "source": "no_tree_engine"},
        )
    tree_m = eng.mask_at(when, hour_local)
    combined, residual = combine_light_and_hp_trees(light_mask, tree_m)
    info = {
        "tree_shade_applied": True,
        "source": eng.source,
        "tree_height_m": eng.tree_height_m,
        "tree_cells": int(np.sum(tree_m > 0)),
        "residual_cells": int(np.sum(residual > 0)),
        "combined_cells": int(np.sum(combined > 0)),
    }
    return combined, residual, info


_OBS2_MEM: dict[tuple, tuple] = {}


def _nearest_resize(src: np.ndarray, ny2: int, nx2: int) -> np.ndarray:
    ny, nx = src.shape
    rows = np.minimum((np.arange(ny2, dtype=np.int64) * ny) // ny2, ny - 1)
    cols = np.minimum((np.arange(nx2, dtype=np.int64) * nx) // nx2, nx - 1)
    return np.ascontiguousarray(src[np.ix_(rows, cols)])


def build_observation_2m(
    building_mask: np.ndarray,
    grid,
    when: date,
    hour_local: int,
    *,
    include_tree_shade: bool,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """ルート観測用の 2 m 格子。

    建物影は 12 m の計算結果を、同じ範囲の 2 m 格子へ最近傍で載せる。
    樹木影は樹冠を 2 m 格子に載せてから、その時刻の太陽で影を引き直す。
    """
    cell = 2.0
    nx2 = max(8, int(round((float(grid.xmax) - float(grid.xmin)) / cell)))
    ny2 = max(8, int(round((float(grid.ymax) - float(grid.ymin)) / cell)))
    key = (
        when.isoformat(),
        int(hour_local),
        bool(include_tree_shade),
        round(float(grid.xmin), 3),
        round(float(grid.ymin), 3),
        nx2,
        ny2,
        tuple(building_mask.shape),
    )
    cached = _OBS2_MEM.get(key)
    if cached is not None:
        return cached

    bld = _nearest_resize((np.asarray(building_mask) > 0).astype(np.uint8), ny2, nx2)
    elev = None
    if getattr(grid, "elev_m", None) is not None and np.asarray(grid.elev_m).shape == building_mask.shape:
        elev = _nearest_resize(np.asarray(grid.elev_m, dtype=np.float64), ny2, nx2)

    info = {
        "tree_shade_applied": False,
        "source": "building_12m_on_2m_lattice",
        "observation_cell_m": 2.0,
        "building_shade_native_m": 12.0,
        "tree_shade_native_m": None,
        "grid_ny": ny2,
        "grid_nx": nx2,
        "samples_note": "辺あたりの点数はルート側で指定する",
    }
    out = bld
    if include_tree_shade:
        eng = get_tree_engine(grid, root=root)
        if eng is not None and np.any(eng.canopy):
            canopy2 = _nearest_resize(eng.canopy.astype(bool), ny2, nx2)
            _tf, X, Y, cellx, celly = _grid_xy(
                float(grid.xmin), float(grid.xmax), float(grid.ymin), float(grid.ymax), nx2, ny2
            )
            az, elev_sun = _solar(when, hour_local)
            tree = _tree_shadow(
                canopy2, X, Y, az, elev_sun, eng.tree_height_m, cellx, celly
            )
            tree_u = tree.astype(np.uint8)
            out = np.maximum(bld, tree_u)
            info.update(
                {
                    "tree_shade_applied": True,
                    "source": eng.source + "+recast_2m",
                    "tree_shade_native_m": 2.0,
                    "tree_height_m": eng.tree_height_m,
                    "tree_cells": int(np.sum(tree_u > 0)),
                    "combined_cells": int(np.sum(out > 0)),
                }
            )
    result = (out, elev, info)
    _OBS2_MEM[key] = result
    return result
