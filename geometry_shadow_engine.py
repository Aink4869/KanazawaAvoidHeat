"""
幾何+DEM 日陰計算（建物・地形を1回だけ読み込み、時刻ごとに影ポリゴン→ラスタ）。

ルート用 NPZ 生成 (compute_geometry_shadow_npz.py) と、必要なら実行時サンプリングで共用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import shadow_analysis_kanazawa as sak
import shadow_geometry as sg
from plateau_load import kanazawa_study_bbox, load_buildings_projected

CANOPY_THR = 0.35
H_TREE = 10.0
ORTHO_Z = 18
R_E = 20037508.342789244


def rasterize_shade_union(shade_union, transform, ny: int, nx: int) -> np.ndarray:
    if shade_union is None or shade_union.is_empty:
        return np.zeros((ny, nx), dtype=np.uint8)
    g = shade_union
    if g.geom_type == "GeometryCollection":
        polys = [x for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return np.zeros((ny, nx), dtype=np.uint8)
        g = unary_union(polys)
    return features.rasterize(
        [(g, 1)],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        merge_alg=MergeAlg.replace,
    ).astype(np.uint8)


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


@dataclass
class GeometryShadowContext:
    """建物・DEM・（任意）樹冠を1回読み込み、時刻ごとに影マスクを返す。"""

    root: Path
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    nx: int
    ny: int
    with_trees: bool = False
    ref_lat: float = field(default=(36.57125 + 36.56204) / 2.0)
    ref_lon: float = field(default=(136.65694 + 136.66245) / 2.0)
    _study: Polygon = field(init=False, repr=False)
    _terrain: callable = field(init=False, repr=False)
    _buildings: list = field(init=False, repr=False)
    _building_rings: list = field(init=False, repr=False)
    _X: np.ndarray = field(init=False, repr=False)
    _Y: np.ndarray = field(init=False, repr=False)
    _canopy: np.ndarray | None = field(init=False, repr=False, default=None)
    _cellx: float = field(init=False, repr=False)
    _celly: float = field(init=False, repr=False)
    _transform: object = field(init=False, repr=False)

    @classmethod
    def from_bounds(
        cls,
        root: Path,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        *,
        resolution_m: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        with_trees: bool = False,
    ) -> GeometryShadowContext:
        if resolution_m is not None:
            nx = max(8, int(np.ceil((xmax - xmin) / resolution_m)))
            ny = max(8, int(np.ceil((ymax - ymin) / resolution_m)))
        if nx is None or ny is None:
            raise ValueError("resolution_m または nx/ny を指定してください")
        return cls(
            root=root,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            nx=int(nx),
            ny=int(ny),
            with_trees=with_trees,
        )

    @classmethod
    def from_meta_json(cls, root: Path, meta_path: Path, *, resolution_m: float | None = 12.0, with_trees: bool = False):
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        b = meta["bounds_xy_m"]
        if resolution_m is None:
            return cls.from_bounds(
                root,
                b["xmin"],
                b["xmax"],
                b["ymin"],
                b["ymax"],
                nx=int(meta["grid_nx"]),
                ny=int(meta["grid_ny"]),
                with_trees=with_trees,
            )
        return cls.from_bounds(
            root,
            b["xmin"],
            b["xmax"],
            b["ymin"],
            b["ymax"],
            resolution_m=resolution_m,
            with_trees=with_trees,
        )

    def __post_init__(self) -> None:
        self._transform = from_bounds(self.xmin, self.ymin, self.xmax, self.ymax, self.nx, self.ny)
        self._cellx = (self.xmax - self.xmin) / self.nx
        self._celly = (self.ymax - self.ymin) / self.ny

        bbox_ll, lat_min, lat_max, lon_min, lon_max = kanazawa_study_bbox()
        transformer = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
        inv_tf = Transformer.from_crs("EPSG:6670", "EPSG:6697", always_xy=True)

        data_dir = self.root / "plateaudata"
        dem_files = sorted(data_dir.glob("*_dem_*_op.gml"))
        dem_lons, dem_lats, dem_zs = sak.load_dem_vertices(dem_files, bbox_ll)
        if dem_lons is None:
            raise RuntimeError("DEM の読み込みに失敗しました")
        dem_interp = sak.make_terrain_interpolator(dem_lons, dem_lats, dem_zs)
        self._terrain = sak.terrain_factory(dem_interp, inv_tf)

        bx, by = transformer.transform(
            np.array([lon_min, lon_max, lon_min, lon_max]),
            np.array([lat_min, lat_min, lat_max, lat_max]),
        )
        sx0, sx1 = float(np.min(bx)), float(np.max(bx))
        sy0, sy1 = float(np.min(by)), float(np.max(by))
        self._study = box(sx0, sy0, sx1, sy1)

        self._buildings = load_buildings_projected(data_dir, bbox_ll, transformer)
        self._building_rings: list[tuple[np.ndarray, float]] = []
        terrain = self._terrain
        study = self._study
        for bf in self._buildings:
            fp = bf.poly_xy
            if fp.is_empty or not fp.intersects(study):
                continue
            try:
                xs, ys = fp.exterior.xy
            except Exception:
                continue
            ring = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
            if ring.shape[0] < 4:
                continue
            zf = np.array([terrain(float(x), float(y)) for x, y in ring], dtype=np.float64)
            self._building_rings.append((ring, float(bf.height_m), zf))

        rows_g, cols_g = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        from rasterio.transform import xy as rio_xy

        X, Y = rio_xy(self._transform, rows_g + 0.5, cols_g + 0.5, offset="center")
        self._X = np.asarray(X, dtype=np.float64).reshape(self.ny, self.nx)
        self._Y = np.asarray(Y, dtype=np.float64).reshape(self.ny, self.nx)

        if self.with_trees:
            to_ll = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
            los, las = [], []
            for x, y in [
                (self.xmin, self.ymin),
                (self.xmin, self.ymax),
                (self.xmax, self.ymin),
                (self.xmax, self.ymax),
            ]:
                lo, la = to_ll.transform(x, y)
                los.append(lo)
                las.append(la)
            mosaic, mx0, my0, mpp = _fetch_mosaic(
                min(las) - 0.001,
                max(las) + 0.001,
                min(los) - 0.001,
                max(los) + 0.001,
                ORTHO_Z,
                self.root / "cache" / "gsi_ortho",
            )
            to_merc = Transformer.from_crs("EPSG:6670", "EPSG:3857", always_xy=True)
            tree_frac = _tree_fraction_grid(self._X, self._Y, mosaic, mx0, my0, mpp, to_merc)
            self._canopy = tree_frac >= CANOPY_THR
        else:
            self._canopy = None

    @property
    def transform(self):
        return self._transform

    def _building_shade_union(self, sun: np.ndarray):
        shadow_parts: list = []
        terrain = self._terrain
        study = self._study
        for ring, height_m, zf in self._building_rings:
            sh = sg.footprint_building_shadow(
                ring,
                zf,
                height_m,
                sun,
                terrain,
                subdivisions_per_edge=1,
            )
            if not sh.is_empty:
                sh = sh.intersection(study)
                if not sh.is_empty:
                    if sh.geom_type == "MultiPolygon":
                        shadow_parts.extend(list(sh.geoms))
                    else:
                        shadow_parts.append(sh)
        return unary_union(shadow_parts) if shadow_parts else Polygon()

    def mask_at(self, ts: pd.Timestamp) -> np.ndarray:
        ts_jp = ts.tz_convert("Asia/Tokyo") if ts.tzinfo else ts.tz_localize("Asia/Tokyo")
        sol = pvlib.solarposition.get_solarposition(ts_jp, self.ref_lat, self.ref_lon)
        zenith = float(sol["apparent_zenith"].iloc[0])
        azimuth = float(sol["azimuth"].iloc[0])
        elevation = 90.0 - zenith
        if elevation <= 0:
            return np.zeros((self.ny, self.nx), dtype=np.uint8)

        sun = sak.sun_vector_enu(azimuth, zenith)
        shade_union = self._building_shade_union(sun)
        mask = rasterize_shade_union(shade_union, self._transform, self.ny, self.nx)

        if self.with_trees and self._canopy is not None:
            tree_m = _tree_shadow(
                self._canopy,
                self._X,
                self._Y,
                azimuth,
                elevation,
                H_TREE,
                self._cellx,
                self._celly,
            )
            mask = np.maximum(mask, tree_m.astype(np.uint8))

        return mask.astype(np.uint8)
