"""
6/1〜9/30・毎日 6:00〜18:00（計13時間刻み）の日陰をグリッド化し、ナビ用データとして保存する。
PNG は出力しない。

高速化のため、各时刻は「水平地盤上の直方体建物」近似:
  影ポリゴン = 建物 footprint を水平ベクトル (-H/sz * sx, -H/sz * sy) だけ平行移動
（DEM 凹凸はナビ用集計では省略。全時間帯×全日の組み合わせが巨大なため）

出力:
  data/summer_shadow_YYYY.npz  … masks (T,ny,nx) uint8, 座標・グリッド情報
  data/summer_shadow_YYYY_meta.json … 時刻インデックスの対応表
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.transform import from_bounds, xy
from scipy.interpolate import LinearNDInterpolator
from shapely.affinity import translate

from dem_load import load_dem_vertices_ll
from plateau_load import kanazawa_study_bbox, load_buildings_projected


def sun_vector_enu(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    el = np.radians(90.0 - zenith_deg)
    az = np.radians(azimuth_deg)
    sx = np.cos(el) * np.sin(az)
    sy = np.cos(el) * np.cos(az)
    sz = np.sin(el)
    v = np.array([sx, sy, sz], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def one_timestep_mask(
    buildings,
    sun: np.ndarray,
    transform,
    shape: tuple[int, int],
) -> np.ndarray:
    """shape = (行数 ny, 列数 nx)。各セル 0/1 で日陰。"""
    sx, sy, sz = float(sun[0]), float(sun[1]), float(sun[2])
    ny, nx = shape
    if sz <= 1e-9:
        return np.zeros((ny, nx), dtype=np.uint8)
    shapes = []
    for b in buildings:
        h = b.height_m
        shp = translate(b.poly_xy, xoff=-h * sx / sz, yoff=-h * sy / sz)
        if shp.is_empty:
            continue
        shapes.append((shp, 1))
    if not shapes:
        return np.zeros((ny, nx), dtype=np.uint8)
    out = features.rasterize(
        shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        merge_alg=MergeAlg.add,
    )
    return (out > 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--resolution", type=float, default=12.0, help="グリッド1辺の長さ[m]（おおよそ）")
    ap.add_argument("--max-days", type=int, default=None, help="デバッグ用: 先頭N日だけ計算")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = root / "plateaudata"
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)

    bbox_ll, lat_min, lat_max, lon_min, lon_max = kanazawa_study_bbox()
    transformer = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
    print("建物読込中…")
    buildings = load_buildings_projected(data_dir, bbox_ll, transformer)
    if not buildings:
        print("建物がありません。", file=sys.stderr)
        return 1
    print(f"建物数: {len(buildings)}")

    bx, by = transformer.transform(
        np.array([lon_min, lon_max, lon_min, lon_max]),
        np.array([lat_min, lat_min, lat_max, lat_max]),
    )
    xmin, xmax = float(np.min(bx)), float(np.max(bx))
    ymin, ymax = float(np.min(by)), float(np.max(by))

    width_m = xmax - xmin
    height_m = ymax - ymin
    nx = max(8, int(np.ceil(width_m / args.resolution)))
    ny = max(8, int(np.ceil(height_m / args.resolution)))
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)

    # ナビ用: 各セル中心の DEM 標高 [m]（勾配コスト用。影グリッドと同じ形状）
    print("DEM から標高グリッド（セル中心）を作成中…")
    inv_wgs = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    dem_files = sorted(data_dir.glob("*_dem_*_op.gml"))
    dem_lons, dem_lats, dem_zs = load_dem_vertices_ll(dem_files, bbox_ll)
    if dem_lons is None:
        elev_m = np.zeros((ny, nx), dtype=np.float32)
        print("警告: DEM 頂点が取得できませんでした。勾配要素は 0 扱いになります。", file=sys.stderr)
    else:
        interp_e = LinearNDInterpolator(
            np.column_stack([dem_lons, dem_lats]), dem_zs, fill_value=np.nan
        )
        rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        xs_c, ys_c = xy(transform, rows, cols, offset="center")
        lon_c, lat_c = inv_wgs.transform(xs_c.ravel(), ys_c.ravel())
        elev_m = interp_e(np.column_stack([lon_c, lat_c])).reshape(ny, nx).astype(np.float32)
        med = float(np.nanmedian(elev_m))
        if not np.isfinite(med):
            med = 0.0
        elev_m = np.nan_to_num(elev_m, nan=med).astype(np.float32)

    ref_lat = (36.57125 + 36.56204) / 2.0
    ref_lon = (136.65694 + 136.66245) / 2.0

    dates = pd.date_range(f"{args.year}-06-01", f"{args.year}-09-30", freq="D")
    if args.max_days is not None:
        dates = dates[: int(args.max_days)]
    hours = list(range(6, 19))
    n_h = len(hours)
    T = len(dates) * n_h
    print(f"タイムステップ数: {T}（{len(dates)} 日 × {n_h} 時間）グリッド {ny}×{nx}")

    masks = np.zeros((T, ny, nx), dtype=np.uint8)
    iso_times: list[str] = []
    t0 = time.perf_counter()
    idx = 0
    for di, day in enumerate(dates):
        for hour in hours:
            ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, tz="Asia/Tokyo")
            sol = pvlib.solarposition.get_solarposition(ts, ref_lat, ref_lon)
            zenith = float(sol["apparent_zenith"].iloc[0])
            azimuth = float(sol["azimuth"].iloc[0])
            el = 90.0 - zenith
            if el <= 0:
                masks[idx] = 0
            else:
                sun = sun_vector_enu(azimuth, zenith)
                masks[idx] = one_timestep_mask(buildings, sun, transform, (ny, nx))
            iso_times.append(ts.isoformat())
            idx += 1
        if (di + 1) % 10 == 0 or di == 0:
            dt = time.perf_counter() - t0
            print(f"  進捗 {di+1}/{len(dates)} 日 … 経過 {dt:.1f}s")

    npz_path = out_dir / f"summer_shadow_{args.year}.npz"
    np.savez_compressed(
        npz_path,
        masks=masks,
        elev_m=elev_m,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        nx=nx,
        ny=ny,
        crs_epsg=6670,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        hours=np.array(hours, dtype=np.int16),
        dates_ordinal=np.array([d.toordinal() for d in dates], dtype=np.int32),
    )

    meta = {
        "year": args.year,
        "grid_nx": nx,
        "grid_ny": ny,
        "bounds_xy_m": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        "crs": "EPSG:6670",
        "hours_local": hours,
        "timezone": "Asia/Tokyo",
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "n_days": len(dates),
        "n_hours_per_day": n_h,
        "n_timesteps": T,
        "time_index": "t = day_index * n_hours + hour_index  where hour_index = hour - 6",
        "iso_times_sample": [iso_times[0], iso_times[min(100, len(iso_times) - 1)], iso_times[-1]],
        "approximation": "flat_ground_translate_building_footprint",
        "building_count": len(buildings),
        "has_elev_grid": bool(dem_lons is not None),
    }
    meta_path = out_dir / f"summer_shadow_{args.year}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    times_path = out_dir / f"summer_shadow_{args.year}_times.json"
    with open(times_path, "w", encoding="utf-8") as f:
        json.dump(iso_times, f, ensure_ascii=False)

    print(f"保存: {npz_path}")
    print(f"保存: {meta_path}")
    print(f"保存: {times_path}（長さ {len(iso_times)}）")
    print(f"総所要時間 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
