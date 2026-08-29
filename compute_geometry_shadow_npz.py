"""
幾何+DEM（+任意で樹木）日陰を summer_shadow と同形式の NPZ に書き出す。

ルート検索 (nav_app / integrated) は既定で本 NPZ を優先読み込みする。

例:
  py -3 compute_geometry_shadow_npz.py --year 2026
  py -3 compute_geometry_shadow_npz.py --year 2026 --with-trees
  py -3 compute_geometry_shadow_npz.py --year 2026 --max-days 7   # 試験用
  py -3 compute_geometry_shadow_npz.py --year 2026 --resolution 2  # 2m（ファイル大・計算重）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import xy as rio_xy
from scipy.interpolate import LinearNDInterpolator

from dem_load import load_dem_vertices_ll
from geometry_shadow_engine import GeometryShadowContext
from plateau_load import kanazawa_study_bbox


def _npz_stem(with_trees: bool) -> str:
    return "summer_shadow_geometry_trees" if with_trees else "summer_shadow_geometry"


def _elev_grid(root: Path, xmin, xmax, ymin, ymax, nx, ny, transform) -> np.ndarray:
    bbox_ll, _, _, _, _ = kanazawa_study_bbox()
    dem_files = sorted((root / "plateaudata").glob("*_dem_*_op.gml"))
    dem_lons, dem_lats, dem_zs = load_dem_vertices_ll(dem_files, bbox_ll)
    if dem_lons is None:
        return np.zeros((ny, nx), dtype=np.float32)
    interp_e = LinearNDInterpolator(
        np.column_stack([dem_lons, dem_lats]), dem_zs, fill_value=np.nan
    )
    inv_wgs = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    xs_c, ys_c = rio_xy(transform, rows, cols, offset="center")
    lon_c, lat_c = inv_wgs.transform(xs_c.ravel(), ys_c.ravel())
    elev_m = interp_e(np.column_stack([lon_c, lat_c])).reshape(ny, nx).astype(np.float32)
    med = float(np.nanmedian(elev_m))
    if not np.isfinite(med):
        med = 0.0
    return np.nan_to_num(elev_m, nan=med).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="幾何+DEM 日陰グリッドを NPZ 出力（ルート検索用）")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--resolution", type=float, default=12.0, help="セルサイズ [m]（既定 12 = 現行 NPZ 互換）")
    ap.add_argument("--with-trees", action="store_true", help="GSI 航空写真から樹冠影を加算")
    ap.add_argument("--max-days", type=int, default=None, help="先頭 N 日だけ（試験用）")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = args.data_dir if args.data_dir is not None else root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    flat_meta_path = data_dir / f"summer_shadow_{args.year}_meta.json"
    if not flat_meta_path.is_file():
        print(f"参照 bounds 用 meta がありません: {flat_meta_path}", file=sys.stderr)
        print("先に compute_summer_shadows.py --year ... を1回実行してください。", file=sys.stderr)
        return 1

    flat_meta = json.loads(flat_meta_path.read_text(encoding="utf-8"))
    b = flat_meta["bounds_xy_m"]
    xmin, xmax, ymin, ymax = b["xmin"], b["xmax"], b["ymin"], b["ymax"]

    print("幾何+DEM コンテキストを初期化 …")
    t_init = time.perf_counter()
    ctx = GeometryShadowContext.from_bounds(
        root,
        xmin,
        xmax,
        ymin,
        ymax,
        resolution_m=float(args.resolution),
        with_trees=bool(args.with_trees),
    )
    print(f"  初期化 {time.perf_counter() - t_init:.1f}s  grid {ctx.ny}×{ctx.nx}")

    elev_m = _elev_grid(root, xmin, xmax, ymin, ymax, ctx.nx, ctx.ny, ctx.transform)

    ref_lat = ctx.ref_lat
    ref_lon = ctx.ref_lon
    dates = pd.date_range(f"{args.year}-06-01", f"{args.year}-09-30", freq="D")
    if args.max_days is not None:
        dates = dates[: int(args.max_days)]
    hours = list(range(6, 19))
    n_h = len(hours)
    T = len(dates) * n_h
    print(f"タイムステップ数: {T}（{len(dates)} 日 × {n_h} 時間）")

    masks = np.zeros((T, ctx.ny, ctx.nx), dtype=np.uint8)
    iso_times: list[str] = []
    t0 = time.perf_counter()
    idx = 0
    for di, day in enumerate(dates):
        for hour in hours:
            ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, tz="Asia/Tokyo")
            masks[idx] = ctx.mask_at(ts)
            iso_times.append(ts.isoformat())
            idx += 1
        if (di + 1) % 5 == 0 or di == 0:
            dt = time.perf_counter() - t0
            rate = (di + 1) / max(dt, 1e-6)
            eta = (len(dates) - di - 1) / max(rate, 1e-6)
            print(f"  進捗 {di+1}/{len(dates)} 日 … 経過 {dt:.1f}s  残り約 {eta/60:.1f} 分")

    stem = _npz_stem(args.with_trees)
    npz_path = data_dir / f"{stem}_{args.year}.npz"
    np.savez_compressed(
        npz_path,
        masks=masks,
        elev_m=elev_m,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        nx=ctx.nx,
        ny=ctx.ny,
        crs_epsg=6670,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        hours=np.array(hours, dtype=np.int16),
        dates_ordinal=np.array([d.toordinal() for d in dates], dtype=np.int32),
    )

    approx = "geometry_dem_trees" if args.with_trees else "geometry_dem_walls"
    meta = {
        "year": args.year,
        "model": "geometry_trees" if args.with_trees else "geometry",
        "grid_nx": ctx.nx,
        "grid_ny": ctx.ny,
        "cell_m": float(args.resolution),
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
        "approximation": approx,
        "with_trees": bool(args.with_trees),
        "has_elev_grid": True,
    }
    meta_path = data_dir / f"{stem}_{args.year}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    times_path = data_dir / f"{stem}_{args.year}_times.json"
    with open(times_path, "w", encoding="utf-8") as f:
        json.dump(iso_times, f, ensure_ascii=False)

    print(f"保存: {npz_path}")
    print(f"保存: {meta_path}")
    print(f"保存: {times_path}（長さ {len(iso_times)}）")
    print(f"総所要時間 {time.perf_counter() - t0:.1f}s")
    print("ルート検索: SHADOW_MODEL=geometry で nav_app / integrated を起動")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
