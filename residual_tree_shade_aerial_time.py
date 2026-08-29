# -*- coding: utf-8 -*-
"""
航空写真の撮影時刻に合わせて残差分解を再実行する。

  S = 航空/衛星画像の暗画素（日陰候補）
  B = 同時刻の PLATEAU 2 m 幾何+DEM 建物影
  O = S ∧ ¬B
  T_under = O ∧ 樹冠
  T_cast  = O ∧ 幾何樹木影 ∧ ¬樹冠
  R = O ∧ ¬(T_under ∨ T_cast)

評価の主指標は「画像 S」に対する一致（時刻整合）。
実測 GeoTIFF（2023-08-05 14時台）は参考のみ（時刻不一致）。

使い方:
  py -3 residual_tree_shade_aerial_time.py --cell-m 2
"""
from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds, xy as rio_xy
from scipy.ndimage import binary_closing, binary_opening, uniform_filter

from best_fine_shade_compare import (
    CANOPY_THR,
    H_TREE,
    ORTHO_Z,
    _load_field_grid,
    area_masks,
    fetch_mosaic,
    tree_fraction_grid,
    tree_shadow,
)
from compare_field_road_tran import _metrics_on_mask
from compare_shadow_web_vs_geometry import _geometry_shade_union, _rasterize_union
from eval_surface_mask import build_eval_surface_mask, surface_label
from kanazawa_areas import castle_mask_raster, omicho_mask_raster
from residual_tree_shade_decompose import (
    _count,
    decompose,
    ortho_shadow_grid,
    render_map,
    summarize_layers,
)
from plateau_load import kanazawa_study_bbox

R_E = 20037508.342789244
REF_LAT = (36.57125 + 36.56204) / 2.0
REF_LON = (136.65694 + 136.66245) / 2.0


# シナリオ: 航空写真時刻に合わせた B + 同時期画像の S
SCENARIOS = [
    {
        "id": "ccb2007_nendo_1045",
        "label": "CCB2007想定 2007-05-18 10:45 + nendophoto2007",
        "datetime_jst": "2007-05-18 10:45:00",
        "s_source": "nendophoto2007",
        "note": "CCB20076枠の年=2007・時計≈10時台・緑葉から5月仮説。月日は未公式確定。",
    },
    {
        "id": "ccb2007_nendo_1030",
        "label": "CCB2007想定 2007-05-18 10:30 + nendophoto2007",
        "datetime_jst": "2007-05-18 10:30:00",
        "s_source": "nendophoto2007",
        "note": "時計読み感度（10:30）。",
    },
    {
        "id": "ccb2007_nendo_1100",
        "label": "CCB2007想定 2007-05-18 11:00 + nendophoto2007",
        "datetime_jst": "2007-05-18 11:00:00",
        "s_source": "nendophoto2007",
        "note": "時計読み感度（11:00）。",
    },
    {
        "id": "s2_20230724_1047",
        "label": "Sentinel-2 2023-07-24 10:47JST",
        "datetime_jst": "2023-07-24 10:47:00",
        "s_source": "sentinel2_20230724",
        "note": "ESA S2B 正確な撮影時刻（UTC 01:47）。解像度10m。",
    },
]


def lonlat_to_tile(lon, lat, z):
    n = 2**z
    xt = (lon + 180.0) / 360.0 * n
    yt = (
        1
        - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi
    ) / 2 * n
    return xt, yt


def fetch_gsi_layer_mosaic(lat_min, lat_max, lon_min, lon_max, z, layer: str, ext: str, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    n = 2**z
    x0f, y0f = lonlat_to_tile(lon_min, lat_max, z)
    x1f, y1f = lonlat_to_tile(lon_max, lat_min, z)
    xt0, xt1 = int(math.floor(x0f)), int(math.floor(x1f))
    yt0, yt1 = int(math.floor(y0f)), int(math.floor(y1f))
    mosaic = Image.new("RGB", ((xt1 - xt0 + 1) * 256, (yt1 - yt0 + 1) * 256), (20, 20, 20))
    for j, yt in enumerate(range(yt0, yt1 + 1)):
        for i, xt in enumerate(range(xt0, xt1 + 1)):
            fp = cache / f"{layer}_{z}_{xt}_{yt}.{ext}"
            if not fp.is_file():
                url = f"https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{xt}/{yt}.{ext}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    fp.write_bytes(urllib.request.urlopen(req, timeout=25).read())
                except Exception:
                    continue
            try:
                mosaic.paste(Image.open(fp).convert("RGB"), (i * 256, j * 256))
            except Exception:
                pass
    mx0 = -R_E + xt0 / n * 2 * R_E
    my0 = R_E - yt0 / n * 2 * R_E
    mpp = (2 * R_E / n) / 256.0
    return np.asarray(mosaic), mx0, my0, mpp


def sentinel_shadow_grid(X, Y, href: str, bright_pct: float = 30.0) -> np.ndarray:
    import rasterio
    from rasterio.warp import transform as rio_transform

    with rasterio.open(href) as ds:
        # X,Y は EPSG:6670 → 画像 CRS へ
        xs, ys = rio_transform("EPSG:6670", ds.crs, X.ravel(), Y.ravel())
        samples = list(ds.sample(zip(xs, ys)))
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] >= 3:
            bright = arr[:, :3].mean(axis=1)
        else:
            bright = arr[:, 0]
        bright = bright.reshape(X.shape)
        # nodata / 異常
        valid = np.isfinite(bright) & (bright > 0)
        vals = bright[valid]
        if vals.size < 100:
            return np.zeros(X.shape, bool)
        thr = float(np.percentile(vals, bright_pct))
        dark = valid & (bright <= thr) & (bright > 5)
        dark = binary_opening(dark, iterations=1)
        dark = binary_closing(dark, iterations=1)
        return dark


def pseudo_field_from_S(S: np.ndarray, road: np.ndarray) -> np.ndarray:
    """S を『正解日陰』とみなす疑似フィールド（道路外は無効）。"""
    field = np.full(S.shape, -1, dtype=np.int16)
    field[road & S] = 1
    field[road & ~S] = 0
    return field


def tag_from_dt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d_%H%M")


def run_one(
    root: Path,
    scenario: dict,
    *,
    cell_m: float,
    bright_pct: float,
    mask_source: str,
    shared: dict,
) -> dict:
    ts = pd.Timestamp(scenario["datetime_jst"], tz="Asia/Tokyo")
    out_dir = root / "成果物" / "11_残差樹木日陰" / f"航空時刻整合_{scenario['id']}_{cell_m:.0f}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = shared["transform"]
    nx, ny = shared["nx"], shared["ny"]
    X, Y = shared["X"], shared["Y"]
    cellx, celly = shared["cellx"], shared["celly"]
    road_mask = shared["road_mask"]
    canopy = shared["canopy"]
    sel_map = shared["sel_map"]
    field_real = shared["field_real"]
    bbox_ll = shared["bbox_ll"]
    lat_min, lat_max, lon_min, lon_max = shared["ll"]

    print(f"\n======== {scenario['id']}  {ts} ========")
    print(f"  S={scenario['s_source']}  note={scenario['note']}")

    # S
    if scenario["s_source"] == "nendophoto2007":
        mosaics = shared["nendo2007"]
        S = ortho_shadow_grid(X, Y, *mosaics, bright_pct=bright_pct)
    elif scenario["s_source"] == "sentinel2_20230724":
        href = shared["s2_href"]
        print("  sampling Sentinel-2 TCI ...")
        S = sentinel_shadow_grid(X, Y, href, bright_pct=bright_pct)
    else:
        raise ValueError(scenario["s_source"])

    # B at matching time
    print(f"  building geometry {ts} ...")
    shade_union, _, info = _geometry_shade_union(root, ts, bbox_ll, lat_min, lat_max, lon_min, lon_max)
    if shade_union is None:
        raise RuntimeError(f"geometry shade failed: {info}")
    B = _rasterize_union(shade_union, transform, ny, nx) > 0

    sol = pvlib.solarposition.get_solarposition(ts, REF_LAT, REF_LON)
    az = float(sol["azimuth"].iloc[0])
    elev = 90.0 - float(sol["apparent_zenith"].iloc[0])
    geom_tree = tree_shadow(canopy, X, Y, az, elev, H_TREE, cellx, celly) if elev > 0 else np.zeros_like(B)
    print(f"  sun az={az:.1f} elev={elev:.1f}  S={int(S.sum())} B={int(B.sum())} canopy={int(canopy.sum())}")

    layers = decompose(S, B, canopy)
    layers["T_cast_candidate"] = layers["O_other"] & geom_tree & ~canopy
    layers["T_tree_all"] = layers["T_tree_under"] | layers["T_cast_candidate"]
    layers["R_after_tree_all"] = layers["O_other"] & ~layers["T_tree_all"]

    # 主評価: 画像 S を正解とみなす
    field_S = pseudo_field_from_S(S, road_mask)
    summary_vs_S = summarize_layers(field_S, layers, sel_map, geom_tree)
    for area, sel in sel_map.items():
        summary_vs_S[area]["vs_field"]["recon_B_plus_T_all"] = _metrics_on_mask(
            field_S, layers["B_building"] | layers["T_tree_all"], sel
        )
        for key in ("T_cast_candidate", "T_tree_all", "R_after_tree_all"):
            summary_vs_S[area]["layers"][key] = {
                "cells": _count(sel, layers[key]),
                "share_of_S": _count(sel, layers[key]) / max(1, _count(sel, layers["S_image"])),
            }

    # 参考: 実測（時刻不一致）
    summary_vs_field = summarize_layers(field_real, layers, sel_map, geom_tree)
    for area, sel in sel_map.items():
        summary_vs_field[area]["vs_field"]["recon_B_plus_T_all"] = _metrics_on_mask(
            field_real, layers["B_building"] | layers["T_tree_all"], sel
        )

    render_map(
        out_dir / "分解図_全体.png",
        X,
        Y,
        layers,
        sel_map["全体"],
        f"残差分解 航空時刻整合 [{scenario['id']}] {ts.strftime('%Y-%m-%d %H:%M')} 2m",
    )
    render_map(
        out_dir / "分解図_金沢城.png",
        X,
        Y,
        layers,
        sel_map["金沢城公園全体"],
        f"残差分解 金沢城 [{scenario['id']}] {ts.strftime('%Y-%m-%d %H:%M')}",
    )

    print("  --- vs 画像S（主評価・時刻整合）---")
    for area in ("全体", "近江町市場周辺", "金沢城公園全体"):
        a = summary_vs_S[area]
        vf = a["vs_field"]
        print(
            f"  {area}: S={a['layers']['S_image']['cells']} O={a['layers']['O_other']['cells']} "
            f"T_under={a['layers']['T_tree_under']['cells']} "
            f"T_cast={a['layers']['T_cast_candidate']['cells']} "
            f"R={a['layers']['R_after_tree_all']['cells']}"
        )
        print(
            f"    F1(vs S) B={vf['B_building']['f1_shade']:.3f}  "
            f"B+T_under={vf['recon_B_plus_T']['f1_shade']:.3f}  "
            f"B+T_all={vf['recon_B_plus_T_all']['f1_shade']:.3f}"
        )

    result = {
        "scenario": scenario,
        "datetime_jst": str(ts),
        "solar": {"azimuth_deg": az, "elevation_deg": elev},
        "grid": {"nx": nx, "ny": ny, "cell_m": cell_m},
        "bounds_xy_m": {
            "xmin": float(shared["xmin"]),
            "xmax": float(shared["xmax"]),
            "ymin": float(shared["ymin"]),
            "ymax": float(shared["ymax"]),
        },
        "counts_road_all": {
            "S": _count(road_mask, S),
            "B": _count(road_mask, B),
            "O": _count(road_mask, layers["O_other"]),
            "T_under": _count(road_mask, layers["T_tree_under"]),
            "T_cast": _count(road_mask, layers["T_cast_candidate"]),
            "T_all": _count(road_mask, layers["T_tree_all"]),
            "R": _count(road_mask, layers["R_after_tree_all"]),
        },
        "vs_image_S": summary_vs_S,
        "vs_field_20230805_reference_only": summary_vs_field,
        "out_dir": str(out_dir),
    }
    (out_dir / "残差分解_航空時刻整合_結果.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 地図ビューア用キャッシュ（レイヤ重ね合わせ）
    cache_dir = root / "data" / "residual_viewer"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{scenario['id']}.npz"
    np.savez_compressed(
        npz_path,
        S=layers["S_image"].astype(np.uint8),
        B=layers["B_building"].astype(np.uint8),
        O=layers["O_other"].astype(np.uint8),
        T_under=layers["T_tree_under"].astype(np.uint8),
        T_cast=layers["T_cast_candidate"].astype(np.uint8),
        T_all=layers["T_tree_all"].astype(np.uint8),
        R=layers["R_after_tree_all"].astype(np.uint8),
        canopy=canopy.astype(np.uint8),
        road=road_mask.astype(np.uint8),
        building_agree=layers["building_agree"].astype(np.uint8),
        building_only_model=layers["building_only_model"].astype(np.uint8),
    )
    meta_path = cache_dir / f"{scenario['id']}.json"
    meta_path.write_text(
        json.dumps(
            {
                "id": scenario["id"],
                "label": scenario["label"],
                "note": scenario["note"],
                "datetime_jst": str(ts),
                "solar": result["solar"],
                "bounds_xy_m": result["bounds_xy_m"],
                "grid": result["grid"],
                "counts_road_all": result["counts_road_all"],
                "f1_vs_S_all": {
                    "B": summary_vs_S["全体"]["vs_field"]["B_building"]["f1_shade"],
                    "B_plus_T_under": summary_vs_S["全体"]["vs_field"]["recon_B_plus_T"]["f1_shade"],
                    "B_plus_T_all": summary_vs_S["全体"]["vs_field"]["recon_B_plus_T_all"]["f1_shade"],
                },
                "npz": npz_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  viewer cache -> {npz_path.name}")
    print(f"  wrote {out_dir}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-m", type=float, default=2.0)
    ap.add_argument("--bright-pct", type=float, default=28.0)
    ap.add_argument("--mask", type=str, default="plateau_road")
    ap.add_argument("--only", type=str, default="", help="scenario id をカンマ区切りで限定")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cell_m = args.cell_m
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    meta = json.loads((root / "data" / "summer_shadow_2026_meta.json").read_text(encoding="utf-8"))
    b = meta["bounds_xy_m"]
    xmin, xmax, ymin, ymax = b["xmin"], b["xmax"], b["ymin"], b["ymax"]
    nx = max(8, int(np.ceil((xmax - xmin) / cell_m)))
    ny = max(8, int(np.ceil((ymax - ymin) / cell_m)))
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    cellx = (xmax - xmin) / nx
    celly = (ymax - ymin) / ny

    print(f"grid {nx}x{ny} cell~{cell_m}m")
    field_real, _, _ = _load_field_grid(root, xmin, ymin, xmax, ymax, nx, ny)

    bbox_ll, lat_min, lat_max, lon_min, lon_max = kanazawa_study_bbox()
    tr = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
    road_mask, surface_meta = build_eval_surface_mask(
        args.mask,
        transform=transform,
        ny=ny,
        nx=nx,
        data_dir=root / "plateaudata",
        bbox_ll=bbox_ll,
        transformer=tr,
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
    )
    print(f"eval mask: {surface_label(surface_meta)} cells={int(road_mask.sum())}")

    rows_g, cols_g = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    X, Y = rio_xy(transform, rows_g + 0.5, cols_g + 0.5, offset="center")
    X = np.asarray(X, dtype=np.float64).reshape(ny, nx)
    Y = np.asarray(Y, dtype=np.float64).reshape(ny, nx)

    to_ll = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    los, las = [], []
    for x, y in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
        lo, la = to_ll.transform(x, y)
        los.append(lo)
        las.append(la)
    lat0, lat1 = min(las) - 0.001, max(las) + 0.001
    lon0, lon1 = min(los) - 0.001, max(los) + 0.001

    # 樹冠は seamlessphoto（既存と同じ）
    mosaic_sp, mx0, my0, mpp = fetch_mosaic(lat0, lat1, lon0, lon1, ORTHO_Z, root / "cache" / "gsi_ortho")
    to_merc = Transformer.from_crs("EPSG:6670", "EPSG:3857", always_xy=True)
    tree_frac = tree_fraction_grid(X, Y, mosaic_sp, mx0, my0, mpp, to_merc)
    canopy = tree_frac >= CANOPY_THR
    print(f"canopy cells={int(canopy.sum())} ({100 * canopy.mean():.1f}%)")

    print("fetch nendophoto2007 ...")
    nendo = fetch_gsi_layer_mosaic(lat0, lat1, lon0, lon1, ORTHO_Z, "nendophoto2007", "png", root / "cache" / "gsi_nendo2007")
    nendo_pack = (nendo[0], nendo[1], nendo[2], nendo[3], to_merc)

    s2_href = (
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
        "sentinel-s2-l2a-cogs/53/S/PA/2023/7/S2B_53SPA_20230724_0_L2A/TCI.tif"
    )

    ctx_masks = {
        "surface_mask": road_mask,
        "road_mask": road_mask,
        "omicho_market_mask": omicho_mask_raster(transform, ny, nx),
        "castle_park_mask": castle_mask_raster(transform, ny, nx),
    }
    sel_map = area_masks(ctx_masks)

    shared = {
        "transform": transform,
        "nx": nx,
        "ny": ny,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "X": X,
        "Y": Y,
        "cellx": cellx,
        "celly": celly,
        "road_mask": road_mask,
        "canopy": canopy,
        "sel_map": sel_map,
        "field_real": field_real,
        "bbox_ll": bbox_ll,
        "ll": (lat_min, lat_max, lon_min, lon_max),
        "nendo2007": nendo_pack,
        "s2_href": s2_href,
    }

    all_results = []
    for sc in SCENARIOS:
        if only and sc["id"] not in only:
            continue
        all_results.append(
            run_one(
                root,
                sc,
                cell_m=cell_m,
                bright_pct=args.bright_pct,
                mask_source=args.mask,
                shared=shared,
            )
        )

    # サマリー表
    summary_rows = []
    for r in all_results:
        vs = r["vs_image_S"]["全体"]["vs_field"]
        c = r["counts_road_all"]
        summary_rows.append(
            {
                "id": r["scenario"]["id"],
                "datetime_jst": r["datetime_jst"],
                "sun_elev_deg": round(r["solar"]["elevation_deg"], 2),
                "S": c["S"],
                "O": c["O"],
                "T_under": c["T_under"],
                "T_cast": c["T_cast"],
                "R": c["R"],
                "F1_B_vs_S": round(vs["B_building"]["f1_shade"], 3),
                "F1_BplusTunder_vs_S": round(vs["recon_B_plus_T"]["f1_shade"], 3),
                "F1_BplusTall_vs_S": round(vs["recon_B_plus_T_all"]["f1_shade"], 3),
                "recall_B_vs_S": round(vs["B_building"]["recall_shade"], 3),
                "prec_B_vs_S": round(vs["B_building"]["precision_shade"], 3),
            }
        )

    out_root = root / "成果物" / "11_残差樹木日陰"
    summary_path = out_root / f"航空時刻整合_サマリー_{cell_m:.0f}m.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": "residual_decompose_aerial_time_matched",
                "cell_m": cell_m,
                "bright_pct": args.bright_pct,
                "primary_metric": "F1 against image-derived S (time-matched), not 2023-08-05 field",
                "scenarios": summary_rows,
                "details": [
                    {"id": r["scenario"]["id"], "out_dir": r["out_dir"], "note": r["scenario"]["note"]}
                    for r in all_results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n==== SUMMARY (vs image S) ====")
    for row in summary_rows:
        print(
            f"{row['id']}: F1 B={row['F1_B_vs_S']}  B+Tu={row['F1_BplusTunder_vs_S']}  "
            f"B+Tall={row['F1_BplusTall_vs_S']}  O/S={row['O']}/{row['S']}"
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
