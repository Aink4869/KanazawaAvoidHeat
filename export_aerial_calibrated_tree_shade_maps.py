# -*- coding: utf-8 -*-
"""
航空写真（nendophoto2024）+ PLATEAU 建物3D から:
  1) 樹冠を認識（建物フットプリント除外）
  2) 写っている日陰の向きで撮影月・時刻を確定
  3) 日陰の長さから代表樹高を推定（H = L · tan(α)）
  4) 推定樹高で 10:30 / 12:30 / 14:30 の日陰分布を
     road_shade_sim_calc_*_2m_geometryDEM と同形式で出力

例:
  py -3 export_aerial_calibrated_tree_shade_maps.py --cell-m 2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import rasterio
from matplotlib.lines import Line2D
from pyproj import Transformer
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.transform import from_bounds, xy as rio_xy
from scipy.ndimage import binary_closing, binary_opening, uniform_filter

import best_fine_shade_compare as bfs
import export_road_shade_sim_bundle as sim
import export_road_shade_time_maps as maps
from best_fine_shade_compare import (
    C_SHADE_RGB,
    C_SUN_RGB,
    DATE,
    TRAN_BG,
    _pred,
    hour_slot_label,
    tree_shadow,
)
from compare_field_road_tran import _configure_japanese_font, _metrics_on_mask
from compare_shadow_web_vs_geometry import _geometry_shade_union, _rasterize_union
from plateau_load import kanazawa_study_bbox, load_buildings_projected

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from estimate_nendo2024_datetime import (  # noqa: E402
    estimate_shadow_azimuth,
    fetch_kanazawa_shoot_month,
    match_solar_times,
)
AERIAL_DIR = ROOT / "成果物" / "11_残差樹木日陰" / "13_航空写真"
AERIAL_TIF = AERIAL_DIR / "航空写真_nendophoto2024_z18_研究範囲.tif"
AERIAL_JPG = AERIAL_DIR / "航空写真_nendophoto2024_z18_研究範囲.jpg"
DT_JSON = AERIAL_DIR / "nendophoto2024_撮影月と時刻推定.json"
OUT_ROOT = ROOT / "成果物" / "11_残差樹木日陰" / "14_航空校正樹高_日陰計算マップ"

REF_LAT = (36.57125 + 36.56204) / 2.0
REF_LON = (136.65694 + 136.66245) / 2.0
CANOPY_THR = 0.35
SLOTS = [10.5, 12.5, 14.5]


def _solar(ts: pd.Timestamp) -> tuple[float, float]:
    sol = pvlib.solarposition.get_solarposition(ts, REF_LAT, REF_LON)
    az = float(sol["azimuth"].iloc[0])
    elev = 90.0 - float(sol["apparent_zenith"].iloc[0])
    return az, elev


def resolve_calibration_datetime() -> dict:
    """公式撮影月 + 影方位 → 代表日時。"""
    if DT_JSON.is_file():
        prev = json.loads(DT_JSON.read_text(encoding="utf-8"))
        matches = prev.get("solar_matches_within_10deg") or []
        if matches:
            best = matches[0]
            return {
                "datetime_jst": best["datetime_jst"],
                "sun_az": best["sun_az"],
                "sun_elev": best["sun_elev"],
                "daz": best["daz"],
                "month_text": (prev.get("preferred_interpretation") or {}).get("month_text"),
                "source": str(DT_JSON),
                "shadow_from_image": prev.get("shadow_from_image"),
            }

    url, uniq, _ = fetch_kanazawa_shoot_month()
    shadow = estimate_shadow_azimuth(AERIAL_JPG)
    months: list[int] = []
    for p in uniq:
        months.extend(int(x) for x in re.findall(r"(\d{1,2})月", str(p.get("撮影年月", ""))))
    months = sorted(set(months)) or [4]
    matches = match_solar_times(shadow["implied_sun_azimuth_deg"], months)
    best = matches[0]
    return {
        "datetime_jst": best["datetime_jst"],
        "sun_az": best["sun_az"],
        "sun_elev": best["sun_elev"],
        "daz": best["daz"],
        "month_text": uniq[0].get("撮影年月") if uniq else None,
        "source": url,
        "shadow_from_image": shadow,
        "matches_top5": matches[:5],
    }


def load_aerial_arrays() -> tuple[np.ndarray, float, float, float]:
    """EPSG:3857 RGB mosaic + origin + mpp（ピクセル正方形想定）。"""
    with rasterio.open(AERIAL_TIF) as ds:
        rgb = np.moveaxis(ds.read((1, 2, 3)), 0, -1)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        transform = ds.transform
        mx0 = float(transform.c)
        my0 = float(transform.f)
        mpp = float(abs(transform.a))
    return rgb, mx0, my0, mpp


def canopy_from_aerial(
    X: np.ndarray,
    Y: np.ndarray,
    mosaic: np.ndarray,
    mx0: float,
    my0: float,
    mpp: float,
    building_mask: np.ndarray,
    *,
    thr: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    航空オルソから樹冠率を推定。
    nendophoto2024 は seamlessphoto より ExG が弱いため、固定閾値に加え
    ExG 上位パーセンタイルの適応マスクを併用する。建物フットプリントは除外。
    """
    to_merc = Transformer.from_crs("EPSG:6670", "EPSG:3857", always_xy=True)
    H, W = mosaic.shape[:2]
    Rr = mosaic[:, :, 0].astype(np.float32)
    Gg = mosaic[:, :, 1].astype(np.float32)
    Bb = mosaic[:, :, 2].astype(np.float32)
    exg = 2 * Gg - Rr - Bb
    bright = (Rr + Gg + Bb) / 3.0
    # 固定閾（seamless 向け）
    veg_fixed = (exg > 38) & (Gg > Rr + 20) & (Gg >= Bb + 4) & (bright > 45) & (bright < 165)
    # 適応: 明るい画素のうち ExG 上位（春の薄緑・桜寄りにも対応）
    valid = (bright > 35) & (bright < 200) & (Gg >= Rr - 5)
    if valid.any():
        p80 = float(np.percentile(exg[valid], 80))
        p88 = float(np.percentile(exg[valid], 88))
    else:
        p80, p88 = 10.0, 20.0
    veg_adapt = valid & (exg >= p80) & (Gg > Rr + 5) & (Gg >= Bb - 8)
    veg_strict = valid & (exg >= p88) & (Gg > Rr + 10) & (Gg >= Bb - 2)
    veg = veg_fixed | veg_adapt
    # 適応が広すぎる場合は厳しめ側へ
    if float(veg.mean()) > 0.35:
        veg = veg_fixed | veg_strict
    if float(veg.mean()) < 0.01:
        veg = veg_adapt

    half = 8
    veg_sm = uniform_filter(veg.astype(np.float64), size=2 * half + 1)
    mxx, myy = to_merc.transform(X.ravel(), Y.ravel())
    # GeoTIFF Affine: row = (my0 - y) / mpp （北上が前提）
    px = ((mxx - mx0) / mpp).astype(np.int32)
    py = ((my0 - myy) / mpp).astype(np.int32)
    tree_frac = np.zeros(X.size, dtype=np.float64)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    tree_frac[ok] = veg_sm[py[ok], px[ok]]
    tree_frac = tree_frac.reshape(X.shape)
    canopy = (tree_frac >= thr) & ~building_mask
    meta = {
        "veg_pixel_fraction": float(veg.mean()),
        "veg_fixed_fraction": float(veg_fixed.mean()),
        "exg_p80": p80,
        "exg_p88": p88,
        "canopy_thr": thr,
        "sample_ok_fraction": float(ok.mean()),
        "tree_frac_p90": float(np.percentile(tree_frac, 90)),
        "tree_frac_max": float(tree_frac.max()),
    }
    return canopy, tree_frac, meta


def dark_shadow_from_aerial(
    X: np.ndarray,
    Y: np.ndarray,
    mosaic: np.ndarray,
    mx0: float,
    my0: float,
    mpp: float,
    *,
    bright_pct: float = 28.0,
) -> np.ndarray:
    to_merc = Transformer.from_crs("EPSG:6670", "EPSG:3857", always_xy=True)
    H, W = mosaic.shape[:2]
    bright = mosaic.astype(np.float32).mean(axis=2)
    thr = float(np.percentile(bright, bright_pct))
    dark = (bright <= thr) & (bright > 12)
    mxx, myy = to_merc.transform(X.ravel(), Y.ravel())
    px = ((mxx - mx0) / mpp).astype(np.int32)
    py = ((my0 - myy) / mpp).astype(np.int32)
    out = np.zeros(X.size, dtype=bool)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    out[ok] = dark[py[ok], px[ok]]
    sh = out.reshape(X.shape)
    sh = binary_opening(sh, iterations=1)
    sh = binary_closing(sh, iterations=1)
    return sh


def rasterize_buildings(buildings, transform, ny: int, nx: int) -> np.ndarray:
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
            shapes, out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8, merge_alg=MergeAlg.add
        )
        > 0
    )


def estimate_height_from_shadow_length(
    canopy: np.ndarray,
    cast: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    az: float,
    elev: float,
    cellx: float,
    celly: float,
    *,
    max_L_m: float = 45.0,
    subsample: int = 8,
) -> dict:
    """
    樹冠から太陽の反対方向へ、建物以外の暗領域（cast）の連続長 L を測り
    H = L · tan(elev) の中央値を代表樹高とする。
    """
    if elev <= 1.0 or not canopy.any() or not cast.any():
        return {"tree_height_m": 10.0, "n_samples": 0, "note": "fallback H=10"}

    dx = -math.sin(math.radians(az))
    dy = -math.cos(math.radians(az))
    step = min(cellx, celly) * 0.5
    nstep = int(max_L_m / step) + 1
    ny, nx = canopy.shape
    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())

    rows, cols = np.where(canopy)
    # 影方向に隣接が非樹冠の縁を優先
    edge_idx = []
    for i in range(0, len(rows), subsample):
        r, c = int(rows[i]), int(cols[i])
        x1 = X[r, c] + step * dx
        y1 = Y[r, c] + step * dy
        c1 = int((x1 - xmin) / (xmax - xmin) * nx)
        r1 = int((ymax - y1) / (ymax - ymin) * ny)
        if 0 <= r1 < ny and 0 <= c1 < nx and not canopy[r1, c1]:
            edge_idx.append(i)
    if len(edge_idx) < 30:
        edge_idx = list(range(0, len(rows), subsample))

    lengths: list[float] = []
    for i in edge_idx:
        r0, c0 = int(rows[i]), int(cols[i])
        xs, ys = float(X[r0, c0]), float(Y[r0, c0])
        L = 0.0
        seen_cast = False
        for s in range(1, nstep + 1):
            t = s * step
            if t > max_L_m:
                break
            px = xs + t * dx
            py = ys + t * dy
            c2 = int((px - xmin) / (xmax - xmin) * nx)
            r2 = int((ymax - py) / (ymax - ymin) * ny)
            if not (0 <= r2 < ny and 0 <= c2 < nx):
                break
            if canopy[r2, c2]:
                continue
            if cast[r2, c2]:
                seen_cast = True
                L = t
            elif seen_cast:
                break
            elif t > 3.0 * min(cellx, celly):
                break
        if seen_cast and 2.0 <= L <= max_L_m:
            lengths.append(L)

    if len(lengths) < 20:
        # IoU スイープで補完
        best_h, best_score = 10.0, -1.0
        scores = []
        for h in np.arange(4.0, 22.5, 0.5):
            pred = tree_shadow(canopy, X, Y, az, elev, float(h), cellx, celly) & ~canopy
            inter = int(np.sum(pred & cast))
            union = int(np.sum(pred | cast))
            score = inter / max(1, union)
            scores.append({"h": float(h), "iou": score, "inter": inter})
            if score > best_score:
                best_score, best_h = score, float(h)
        return {
            "tree_height_m": round(best_h, 2),
            "method": "iou_sweep_fallback",
            "n_samples": len(lengths),
            "best_iou": round(best_score, 4),
            "iou_top": sorted(scores, key=lambda d: -d["iou"])[:5],
            "sun_elev_deg": elev,
            "sun_az_deg": az,
        }

    L_med = float(np.median(lengths))
    L_p25 = float(np.percentile(lengths, 25))
    L_p75 = float(np.percentile(lengths, 75))
    H = L_med * math.tan(math.radians(elev))
    H = float(np.clip(H, 4.0, 25.0))
    return {
        "tree_height_m": round(H, 2),
        "method": "median_cast_length",
        "n_samples": len(lengths),
        "L_median_m": round(L_med, 2),
        "L_p25_m": round(L_p25, 2),
        "L_p75_m": round(L_p75, 2),
        "formula": "H = L_median * tan(elev)",
        "sun_elev_deg": elev,
        "sun_az_deg": az,
    }


def _method_note_calibrated(lang: str, cell_m: float, h_m: float) -> str:
    if lang == "en":
        return (
            f"(shade=blue / sunlit=red / cell={cell_m:.0f} m / "
            f"geometry+DEM+trees H≈{h_m:.1f}m from aerial cast length)"
        )
    return (
        f"（日陰=青 / 日向=赤 / 1セル={cell_m:.0f}m / "
        f"幾何+DEM+樹木 航空影長校正 H≈{h_m:.1f}m）"
    )


def render_sim_calc_combined_calibrated(
    ctx: dict,
    *,
    windows_data: list,
    cell_m: float,
    out_path: Path,
    lang: str,
    h_m: float,
    eval_mask: np.ndarray | None = None,
) -> None:
    """export_road_shade_sim_bundle.render_sim_calc_combined_map と同レイアウト。"""
    road_mask = ctx["road_mask"]
    extent, ny, nx = ctx["extent"], ctx["ny"], ctx["nx"]
    if eval_mask is None:
        eval_mask = road_mask
    sim_lbl = "geometry+DEM+trees (aerial H)" if lang == "en" else "幾何+DEM+樹木（航空校正）"
    ncols = len(windows_data) * 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 5.4), squeeze=False)
    axs = axes.ravel()
    for block, (win, field, _) in enumerate(windows_data):
        base, slot = block * 3, win["slot"]
        pred = _pred(ctx, True, slot)
        wlabel = win["window_en"] if lang == "en" else win["window_ja"]
        grid_rgb = np.ones((ny, nx, 3), dtype=np.float32) * TRAN_BG
        grid_rgb[pred] = C_SHADE_RGB
        grid_rgb[~pred] = C_SUN_RGB
        axs[base].imshow(grid_rgb, origin="upper", extent=extent, aspect="equal", interpolation="nearest")
        if lang == "en":
            axs[base].set_title(f"Sim grid ({wlabel})\n{sim_lbl}", fontsize=9)
        else:
            axs[base].set_title(f"日陰計算・全域（{wlabel}）\n{sim_lbl}", fontsize=9)
        road_disp = np.ma.masked_where(~road_mask, np.ones((ny, nx)))
        axs[base + 1].imshow(road_disp, origin="upper", extent=extent, cmap="Greens", vmin=0, vmax=1)
        axs[base + 1].set_title("PLATEAU roads" if lang == "en" else "PLATEAU 道路", fontsize=9)
        road_rgb = np.ones((ny, nx, 3), dtype=np.float32) * TRAN_BG
        road_rgb[road_mask & pred] = C_SHADE_RGB
        road_rgb[road_mask & ~pred] = C_SUN_RGB
        axs[base + 2].imshow(road_rgb, origin="upper", extent=extent, aspect="equal", interpolation="nearest")
        m = _metrics_on_mask(field, pred, eval_mask)
        if lang == "en":
            axs[base + 2].set_title(
                f"{hour_slot_label(slot)} all PLATEAU roads\n"
                f"(metrics vs field: F1={m['f1_shade']:.2f} Rec={m['recall_shade']:.0%} "
                f"Prec={m['precision_shade']:.0%} n={m['n_eval_cells']})",
                fontsize=8.5,
            )
        else:
            axs[base + 2].set_title(
                f"{hour_slot_label(slot)} 道路（全PLATEAU）\n"
                f"（実測比較: F1={m['f1_shade']:.2f} 再現={m['recall_shade']:.0%} "
                f"適合={m['precision_shade']:.0%} n={m['n_eval_cells']}）",
                fontsize=8.5,
            )
        axs[base + 2].set_xlabel("X [m]")
        for j in range(3):
            bfs._set_view(axs[base + j], None)
    title = (
        f"Road shade simulation  {DATE}  JST 10:30 / 12:30 / 14:30"
        if lang == "en"
        else f"道路日陰計算  {DATE}  10:30 / 12:30 / 14:30"
    )
    fig.suptitle(title + "\n" + _method_note_calibrated(lang, cell_m, h_m), fontsize=11)
    leg = (
        [
            Line2D([0], [0], marker="s", ls="", mfc=C_SHADE_RGB, mec="none", ms=9, label="Shade"),
            Line2D([0], [0], marker="s", ls="", mfc=C_SUN_RGB, mec="none", ms=9, label="Sunlit"),
        ]
        if lang == "en"
        else [
            Line2D([0], [0], marker="s", ls="", mfc=C_SHADE_RGB, mec="none", ms=9, label="日陰"),
            Line2D([0], [0], marker="s", ls="", mfc=C_SUN_RGB, mec="none", ms=9, label="日向"),
        ]
    )
    fig.legend(handles=leg, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def render_sim_calc_single_calibrated(
    ctx: dict,
    *,
    win: dict,
    field: np.ndarray,
    cell_m: float,
    out_path: Path,
    lang: str,
    h_m: float,
    eval_mask: np.ndarray | None = None,
) -> None:
    road_mask, extent, ny, nx = ctx["road_mask"], ctx["extent"], ctx["ny"], ctx["nx"]
    if eval_mask is None:
        eval_mask = road_mask
    slot = win["slot"]
    pred = _pred(ctx, True, slot)
    sim_en = "geometry+DEM+trees (aerial H)"
    sim_ja = "幾何+DEM+樹木（航空校正）"
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 5.2), squeeze=False)
    axs = axs.ravel()
    grid_rgb = np.ones((ny, nx, 3), dtype=np.float32) * TRAN_BG
    grid_rgb[pred] = C_SHADE_RGB
    grid_rgb[~pred] = C_SUN_RGB
    axs[0].imshow(grid_rgb, origin="upper", extent=extent, aspect="equal", interpolation="nearest")
    wlabel = win["window_en"] if lang == "en" else win["window_ja"]
    if lang == "en":
        axs[0].set_title(f"Sim grid ({wlabel})\n{sim_en}")
    else:
        axs[0].set_title(f"日陰計算・全域（{wlabel}）\n{sim_ja}")
    axs[1].imshow(
        np.ma.masked_where(~road_mask, np.ones((ny, nx))),
        origin="upper",
        extent=extent,
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    axs[1].set_title("PLATEAU roads" if lang == "en" else "PLATEAU 道路")
    road_rgb = np.ones((ny, nx, 3), dtype=np.float32) * TRAN_BG
    road_rgb[road_mask & pred] = C_SHADE_RGB
    road_rgb[road_mask & ~pred] = C_SUN_RGB
    axs[2].imshow(road_rgb, origin="upper", extent=extent, aspect="equal", interpolation="nearest")
    m = _metrics_on_mask(field, pred, eval_mask)
    if lang == "en":
        axs[2].set_title(
            f"{hour_slot_label(slot)} all PLATEAU roads\n"
            f"(metrics vs field: F1={m['f1_shade']:.2f} Rec={m['recall_shade']:.0%} "
            f"Prec={m['precision_shade']:.0%} n={m['n_eval_cells']})",
            fontsize=9,
        )
    else:
        axs[2].set_title(
            f"{hour_slot_label(slot)} 道路（全PLATEAU）\n"
            f"（実測比較: F1={m['f1_shade']:.2f} 再現={m['recall_shade']:.0%} "
            f"適合={m['precision_shade']:.0%} n={m['n_eval_cells']}）",
            fontsize=9,
        )
    axs[2].set_xlabel("X [m] EPSG:6670")
    axs[2].set_ylabel("Y [m]")
    for ax in axs:
        bfs._set_view(ax, None)
    if lang == "en":
        supt = f"Road shade simulation  {DATE}  JST {win['window_en']} ({hour_slot_label(slot)})\n"
    else:
        supt = f"道路日陰計算  {DATE}  JST {win['window_ja']}（{hour_slot_label(slot)}）\n"
    fig.suptitle(supt + _method_note_calibrated(lang, cell_m, h_m), fontsize=11)
    leg = (
        [
            Line2D([0], [0], marker="s", ls="", mfc=C_SHADE_RGB, mec="none", ms=9, label="Shade"),
            Line2D([0], [0], marker="s", ls="", mfc=C_SUN_RGB, mec="none", ms=9, label="Sunlit"),
        ]
        if lang == "en"
        else [
            Line2D([0], [0], marker="s", ls="", mfc=C_SHADE_RGB, mec="none", ms=9, label="日陰"),
            Line2D([0], [0], marker="s", ls="", mfc=C_SUN_RGB, mec="none", ms=9, label="日向"),
        ]
    )
    fig.legend(handles=leg, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-m", type=float, default=2.0)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    _configure_japanese_font()

    if not AERIAL_TIF.is_file():
        print(f"Missing aerial GeoTIFF: {AERIAL_TIF}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = out_dir / "02_日陰計算マップ"
    sim_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1) 撮影月・時刻（影方位） ===", flush=True)
    calib = resolve_calibration_datetime()
    ts_calib = pd.Timestamp(calib["datetime_jst"])
    if ts_calib.tzinfo is None:
        ts_calib = ts_calib.tz_localize("Asia/Tokyo")
    else:
        ts_calib = ts_calib.tz_convert("Asia/Tokyo")
    az_c, elev_c = _solar(ts_calib)
    print(
        f"  month={calib.get('month_text')}  calib={ts_calib}  "
        f"sun az={az_c:.1f} elev={elev_c:.1f}",
        flush=True,
    )

    print("=== 2) 2m グリッド + 建物フットプリント + 航空樹冠 ===", flush=True)
    meta = json.loads((ROOT / "data" / "summer_shadow_2026_meta.json").read_text(encoding="utf-8"))
    b = meta["bounds_xy_m"]
    xmin, xmax, ymin, ymax = b["xmin"], b["xmax"], b["ymin"], b["ymax"]
    nx = max(8, int(np.ceil((xmax - xmin) / args.cell_m)))
    ny = max(8, int(np.ceil((ymax - ymin) / args.cell_m)))
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    cellx = (xmax - xmin) / nx
    celly = (ymax - ymin) / ny
    rows_g, cols_g = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    X, Y = rio_xy(transform, rows_g + 0.5, cols_g + 0.5, offset="center")
    X = np.asarray(X, dtype=np.float64).reshape(ny, nx)
    Y = np.asarray(Y, dtype=np.float64).reshape(ny, nx)

    bbox_ll, lat_min, lat_max, lon_min, lon_max = kanazawa_study_bbox()
    tr = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
    buildings = load_buildings_projected(ROOT / "plateaudata", bbox_ll, tr)
    building_mask = rasterize_buildings(buildings, transform, ny, nx)
    print(f"  grid {nx}×{ny}  buildings={len(buildings)}  footprint={building_mask.mean()*100:.1f}%", flush=True)

    mosaic, mx0, my0, mpp = load_aerial_arrays()
    canopy, tree_frac, canopy_meta = canopy_from_aerial(X, Y, mosaic, mx0, my0, mpp, building_mask)
    S = dark_shadow_from_aerial(X, Y, mosaic, mx0, my0, mpp)
    print(
        f"  canopy={canopy.mean()*100:.1f}%  dark S={S.mean()*100:.1f}%  "
        f"veg_px={canopy_meta['veg_pixel_fraction']*100:.1f}%  "
        f"frac_p90={canopy_meta['tree_frac_p90']:.2f}  (building excluded)",
        flush=True,
    )

    print("=== 3) 校正時刻の建物影 B → 影長から樹高 ===", flush=True)
    shade_union, _, _ = _geometry_shade_union(
        ROOT, ts_calib, bbox_ll, lat_min, lat_max, lon_min, lon_max
    )
    B = _rasterize_union(shade_union, transform, ny, nx) > 0
    cast = S & ~B & ~canopy & ~building_mask
    print(f"  B={B.mean()*100:.1f}%  cast(O\\canopy)={cast.mean()*100:.1f}%", flush=True)
    h_info = estimate_height_from_shadow_length(
        canopy, cast, X, Y, az_c, elev_c, cellx, celly
    )
    H_est = float(h_info["tree_height_m"])
    print(f"  estimated tree height H={H_est:.2f} m  ({h_info.get('method')})", flush=True)

    print("=== 4) 建物+樹木 日陰 @ 10:30/12:30/14:30（同手法） ===", flush=True)
    bfs.HOURS = list(SLOTS)
    bfs.H_TREE = H_est
    ctx = maps.build_context(ROOT, args.cell_m, mask_source="plateau_road")
    # 航空写真由来の樹冠 + 校正樹高で樹木影を差し替え
    for slot in SLOTS:
        ts = bfs.hour_slot_ts(slot)
        az, elev = _solar(ts)
        mask_tree = (
            tree_shadow(canopy, ctx["X"], ctx["Y"], az, elev, H_est, ctx["cellx"], ctx["celly"])
            if elev > 0
            else np.zeros((ny, nx), dtype=bool)
        )
        ctx["per_hour"][slot]["mask_tree"] = mask_tree
        print(
            f"  {hour_slot_label(slot)}  tree_shade={mask_tree.mean()*100:.1f}%  "
            f"combined={(ctx['per_hour'][slot]['mask_geom'] | mask_tree).mean()*100:.1f}%",
            flush=True,
        )
    ctx["canopy_aerial"] = canopy
    ctx["tree_height_aerial_m"] = H_est

    class _Args:
        field_georef = "auto"
        per_time_eval = False

    windows_data, common_valid = sim._load_windows(ctx, _Args())
    eval_mask = ctx["road_mask"]
    if common_valid is not None:
        eval_mask = eval_mask & common_valid

    metrics = {}
    for win, field, _ in windows_data:
        pred = _pred(ctx, True, win["slot"])
        metrics[win["slug"]] = _metrics_on_mask(field, pred, eval_mask)

    for lang in ("en", "ja"):
        render_sim_calc_combined_calibrated(
            ctx,
            windows_data=windows_data,
            cell_m=args.cell_m,
            out_path=sim_dir / sim._fname_sim_combined(lang, True),
            lang=lang,
            h_m=H_est,
            eval_mask=eval_mask,
        )
    for win, field, _ in windows_data:
        for lang in ("en", "ja"):
            render_sim_calc_single_calibrated(
                ctx,
                win=win,
                field=field,
                cell_m=args.cell_m,
                out_path=sim_dir / sim._fname_sim_whole(win, lang=lang, with_trees=True),
                lang=lang,
                h_m=H_est,
                eval_mask=eval_mask,
            )

    report = {
        "aerial": str(AERIAL_TIF),
        "calibration": {
            **calib,
            "datetime_used": str(ts_calib),
            "sun_az_recomputed": az_c,
            "sun_elev_recomputed": elev_c,
        },
        "canopy": {
            "source": "nendophoto2024 GeoTIFF adaptive ExG",
            "building_footprint_excluded": True,
            "canopy_fraction": float(canopy.mean()),
            "canopy_cells": int(canopy.sum()),
            **canopy_meta,
        },
        "tree_height_estimate": h_info,
        "simulation": {
            "date": DATE,
            "slots_jst": [hour_slot_label(s) for s in SLOTS],
            "method": "2m geometry+DEM buildings ∪ aerial canopy tree shadow",
            "tree_height_m": H_est,
            "same_layout_as": (
                "成果物/05_道路日陰_実測比較/06_日陰計算_2m_幾何DEM_QGIS実測/"
                "02_日陰計算マップ/road_shade_sim_calc_1030_1230_1430_2m_geometryDEM_map_en.png"
            ),
        },
        "metrics_vs_field_road": metrics,
        "outputs": {
            "sim_dir": str(sim_dir),
            "combined_en": str(sim_dir / sim._fname_sim_combined("en", True)),
        },
    }
    report_path = out_dir / "航空校正_樹高と日陰_結果.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# 航空校正樹高による日陰計算マップ\n\n"
        f"- 航空: nendophoto2024（撮影月 **{calib.get('month_text')}**、影方位校正 **{ts_calib}**）\n"
        f"- 樹冠: ExG + PLATEAU建物フットプリント除外\n"
        f"- 樹高: 影長 L から H=L·tan(α) → **{H_est:.2f} m**\n"
        f"- 出力時刻: {DATE} の 10:30 / 12:30 / 14:30（既存 sim_calc と同レイアウト）\n"
        f"- フォルダ: `02_日陰計算マップ/`\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}", flush=True)
    print(f"Maps: {sim_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
