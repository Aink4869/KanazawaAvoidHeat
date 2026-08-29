"""
金沢エリア PLATEAU 建物＋DEM を用いた日陰マップ生成。

専門外の方向け（概要）:
- 国の「PLATEAU」公開データの建物の形・高さと、同じく PLATEAU の地形（DEM）を使います。
- 太陽の位置（日時・場所）から「光がどちらから当たるか」を決め、建物の影が地面のどこに落ちるかを
  数学的にたどります（shadow_geometry モジュール）。
- 図の濃い色は「その瞬間、日陰になりやすい地面の範囲」のイメージです。実際の日照は天気・周辺建物などでも変わります。
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import shadow_geometry as sg


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


@dataclass
class Building:
    footprint_ll: np.ndarray
    height_m: float


def _parse_poslist(text: str) -> np.ndarray:
    vals = np.array([float(x) for x in text.split()], dtype=np.float64)
    return vals.reshape(-1, 3)


def iter_buildings_gml(path: Path):
    for event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "Building":
            continue
        height = None
        for child in elem:
            if _local(child.tag) == "measuredHeight" and child.text:
                height = float(child.text.strip())
                break
        if height is None or height <= 0:
            elem.clear()
            continue
        lod0 = None
        for child in elem:
            if _local(child.tag) == "lod0FootPrint":
                lod0 = child
                break
        if lod0 is None:
            elem.clear()
            continue
        pos_text = None
        for node in lod0.iter():
            if _local(node.tag) == "posList" and node.text:
                pos_text = node.text
                break
        elem.clear()
        if not pos_text:
            continue
        try:
            pts = _parse_poslist(pos_text)
        except ValueError:
            continue
        if len(pts) < 3:
            continue
        yield Building(footprint_ll=pts, height_m=height)


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


def load_dem_vertices(dem_paths: list[Path], bbox_ll: Polygon, pad_deg: float = 0.012):
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


def make_terrain_interpolator(
    dem_lons: np.ndarray, dem_lats: np.ndarray, dem_zs: np.ndarray
) -> LinearNDInterpolator:
    pts = np.column_stack([dem_lons, dem_lats])
    return LinearNDInterpolator(pts, dem_zs, fill_value=np.nan)


def terrain_factory(interp: LinearNDInterpolator, inv_tf: Transformer) -> sg.TerrainFn:
    """平面直角 XY [m] で標高を返すラッパ（内部で経緯度補間）。"""

    def terrain(xm: float, ym: float) -> float:
        lon, lat = inv_tf.transform(float(xm), float(ym))
        z = interp(np.array([[lon, lat]], dtype=np.float64))
        z = float(np.asarray(z).ravel()[0])
        if not np.isfinite(z):
            return 0.0
        return z

    return terrain


def sun_vector_enu(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    el = np.radians(90.0 - zenith_deg)
    az = np.radians(azimuth_deg)
    sx = np.cos(el) * np.sin(az)
    sy = np.cos(el) * np.cos(az)
    sz = np.sin(el)
    v = np.array([sx, sy, sz], dtype=np.float64)
    return v / np.linalg.norm(v)


def horizontal_shadow_length_flat_ground(H: float, sun_unit: np.ndarray) -> float:
    """
    【検証用】地面が水平で足場標高が一定、建物が完全に鉛直な柱のとき、
    屋上一点の影が水平面上で移動する距離（2D）は

        L = (H / sz) * sqrt(sx^2 + sy^2)

    太陽高度が高いほど sz が大きく L は短く、低いほど影は長くなる。
    """
    s = np.asarray(sun_unit, dtype=np.float64)
    s = s / np.linalg.norm(s)
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    if sz <= 1e-9:
        return 0.0
    u = H / sz
    return float(u * np.hypot(sx, sy))


def verify_building_shadow_length(
    ring_xy: np.ndarray,
    z_foot: np.ndarray,
    H: float,
    sun_unit: np.ndarray,
    terrain: sg.TerrainFn,
) -> dict[str, float]:
    """
    代表点（底面の重心）で、理論値（平坦地近似）と DEM 上の光線追跡結果を比較。
    """
    xy = np.asarray(ring_xy, dtype=np.float64)
    c = xy.mean(axis=0)
    cx, cy = float(c[0]), float(c[1])
    zg = float(np.mean(z_foot))
    z_roof = zg + float(H)
    L_flat = horizontal_shadow_length_flat_ground(H, sun_unit)
    hit = sg.sun_ray_intersect_terrain(cx, cy, z_roof, sun_unit, terrain)
    if hit is None:
        return {"H_m": H, "L_flat_m": L_flat, "L_dem_m": float("nan"), "delta_m": float("nan")}
    L_dem = float(np.hypot(hit[0] - cx, hit[1] - cy))
    return {
        "H_m": H,
        "L_flat_m": L_flat,
        "L_dem_m": L_dem,
        "delta_m": L_dem - L_flat,
        "delta_pct": (L_dem - L_flat) / L_flat * 100.0 if L_flat > 1e-6 else 0.0,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    data_dir = root / "plateaudata"
    gml_files = sorted(data_dir.glob("*_bldg_*_op.gml"))
    dem_files = sorted(data_dir.glob("*_dem_*_op.gml"))
    if not gml_files or not dem_files:
        print("plateaudata に建物または DEM の GML がありません。", file=sys.stderr)
        return 1

    # 解析範囲（近江町〜兼六園周辺）
    omicho_lat, omicho_lon = 36.57125, 136.65694
    kenrokuen_lat, kenrokuen_lon = 36.56204, 136.66245
    buf_deg = 0.0022
    lat_min = min(omicho_lat, kenrokuen_lat) - buf_deg
    lat_max = max(omicho_lat, kenrokuen_lat) + buf_deg
    lon_min = min(omicho_lon, kenrokuen_lon) - buf_deg
    lon_max = max(omicho_lon, kenrokuen_lon) + buf_deg
    bbox_ll = box(lon_min, lat_min, lon_max, lat_max)

    ts = pd.Timestamp("2026-08-05 14:00:00", tz="Asia/Tokyo")
    ref_lat = (omicho_lat + kenrokuen_lat) / 2.0
    ref_lon = (omicho_lon + kenrokuen_lon) / 2.0
    sol = pvlib.solarposition.get_solarposition(ts, ref_lat, ref_lon)
    zenith = float(sol["apparent_zenith"].iloc[0])
    azimuth = float(sol["azimuth"].iloc[0])
    elevation = 90.0 - zenith
    if elevation <= 0:
        print("指定時刻は太陽が地平線下です。", file=sys.stderr)
        return 1

    sun = sun_vector_enu(azimuth, zenith)
    transformer = Transformer.from_crs("EPSG:6697", "EPSG:6670", always_xy=True)
    inv_tf = Transformer.from_crs("EPSG:6670", "EPSG:6697", always_xy=True)

    dem_lons, dem_lats, dem_zs = load_dem_vertices(dem_files, bbox_ll)
    if dem_lons is None:
        print("DEM 読込失敗", file=sys.stderr)
        return 1
    dem_interp = make_terrain_interpolator(dem_lons, dem_lats, dem_zs)
    terrain = terrain_factory(dem_interp, inv_tf)

    bx, by = transformer.transform(
        np.array([lon_min, lon_max, lon_min, lon_max]),
        np.array([lat_min, lat_min, lat_max, lat_max]),
    )
    xmin, xmax = float(np.min(bx)), float(np.max(bx))
    ymin, ymax = float(np.min(by)), float(np.max(by))
    study = box(xmin, ymin, xmax, ymax)

    buildings: list[Building] = []
    for gml in gml_files:
        for b in iter_buildings_gml(gml):
            poly_ll = Polygon([(p[1], p[0]) for p in b.footprint_ll])
            if not poly_ll.intersects(bbox_ll):
                continue
            buildings.append(b)

    shadow_parts: list = []
    foot_patches: list[np.ndarray] = []
    # 検証: 範囲内で最も高い建物を1棟選ぶ
    best_b: Building | None = None
    best_ring: np.ndarray | None = None
    best_zf: np.ndarray | None = None

    for b in buildings:
        lat = b.footprint_ll[:, 0]
        lon = b.footprint_ll[:, 1]
        xs, ys = transformer.transform(lon, lat)
        ring = np.column_stack([xs, ys])
        zf = np.array([terrain(float(ring[i, 0]), float(ring[i, 1])) for i in range(len(ring))])
        fp = Polygon(ring)
        if fp.is_empty or not fp.intersects(study):
            continue
        sh = sg.footprint_building_shadow(ring, zf, b.height_m, sun, terrain, subdivisions_per_edge=1)
        foot_patches.append(ring)
        if not sh.is_empty:
            sh = sh.intersection(study)
            if not sh.is_empty:
                if sh.geom_type == "MultiPolygon":
                    shadow_parts.extend(list(sh.geoms))
                else:
                    shadow_parts.append(sh)
        if best_b is None or b.height_m > best_b.height_m:
            best_b = b
            best_ring = ring
            best_zf = zf

    shade_union = unary_union(shadow_parts) if shadow_parts else Polygon()

    # --- 検証レポート（コンソール）---
    print("=== 日陰の長さと建物高さの対応（検証） ===")
    print(
        f"日時: {ts}  太陽高度 {elevation:.2f}°  方位角 {azimuth:.2f}°（北0°・時計回り）"
    )
    ratio_hs = float(np.hypot(sun[0], sun[1]) / sun[2])
    print(
        "※ 水平地盤の簡易式: 影の水平移動距離 L = H * sqrt(sx^2+sy^2) / sz "
        f"（本条件で sqrt(sx^2+sy^2)/sz = {ratio_hs:.4f}）"
    )
    if best_b is not None and best_ring is not None and best_zf is not None:
        rep = verify_building_shadow_length(best_ring, best_zf, best_b.height_m, sun, terrain)
        print(
            f"代表建物（範囲内で最高い measuredHeight）: H = {rep['H_m']:.2f} m"
        )
        print(f"  → 平坦地近似の影の水平距離 L_flat = {rep['L_flat_m']:.2f} m")
        print(f"  → DEM＋光線追跡の重心代表点 L_dem   = {rep['L_dem_m']:.2f} m")
        dm = rep.get("delta_m", float("nan"))
        if np.isfinite(dm):
            pct = rep.get("delta_pct", 0.0)
            print(f"  → 差 Δ = {dm:+.2f} m ({pct:+.1f} %)")
        print(
            "解説: 差が出る主因は (1) 地面の傾斜・DEM 補間 (2) 図は全建物影の重なりで、"
            "1棟だけの影の端まで読み取るのは難しい、の両方です。比率が近ければ計算とデータは整合しています。"
        )
    print("==========================================")

    plt.rcParams["font.sans-serif"] = [
        "MS Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "Noto Sans CJK JP",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 12.5), dpi=140)
    ax.set_facecolor("#e8e4dc")

    gx = np.linspace(xmin, xmax, 300)
    gy = np.linspace(ymin, ymax, 300)
    GX, GY = np.meshgrid(gx, gy)
    GLon, GLat = inv_tf.transform(GX.ravel(), GY.ravel())
    GZ = dem_interp(np.column_stack([GLon, GLat])).reshape(GX.shape)
    tcf = ax.contourf(
        GX, GY, GZ, levels=26, cmap="gist_earth", alpha=0.42, zorder=0, extend="both"
    )
    fig.colorbar(tcf, ax=ax, shrink=0.5, label="地形の高さ（DEM）［m］")

    if not shade_union.is_empty:
        geoms = [shade_union] if shade_union.geom_type != "MultiPolygon" else list(shade_union.geoms)
        for g in geoms:
            ax.fill(
                *zip(*g.exterior.coords),
                facecolor="#1a237e",
                edgecolor="none",
                alpha=0.5,
                zorder=2,
            )

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPoly

    ax.add_collection(
        PatchCollection(
            [MplPoly(p, closed=True) for p in foot_patches],
            facecolor="#6d4c41",
            edgecolor="#3e2723",
            linewidths=0.2,
            alpha=0.92,
            zorder=3,
        )
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(
        "金沢（近江町〜兼六園周辺）日陰のイメージ\n"
        "PLATEAU 建物高さ・外形 ＋ 地形（DEM）＋ 太陽位置（2026-08-05 14:00 JST）",
        fontsize=12,
    )
    ax.set_xlabel("平面直角座標 第VI系 X［m］（位置の正しさ用。影の長さもメートルで比例）")
    ax.set_ylabel("平面直角座標 第VI系 Y［m］")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.45)

    # 専門外向け説明＋検証サマリ（図内ボックス）
    explain = (
        "【この図の読み方（やさしく）】\n"
        "・色の起伏＝デジタル地形データ（DEM）による地面の高さのイメージです。\n"
        "・茶色＝建物の平面の形。濃い青＝その時間帯に日陰になりやすい地面の範囲の計算結果です。\n"
        "・実際の日当たりは天気・窓・細かい建物などでも変わります。あくまでデータに基づく近似です。\n"
        "\n"
        "【影の長さと建物の高さ】\n"
        "・太陽が低いほど影は長く、高いほど短くなります（学校の理科の柱の影と同じ考え方）。\n"
        "・地面が平らなら「高さ H の建物」に対し、水平方向の影ののばしはおおよそ H と太陽の高さから決まります。\n"
        "・この図では地面の凹凸も入れるため、平らな場合の式と数メートル単位で前後することがあります。"
    )
    if best_b is not None and best_ring is not None and best_zf is not None:
        rep = verify_building_shadow_length(best_ring, best_zf, best_b.height_m, sun, terrain)
        explain += (
            f"\n\n【自動チェック・1棟サンプル（最高い建物 H≈{best_b.height_m:.1f}m）】\n"
            f"・平坦地だけの式での影ののばし ≈ {rep['L_flat_m']:.1f} m\n"
            f"・地形を考えた計算（代表点）≈ {rep['L_dem_m']:.1f} m\n"
        )
        if np.isfinite(rep.get("delta_m", np.nan)):
            explain += f"・差は主に地面の傾斜・データ補間のため（Δ≈{rep['delta_m']:+.1f} m）。\n"
        explain += (
            "・図では多くの建物の影が重なるため、地図上で1本の影の長さだけ測ると一致しないことがあります。"
        )

    ax.text(
        0.02,
        0.98,
        explain,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88, edgecolor="#555"),
        family="sans-serif",
        zorder=10,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"shadow_kanazawa_2026-08-05_1400_{stamp}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"出力（上書きなし）: {out}")
    print(f"建物描画数: {len(foot_patches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
