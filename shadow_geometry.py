"""
建物 Brep（多角形底面＋鉛直壁＋一定屋根高）と地形標高関数 terrain(x, y) から、
幾何的に一貫した日影領域（2D 平面投影）を構築するモジュール。

幾何の意味（平行光線・太陽は無限遠）:
- 単位ベクトル s は「観測点から太陽へ向く方向」（ENU 等で X東・Y北・Z上）。
- 届く直射光の進行方向は -s。屋上点 P = (x, y, z_roof) から半直線
    L(u) = P - u * s  (u >= 0)
  を辿り、最初に「地形面 z = terrain(x, y)」と交わる (x, y) を影の着地点とする。
- 鉛直壁（底辺が (Gi, Gj)、上端が同じ (xi,yi), (xj,yj) で高さ z_roof）が落とす影は、
  一般に Gi-Gj を結ぶ底辺と、上端2点の着地点 Si-Sj を結ぶ辺で囲まれた単純四辺形
  (Gi, Gj, Sj, Si) として表せる（光が壁面を横切る場合の可視シルエットの地上投影）。

制約・前提:
- 凸包は使用しない。凹多角形は「壁ごとの四辺形」の和集合で表現する。
- terrain は単調で連続に近いと仮定し、交点を二分探索で求める。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

# 地形標高 [m]。平面座標 (x, y) は建物と同じ水平座標系（例: 平面直角）。
TerrainFn: TypeAlias = Callable[[float, float], float]

# 太陽方向単位ベクトル (sx, sy, sz)、Z 成分は昼間正を想定。
SunVector: TypeAlias = np.ndarray


def sun_ray_intersect_terrain(
    x0: float,
    y0: float,
    z_start: float,
    sun_unit: SunVector,
    terrain: TerrainFn,
    *,
    u_max: float = 8000.0,
    iters: int = 56,
) -> tuple[float, float] | None:
    """
    半直線 L(u) = (x0, y0, z_start) - u * s と曲面 z = terrain(x, y) の最初の交点を、
    スカラー u について二分探索で求め、(x_hit, y_hit) を返す。

    数学的には、残差
        f(u) = (z_start - u * sz) - terrain(x0 - u*sx, y0 - u*sy)
    の零点を探す。u=0 で屋上、u 増加で高度が下がり、地形と交差する最小の正の u を仮定。

    Parameters
    ----------
    x0, y0 : 鉛直線上の水平位置（壁の上端と一致）
    z_start : その点の屋根高さ（標高）
    sun_unit : 長さ1の太陽方向ベクトル (sx, sy, sz)
    terrain : 標高関数 z = terrain(x, y)

    Returns
    -------
    (x_hit, y_hit) または交点が見つからない場合 None
    """
    sx, sy, sz = float(sun_unit[0]), float(sun_unit[1]), float(sun_unit[2])
    if sz <= 1e-9:
        return None

    def z_ray(u: float) -> float:
        return z_start - u * sz

    def z_ground(u: float) -> float:
        return float(terrain(x0 - u * sx, y0 - u * sy))

    def residual(u: float) -> float:
        return z_ray(u) - z_ground(u)

    z0 = z_ground(0.0)
    if residual(0.0) <= 0.0:
        # 既に屋上が地形以下（数値誤差・データ不整合）
        return x0, y0

    lo, hi = 0.0, min((z_start - z0 + 100.0) / sz, u_max)
    for _ in range(24):
        if residual(hi) < 0.0:
            break
        hi = min(hi * 1.4, u_max)
    else:
        if residual(hi) >= 0.0:
            # 十分遠くまで延ばしても地形より上 → フォールバック
            u = (z_start - z0) / sz
            return x0 - u * sx, y0 - u * sy

    if residual(hi) >= 0.0:
        return None

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    u_hit = 0.5 * (lo + hi)
    return x0 - u_hit * sx, y0 - u_hit * sy


def iter_vertical_wall_edges(
    ring_xy: np.ndarray,
    z_foot: np.ndarray,
    building_height: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """
    多角形外周（CCW 推奨）の各辺について、鉛直壁の「上端2点」を返すイテレータ用リスト。

    - ring_xy : (n, 2) 各頂点の水平座標
    - z_foot[i] : 頂点 i の足場標高（DEM 等）
    - building_height : 一様な建物高（屋根は水平板とみなす）

    戻り値の各要素は ((xi,yi,z_roof_i), (xj,yj,z_roof_j))。
    """
    n = ring_xy.shape[0]
    if n < 3:
        return []
    edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for i in range(n):
        j = (i + 1) % n
        zi = float(z_foot[i]) + float(building_height)
        zj = float(z_foot[j]) + float(building_height)
        pi = (float(ring_xy[i, 0]), float(ring_xy[i, 1]), zi)
        pj = (float(ring_xy[j, 0]), float(ring_xy[j, 1]), zj)
        edges.append((pi, pj))
    return edges


def quad_shadow_strip(
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    sa_xy: tuple[float, float],
    sb_xy: tuple[float, float],
) -> Polygon | None:
    """
    壁上（または足場上）の水平投影2点 A,B と、それらの光線着地点 Sa,Sb から
    単純四辺形 (A, B, Sb, Sa) を構築する。

    鉛直壁1枚の場合、A=Gi・B=Gj（足場）、Sa,Si が屋上端からの着地点。
    屋根上辺を分割した場合は、A,B が屋上辺上の2点の水平投影（Gi,Gj と XY は一致）。
    """
    coords = [a_xy, b_xy, sb_xy, sa_xy, a_xy]
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-12:
            return None
        return poly
    except Exception:
        return None


def wall_edges_to_shadow_multipolygon(
    wall_edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    z_foot_vertices: np.ndarray,
    sun_unit: SunVector,
    terrain: TerrainFn,
    *,
    subdivisions_per_edge: int = 1,
    self_shadow_check: bool = False,
    footprint_2d: Polygon | None = None,
    roof_height_uniform: float | None = None,
) -> Polygon | MultiPolygon:
    """
    全鉛直壁について影四辺形を生成し、unary_union で合体した日影領域を返す。

    Parameters
    ----------
    wall_edges : iter_vertical_wall_edges の出力と同形式
    z_foot_vertices : 各頂点の足場標高（自己交差判定用）
    ring_xy, footprint_2d : self_shadow_check 時に建物水平投影へレイ内点判定で使用

    subdivisions_per_edge : 1 より大きい場合、**屋根上辺** Pi→Pj を等分割し、
        各小線分 (Tk,T{k+1}) について着地点 (Sk,S{k+1}) を求め四辺形を積む。
        鉛直壁の曲線束を折れ線で近似する（長壁で光線が地形と複数回交わる場合の誤差低減）。

    self_shadow_check : True のとき、簡易的にレイ先が建物占有柱体内に入るかを検査（ヒューリスティック）。

    roof_height_uniform : self_shadow 検査で用いる一様建物高（None なら頂点間の z_roof 差から推定）

    Returns
    -------
    shapely Polygon または MultiPolygon（影が無ければ空 Polygon）
    """
    sun_unit = np.asarray(sun_unit, dtype=np.float64)
    norm = np.linalg.norm(sun_unit)
    if norm < 1e-12:
        return Polygon()
    sun_unit = sun_unit / norm

    quads: list[Polygon] = []

    for (pi, pj) in wall_edges:
        xi, yi, zri = pi
        xj, yj, zrj = pj
        gi = (xi, yi)
        gj = (xj, yj)

        nsub = max(1, int(subdivisions_per_edge))
        for k in range(nsub):
            t0 = k / nsub
            t1 = (k + 1) / nsub
            p0 = (xi + t0 * (xj - xi), yi + t0 * (yj - yi), zri + t0 * (zrj - zri))
            p1 = (xi + t1 * (xj - xi), yi + t1 * (yj - yi), zri + t1 * (zrj - zri))
            a_xy = (p0[0], p0[1])
            b_xy = (p1[0], p1[1])
            if self_shadow_check and footprint_2d is not None:
                h_uni = roof_height_uniform
                if h_uni is None:
                    h_uni = max(zri, zrj) - float(np.min(z_foot_vertices))
                if _ray_hits_prism_simple(
                    p0[0], p0[1], p0[2], sun_unit, terrain, footprint_2d, z_foot_vertices, h_uni
                ):
                    continue
            s0 = sun_ray_intersect_terrain(p0[0], p0[1], p0[2], sun_unit, terrain)
            s1 = sun_ray_intersect_terrain(p1[0], p1[1], p1[2], sun_unit, terrain)
            if s0 is None or s1 is None:
                continue
            q = quad_shadow_strip(a_xy, b_xy, s0, s1)
            if q is not None:
                quads.append(q)

    if not quads:
        return Polygon()
    merged = unary_union(quads)
    if merged.is_empty:
        return Polygon()
    if merged.geom_type == "Polygon":
        return merged
    if merged.geom_type == "MultiPolygon":
        return merged
    if merged.geom_type == "GeometryCollection":
        polys = [g for g in merged.geoms if g.geom_type == "Polygon"]
        if not polys:
            return Polygon()
        return unary_union(polys)
    return Polygon()


def _ray_hits_prism_simple(
    x0: float,
    y0: float,
    z0: float,
    sun_unit: SunVector,
    terrain: TerrainFn,
    footprint: Polygon,
    z_foot_min: np.ndarray,
    building_height: float,
    samples: int = 16,
) -> bool:
    """
    簡易 self-shadow: 光線 L(u)=P-u s 上に、建物プリズム内（足場 min(z_foot) ～ その＋高さ）の点が
    先に存在するかを粗く検査。厳密な Brep 交差ではない。
    """
    sx, sy, sz = float(sun_unit[0]), float(sun_unit[1]), float(sun_unit[2])
    if sz <= 1e-9:
        return False
    z_bot = float(np.min(z_foot_min))
    z_top_prism = z_bot + float(building_height)
    z_terr0 = float(terrain(x0, y0))
    u_max = max(0.0, (z0 - z_terr0) / sz) * 0.999
    for k in range(1, samples):
        u = (k / samples) * u_max
        x = x0 - u * sx
        y = y0 - u * sy
        z = z0 - u * sz
        p = Point(x, y)
        if not footprint.contains(p) and not footprint.touches(p):
            continue
        if z_bot - 1e-2 <= z <= z_top_prism + 1e-2:
            return True
    return False


def footprint_building_shadow(
    ring_xy: np.ndarray,
    z_foot_per_vertex: np.ndarray,
    building_height: float,
    sun_unit: SunVector,
    terrain: TerrainFn,
    *,
    subdivisions_per_edge: int = 1,
    self_shadow_check: bool = False,
    include_flat_roof: bool = False,
) -> Polygon | MultiPolygon:
    """
    底面が単一多角形（穴なし）の建物について、壁影から日影領域を構築するエントリポイント。

    include_flat_roof : True のとき、水平屋根面（外周＝屋根上辺と同じ）の影を
    「屋根多角形を光方向に投影」したポリゴンを追加（頂点ごとにレイ→着地点、順序維持）。
    """
    edges = iter_vertical_wall_edges(ring_xy, z_foot_per_vertex, building_height)
    try:
        fp_poly = Polygon(ring_xy)
        if not fp_poly.is_valid:
            fp_poly = fp_poly.buffer(0)
    except Exception:
        fp_poly = None
    wall_sh = wall_edges_to_shadow_multipolygon(
        edges,
        z_foot_per_vertex,
        sun_unit,
        terrain,
        subdivisions_per_edge=subdivisions_per_edge,
        self_shadow_check=self_shadow_check,
        footprint_2d=fp_poly,
        roof_height_uniform=building_height,
    )
    if not include_flat_roof:
        return wall_sh
    roof_sh = flat_roof_shadow_polygon(ring_xy, z_foot_per_vertex, building_height, sun_unit, terrain)
    if roof_sh.is_empty:
        return wall_sh
    return unary_union([wall_sh, roof_sh])


def flat_roof_shadow_polygon(
    ring_xy: np.ndarray,
    z_foot_per_vertex: np.ndarray,
    building_height: float,
    sun_unit: SunVector,
    terrain: TerrainFn,
) -> Polygon | MultiPolygon:
    """
    水平屋根（各頂点で z_roof = z_foot + h）の外形を、各屋根頂点からの光線着地点で
    順番どおり繋いだポリゴン。凹形でも頂点順を保つ（凸包は使わない）。

    注: 屋根面が傾いている場合は屋根三角形ごとの処理が必要。
    注: 凹多角形では頂点順ポリゴンが自己交差する場合があり buffer(0) で修復するが、
    厳密には屋根稜線ごとの帯の和集合が望ましい。
    """
    n = ring_xy.shape[0]
    if n < 3:
        return Polygon()
    pts: list[tuple[float, float]] = []
    for i in range(n):
        x, y = float(ring_xy[i, 0]), float(ring_xy[i, 1])
        zr = float(z_foot_per_vertex[i]) + float(building_height)
        hit = sun_ray_intersect_terrain(x, y, zr, sun_unit, terrain)
        if hit is None:
            return Polygon()
        pts.append(hit)
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else Polygon()
    except Exception:
        return Polygon()


def example_flat_terrain_demo() -> None:
    """
    terrain(x,y) = 一定（傾斜ゼロ）のとき、影の水平長さが h/tan(仰角) と一致することのデモ用。
    実行: python -c "from shadow_geometry import example_flat_terrain_demo; example_flat_terrain_demo()"
    """
    z0 = 10.0

    def terrain_flat(x: float, y: float) -> float:
        return z0

    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    zf = np.array([z0, z0, z0, z0], dtype=np.float64)
    h = 5.0
    # 東向き太陽（例）: 仰角約45°
    sun = np.array([0.70710678, 0.0, 0.70710678])
    sh = footprint_building_shadow(square, zf, h, sun, terrain_flat)
    print("影の面積 [m2]:", sh.area)
    print(sh)


if __name__ == "__main__":
    example_flat_terrain_demo()
