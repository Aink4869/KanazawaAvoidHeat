"""統合版ルート探索（日陰コスト＋クーリングシェルター立ち寄り経由）。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from integrated.cs_spatial import (
    CoolingShelterSite,
    candidate_visit_cs_indices,
    get_cs_by_id,
    load_cs_catalog,
)
from integrated.graph_cache import (
    cache_get_prepared,
    cache_put_prepared,
    get_cs_snap_nodes,
    get_undirected_cost_graph,
    get_walk_graph_projected,
    last_graph_load_source,
    prepared_cache_key,
)
from nav_shadow import (
    RouteFactors,
    ShadowGrid,
    _path_edge_attr,
    nearest_node,
    resolve_route_mask,
    sample_edge_shadow_fraction,
    time_index,
    _node_elev,
)

# 自動 CS 選定で評価する候補数（少ないほど速い・精度はやや落ちる）
AUTO_CS_CANDIDATES = 8
AUTO_CS_CANDIDATES_FALLBACK = (15, 30)
SHADOW_SAMPLES_PER_EDGE = 3

# 精度比較の基準（最適化前に近い設定）
REFERENCE_CS_CANDIDATES = 57
REFERENCE_SHADOW_SAMPLES_PER_EDGE = 8


@dataclass
class RouteFactorsIntegrated(RouteFactors):
    use_cooling_shelter: bool = True
    weight_cooling_shelter: float = 2.5
    cs_dwell_time_s: float = 600.0
    prefer_shade: bool = False
    include_tree_shade: bool = False


def _factors_cache_tuple(factors: RouteFactorsIntegrated) -> tuple:
    return (
        factors.use_distance,
        factors.use_time,
        factors.use_slope,
        factors.use_shadow,
        factors.prefer_shade,
        bool(getattr(factors, "include_tree_shade", False)),
        round(factors.weight_distance, 3),
        round(factors.weight_time, 3),
        round(factors.weight_slope, 3),
        round(factors.weight_shadow, 3),
        round(factors.walk_speed_m_s, 2),
    )


def prepare_graph_nav_costs_integrated(
    G_proj: nx.MultiDiGraph,
    mask: np.ndarray,
    grid: ShadowGrid,
    factors: RouteFactorsIntegrated,
    *,
    shadow_samples: int | None = None,
) -> nx.MultiDiGraph:
    if not any(
        [factors.use_distance, factors.use_time, factors.use_slope, factors.use_shadow]
    ):
        raise ValueError(
            "距離・時間・勾配・日陰のうち、少なくとも1つを有効にしてください"
        )

    tf = grid.transform
    ny_g, nx_g = grid.ny, grid.nx
    bounds = (grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    eps = 1e-9
    n_samples = (
        shadow_samples if shadow_samples is not None else SHADOW_SAMPLES_PER_EDGE
    )

    need_shadow = factors.use_shadow
    need_slope = factors.use_slope and grid.elev_m is not None

    for _u, _v, _k, data in G_proj.edges(keys=True, data=True):
        geom = data.get("geometry")
        length = float(data.get("length", 1.0))
        if length < 1e-6:
            length = 1e-6

        if need_shadow:
            if geom is not None:
                frac = sample_edge_shadow_fraction(
                    mask, tf, ny_g, nx_g, geom, n_samples=n_samples, grid_bounds=bounds
                )
            else:
                n1, n2 = G_proj.nodes[_u], G_proj.nodes[_v]
                ls = LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])])
                frac = sample_edge_shadow_fraction(
                    mask, tf, ny_g, nx_g, ls, n_samples=n_samples, grid_bounds=bounds
                )
        else:
            frac = 0.0

        shade_cost_frac = (1.0 - frac) if factors.prefer_shade else frac

        if need_slope:
            zu = _node_elev(G_proj, _u, grid.elev_m, tf, ny_g, nx_g, grid_bounds=bounds)
            zv = _node_elev(G_proj, _v, grid.elev_m, tf, ny_g, nx_g, grid_bounds=bounds)
            if np.isfinite(zu) and np.isfinite(zv):
                grade = abs(float(zv - zu) / length)
            else:
                grade = 0.0
        else:
            grade = 0.0

        term_d = factors.weight_distance * length if factors.use_distance else 0.0
        term_t = (
            factors.weight_time * (length / factors.walk_speed_m_s)
            if factors.use_time
            else 0.0
        )
        term_s = factors.weight_slope * length * grade if factors.use_slope else 0.0
        term_sh = (
            factors.weight_shadow * length * shade_cost_frac if factors.use_shadow else 0.0
        )

        nav_cost = eps * length + term_d + term_t + term_s + term_sh
        data["shadow_fraction"] = frac
        data["slope_grade_abs"] = grade
        data["nav_cost"] = max(1e-12, nav_cost)

    return G_proj


def _get_prepared_graph(
    Gp: nx.MultiDiGraph,
    mask: np.ndarray,
    grid: ShadowGrid,
    factors: RouteFactorsIntegrated,
    bbox_key: tuple,
    ti: int,
    *,
    shadow_samples: int | None = None,
    shadow_cell_m: float = 12.0,
) -> nx.MultiDiGraph:
    samples = (
        shadow_samples if shadow_samples is not None else SHADOW_SAMPLES_PER_EDGE
    )
    cell = 2.0 if float(shadow_cell_m) <= 2.5 else 12.0
    pkey = prepared_cache_key(
        bbox_key, ti, _factors_cache_tuple(factors) + (samples, cell)
    )
    cached = cache_get_prepared(pkey)
    if cached is not None:
        return cached
    Gp_prep = Gp.copy()
    prepare_graph_nav_costs_integrated(
        Gp_prep, mask, grid, factors, shadow_samples=samples
    )
    cache_put_prepared(pkey, Gp_prep)
    return Gp_prep


def _directed_cs_leg_costs(
    Gp: nx.MultiDiGraph, on, dn
) -> tuple[dict, dict]:
    """起点→CS、CS→終点の有向コスト（全 CS 選定用）。"""
    dist_on = nx.single_source_dijkstra_path_length(Gp, on, weight="nav_cost")
    Gr = Gp.reverse(copy=False)
    dist_to_dn = nx.single_source_dijkstra_path_length(Gr, dn, weight="nav_cost")
    return dist_on, dist_to_dn


def _undirected_cs_leg_costs(
    Gp: nx.MultiDiGraph, on, dn
) -> tuple[dict, dict]:
    """無向近似（旧高速化の比較用）。"""
    Gu = get_undirected_cost_graph(Gp)
    dist_on = nx.single_source_dijkstra_path_length(Gu, on, weight="weight")
    dist_from_dn = nx.single_source_dijkstra_path_length(Gu, dn, weight="weight")
    return dist_on, dist_from_dn


def _concat_paths(p1: list, p2: list) -> list:
    if not p1:
        return list(p2)
    if not p2:
        return list(p1)
    if p1[-1] == p2[0]:
        return p1 + p2[1:]
    return p1 + p2


def _path_cost(G: nx.MultiDiGraph, path: list) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if u not in G or v not in G[u]:
            continue
        best = min((float(d.get("nav_cost", 0.0)) for d in G[u][v].values()), default=0.0)
        total += best
    return total


def _shortest_path(
    G: nx.MultiDiGraph, source, target
) -> tuple[list, float] | None:
    if source == target:
        return [source], 0.0
    try:
        _dist, path = nx.single_source_dijkstra(
            G, source, target=target, weight="nav_cost"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return path, _path_cost(G, path)


def _precompute_cs_nodes(
    Gp: nx.MultiDiGraph,
    catalog: list[CoolingShelterSite],
    bbox_key: tuple[float, float, float, float],
) -> list:
    sites_xy = [(s.x, s.y) for s in catalog]
    return get_cs_snap_nodes(bbox_key, Gp, sites_xy)


def _route_via_cs(
    Gp: nx.MultiDiGraph,
    on,
    dn,
    cn,
    *,
    dwell_cost: float,
) -> tuple[list, float, dict] | None:
    if cn == on == dn:
        leg = _shortest_path(Gp, on, dn)
        if leg is None:
            return None
        path, cost = leg
        return path, cost + dwell_cost, {"legs": 1, "visit_at_end": True}

    if cn == dn:
        leg = _shortest_path(Gp, on, cn)
        if leg is None:
            return None
        path, cost = leg
        return path, cost + dwell_cost, {"legs": 1, "visit_at_end": True}

    if cn == on:
        leg = _shortest_path(Gp, on, dn)
        if leg is None:
            return None
        path, cost = leg
        return path, cost + dwell_cost, {"legs": 1, "visit_at_start": True}

    leg1 = _shortest_path(Gp, on, cn)
    leg2 = _shortest_path(Gp, cn, dn)
    if leg1 is None or leg2 is None:
        return None
    path = _concat_paths(leg1[0], leg2[0])
    cost = leg1[1] + leg2[1] + dwell_cost
    return path, cost, {"legs": 2, "visit_at_end": False}


def _pick_auto_visit_cs(
    Gp: nx.MultiDiGraph,
    on,
    dn,
    orig_xy: tuple[float, float],
    dest_xy: tuple[float, float],
    cs_nodes: list,
    *,
    dwell_cost: float,
    max_candidates: int = AUTO_CS_CANDIDATES,
    selection: str = "directed",
) -> tuple[list, float, CoolingShelterSite, dict] | None:
    """全源2回 + 候補のみ経路復元。selection: directed | undirected。"""
    catalog = load_cs_catalog()
    n_cs = len(catalog)
    if max_candidates >= n_cs:
        indices = list(range(n_cs))
    else:
        indices = candidate_visit_cs_indices(
            orig_xy, dest_xy, max_candidates=max_candidates
        )

    try:
        if selection == "undirected":
            dist_on, dist_leg2 = _undirected_cs_leg_costs(Gp, on, dn)
        else:
            dist_on, dist_leg2 = _directed_cs_leg_costs(Gp, on, dn)
    except nx.NodeNotFound:
        return None

    best_idx: int | None = None
    best_sel_cost = float("inf")
    for idx in indices:
        cn = cs_nodes[idx]
        d1 = dist_on.get(cn)
        d2 = dist_leg2.get(cn)
        if d1 is None or d2 is None:
            continue
        cost = float(d1) + float(d2) + dwell_cost
        if cost < best_sel_cost:
            best_sel_cost = cost
            best_idx = idx

    if best_idx is None:
        return None

    site = catalog[best_idx]
    cn = cs_nodes[best_idx]
    got = _route_via_cs(Gp, on, dn, cn, dwell_cost=dwell_cost)
    if got is None:
        return None
    path, cost, meta = got
    meta["candidates_evaluated"] = len(indices)
    meta["candidates_shortlisted"] = max_candidates
    meta["selection_method"] = selection
    meta["selection_cost_estimate"] = float(best_sel_cost)
    return path, cost, site, meta


def _chosen_edges(G: nx.MultiDiGraph, path: list) -> list[dict]:
    """Dijkstra が歩いた辺（nav_cost 最小）。平行辺の別属性の最小値は混ぜない。"""
    rows: list[dict] = []
    for u, v in zip(path[:-1], path[1:]):
        if u not in G or v not in G[u]:
            continue
        _k, data = min(
            G[u][v].items(),
            key=lambda kv: (float(kv[1].get("nav_cost", 0.0)), str(kv[0])),
        )
        rows.append(data)
    return rows


def _length_weighted(rows: list[dict], attr: str) -> float:
    num = 0.0
    den = 0.0
    for data in rows:
        length = float(data.get("length", 0.0) or 0.0)
        if length <= 0:
            continue
        num += length * float(data.get(attr, 0.0) or 0.0)
        den += length
    return num / den if den else 0.0


def _stats_snapshot(
    path: list,
    Gp: nx.MultiDiGraph,
    visited_site: CoolingShelterSite,
    dwell_cost: float,
    factors: RouteFactorsIntegrated,
    visit_meta: dict,
) -> dict:
    rows = _chosen_edges(Gp, path)
    lengths = [float(d.get("length", 0.0) or 0.0) for d in rows]
    nav_lens = [float(d.get("nav_cost", 0.0) or 0.0) for d in rows]
    walk_cost = float(sum(nav_lens))
    return {
        "cs_id": visited_site.cs_id,
        "cs_name": visited_site.name,
        "path_length_geometric_m": float(sum(lengths)),
        "path_walk_cost": walk_cost,
        "path_cost_total": walk_cost + dwell_cost,
        "mean_shadow_fraction_on_path": _length_weighted(rows, "shadow_fraction"),
        "selection_cost_estimate": visit_meta.get("selection_cost_estimate"),
        "candidates_evaluated": visit_meta.get("candidates_evaluated"),
        "selection_method": visit_meta.get("selection_method"),
    }


def route_shadow_aware_integrated(
    grid: ShadowGrid,
    when: date,
    hour_local: int,
    orig_xy: tuple[float, float],
    dest_xy: tuple[float, float],
    *,
    factors: RouteFactorsIntegrated | None = None,
    visit_cs_id: int | None = None,
    network_type: str = "walk",
    auto_cs_max_candidates: int | None = None,
    auto_cs_fallbacks: tuple[int, ...] | None = None,
    shadow_samples: int | None = None,
    shadow_cell_m: float | None = None,
    compare_accuracy: bool = False,
    data_dir=None,
    allow_flat_fallback: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], dict]:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    if factors is None:
        factors = RouteFactorsIntegrated()
    use_cs = bool(factors.use_cooling_shelter)

    mask, active_grid, ti, shadow_fallback = resolve_route_mask(
        grid,
        when,
        hour_local,
        data_dir=data_dir,
        allow_flat_fallback=allow_flat_fallback,
    )

    from nav_tree_shade import apply_tree_residual, build_observation_2m
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parent.parent
    cell_used = 2.0 if shadow_cell_m is not None and float(shadow_cell_m) <= 2.5 else 12.0
    loaded = None
    if cell_used == 2.0 and data_dir is not None:
        from nav_shadow import load_lod2_2m_observation

        loaded = load_lod2_2m_observation(
            _Path(data_dir),
            hour_local,
            include_tree_shade=getattr(factors, "include_tree_shade", False),
            when=when,
        )
    if loaded is not None:
        mask, active_grid, tree_info = loaded
    elif cell_used == 2.0:
        fine, fine_elev, tree_info = build_observation_2m(
            mask,
            active_grid,
            when,
            hour_local,
            include_tree_shade=getattr(factors, "include_tree_shade", False),
            root=_root,
        )
        mask = fine
        active_grid = ShadowGrid(
            masks=fine[np.newaxis, :, :],
            elev_m=fine_elev,
            xmin=active_grid.xmin,
            xmax=active_grid.xmax,
            ymin=active_grid.ymin,
            ymax=active_grid.ymax,
            nx=int(fine.shape[1]),
            ny=int(fine.shape[0]),
            year=active_grid.year,
            date_start=active_grid.date_start,
            date_end=active_grid.date_end,
            hours=list(active_grid.hours),
            model=active_grid.model,
        )
    else:
        mask, residual, tree_info = apply_tree_residual(
            mask,
            active_grid,
            when,
            hour_local,
            include_tree_shade=getattr(factors, "include_tree_shade", False),
            root=_root,
        )

    to_wgs = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    olon, olat = to_wgs.transform(orig_xy[0], orig_xy[1])
    dlon, dlat = to_wgs.transform(dest_xy[0], dest_xy[1])
    pad = 0.008
    lat_max, lat_min = max(olat, dlat) + pad, min(olat, dlat) - pad
    lon_max, lon_min = max(olon, dlon) + pad, min(olon, dlon) - pad

    t_graph = time.perf_counter()
    Gp = get_walk_graph_projected(lon_min, lat_min, lon_max, lat_max, network_type=network_type)
    timings["load_graph_s"] = time.perf_counter() - t_graph

    on = nearest_node(Gp, orig_xy[0], orig_xy[1])
    dn = nearest_node(Gp, dest_xy[0], dest_xy[1])

    bbox_key = (round(lon_min, 4), round(lat_min, 4), round(lon_max, 4), round(lat_max, 4))
    t_prep = time.perf_counter()
    if loaded is not None and tree_info.get("shade_date"):
        cache_ti = date.fromisoformat(str(tree_info["shade_date"])).toordinal() * 24 + int(
            tree_info.get("shade_hour", hour_local)
        )
    else:
        cache_ti = ti
    Gp = _get_prepared_graph(
        Gp,
        mask,
        active_grid,
        factors,
        bbox_key,
        cache_ti,
        shadow_samples=shadow_samples,
        shadow_cell_m=cell_used,
    )
    timings["prepare_costs_s"] = time.perf_counter() - t_prep

    catalog = load_cs_catalog()
    t_cs = time.perf_counter()
    cs_nodes = _precompute_cs_nodes(Gp, catalog, bbox_key) if use_cs else []
    timings["cs_nodes_s"] = time.perf_counter() - t_cs

    visited_site: CoolingShelterSite | None = None
    dwell_cost = factors.weight_cooling_shelter * factors.cs_dwell_time_s if use_cs else 0.0
    visit_meta: dict = {}
    cs_cap = (
        auto_cs_max_candidates
        if auto_cs_max_candidates is not None
        else AUTO_CS_CANDIDATES
    )
    fallbacks = (
        auto_cs_fallbacks
        if auto_cs_fallbacks is not None
        else AUTO_CS_CANDIDATES_FALLBACK
    )
    samples_used = (
        shadow_samples if shadow_samples is not None else SHADOW_SAMPLES_PER_EDGE
    )

    t_route = time.perf_counter()
    if not use_cs:
        try:
            path = nx.shortest_path(Gp, on, dn, weight="nav_cost")
        except nx.NetworkXNoPath as e:
            raise ValueError("起点から終点への歩道ルートが見つかりませんでした") from e
        visit_meta = {"mode": "none", "legs": 1}
    elif visit_cs_id is not None:
        site = get_cs_by_id(visit_cs_id)
        if site is None:
            raise ValueError(f"cs_id={visit_cs_id} のクーリングシェルターが見つかりません")
        cn = nearest_node(Gp, site.x, site.y)
        got = _route_via_cs(Gp, on, dn, cn, dwell_cost=dwell_cost)
        if got is None:
            raise ValueError(
                f"立ち寄り先 CS（{site.name}）を経由する歩道ルートが見つかりませんでした"
            )
        path, _, visit_meta = got
        visited_site = site
        visit_meta["mode"] = "specified"
    else:
        auto = _pick_auto_visit_cs(
            Gp,
            on,
            dn,
            orig_xy,
            dest_xy,
            cs_nodes,
            dwell_cost=dwell_cost,
            max_candidates=cs_cap,
        )
        for fb in fallbacks:
            if auto is not None:
                break
            auto = _pick_auto_visit_cs(
                Gp,
                on,
                dn,
                orig_xy,
                dest_xy,
                cs_nodes,
                dwell_cost=dwell_cost,
                max_candidates=fb,
            )
        if auto is None:
            raise ValueError(
                "経由可能な CS ルートが見つかりませんでした。"
                "起点・終点を変えるか、③で立ち寄り CS を指定してください。"
            )
        path, _, visited_site, visit_meta = auto
        visit_meta["mode"] = "auto"

    timings["route_search_s"] = time.perf_counter() - t_route

    accuracy_delta: dict | None = None
    if compare_accuracy and visit_cs_id is None and visited_site is not None:
        t_acc = time.perf_counter()
        ref = _pick_auto_visit_cs(
            Gp,
            on,
            dn,
            orig_xy,
            dest_xy,
            cs_nodes,
            dwell_cost=dwell_cost,
            max_candidates=REFERENCE_CS_CANDIDATES,
            selection="directed",
        )
        timings["accuracy_compare_s"] = time.perf_counter() - t_acc
        if ref is not None:
            from integrated.accuracy import diff_route_snapshots

            ref_path, _, ref_site, ref_meta = ref
            fast_snap = _stats_snapshot(
                path, Gp, visited_site, dwell_cost, factors, visit_meta
            )
            ref_snap = _stats_snapshot(
                ref_path, Gp, ref_site, dwell_cost, factors, ref_meta
            )
            accuracy_delta = diff_route_snapshots(
                fast_snap,
                ref_snap,
                profile_fast={
                    "cs_candidates": cs_cap,
                    "shadow_samples_per_edge": samples_used,
                },
                profile_reference={
                    "cs_candidates": REFERENCE_CS_CANDIDATES,
                    "shadow_samples_per_edge": REFERENCE_SHADOW_SAMPLES_PER_EDGE,
                    "note": "同一コスト付きグラフ上の CS 選定比較（日陰サンプル差は別ベンチ参照）",
                },
            )

    if use_cs and visited_site is None:
        raise RuntimeError("CS 立ち寄りが確定しませんでした（内部エラー）")

    coords_xy = [(float(Gp.nodes[n]["x"]), float(Gp.nodes[n]["y"])) for n in path]

    rows = _chosen_edges(Gp, path)
    lengths = [float(d.get("length", 0.0) or 0.0) for d in rows]
    fracs = [float(d.get("shadow_fraction", 0.0) or 0.0) for d in rows]
    nav_lens = [float(d.get("nav_cost", 0.0) or 0.0) for d in rows]

    timings["total_s"] = time.perf_counter() - t0

    stats = {
        "timestep_index": int(ti),
        "shadow_model": tree_info.get("shadow_model")
        or (
            (
                f"{active_grid.model}+tree_residual"
                if tree_info.get("tree_shade_applied")
                else active_grid.model
            )
            + ("+obs2m" if cell_used == 2.0 else "")
        ),
        "shadow_fallback_to_flat": shadow_fallback,
        "path_edges": len(path) - 1,
        "path_length_geometric_m": float(sum(lengths)),
        "path_cost_total": float(sum(nav_lens)) + dwell_cost,
        "path_walk_cost": float(sum(nav_lens)),
        "cs_dwell_cost": float(dwell_cost),
        "cs_visit_required": use_cs,
        "mean_shadow_fraction_on_path": _length_weighted(rows, "shadow_fraction"),
        "mean_slope_grade_abs": _length_weighted(rows, "slope_grade_abs"),
        "approx_time_s": float(sum(lengths)) / factors.walk_speed_m_s
        + (factors.cs_dwell_time_s if use_cs else 0.0),
        "timing_s": {k: round(v, 3) for k, v in timings.items()},
        "graph_cached": timings.get("load_graph_s", 0) < 0.5,
        "graph_load_source": last_graph_load_source(),
        "cooling_shelter_visit": {
            "enabled": use_cs,
            "required_minimum": 1 if use_cs else 0,
            "mode": visit_meta.get("mode"),
            "legs": visit_meta.get("legs", 1),
            "cs_id": visited_site.cs_id if visited_site else None,
            "cs_name": visited_site.name if visited_site else None,
            "cs_address": visited_site.address if visited_site else None,
            "candidates_evaluated": visit_meta.get("candidates_evaluated"),
        },
        "factors": {
            "use_distance": factors.use_distance,
            "use_time": factors.use_time,
            "use_slope": factors.use_slope,
            "use_shadow": factors.use_shadow,
            "use_cooling_shelter": factors.use_cooling_shelter,
            "prefer_shade": factors.prefer_shade,
            "include_tree_shade": getattr(factors, "include_tree_shade", False),
            "weight_distance": factors.weight_distance,
            "weight_time": factors.weight_time,
            "weight_slope": factors.weight_slope,
            "weight_shadow": factors.weight_shadow,
            "weight_cooling_shelter": factors.weight_cooling_shelter,
            "cs_dwell_time_s": factors.cs_dwell_time_s,
            "walk_speed_m_s": factors.walk_speed_m_s,
        },
        "elev_grid_available": grid.elev_m is not None,
        "tree_shade": tree_info,
        "cooling_shelter_count": len(catalog),
        "optimization_profile": {
            "auto_cs_candidates": cs_cap if visit_cs_id is None else None,
            "shadow_samples_per_edge": samples_used,
            "shadow_cell_m": cell_used,
            "building_shade_native_m": tree_info.get("building_shade_native_m", 12.0),
            "tree_shade_native_m": tree_info.get("tree_shade_native_m", 12.0 if tree_info.get("tree_shade_applied") else None),
            "reference_cs_candidates": REFERENCE_CS_CANDIDATES,
            "reference_shadow_samples": REFERENCE_SHADOW_SAMPLES_PER_EDGE,
        },
    }
    if accuracy_delta is not None:
        stats["accuracy_vs_reference"] = accuracy_delta

    try:
        from integrated.nav_steps import build_nav_steps

        cs_node = nearest_node(Gp, visited_site.x, visited_site.y) if visited_site else None
        cs_name = visited_site.name if visited_site else None
        stats["edge_shade_fractions"] = [round(float(f), 3) for f in fracs]
        stats["nav_steps"] = build_nav_steps(
            Gp, path, waypoint_node=cs_node, waypoint_name=cs_name, lang="ja"
        )
        stats["nav_steps_en"] = build_nav_steps(
            Gp, path, waypoint_node=cs_node, waypoint_name=cs_name, lang="en"
        )
    except Exception as e:
        stats["nav_steps"] = []
        stats["nav_steps_en"] = []
        stats["nav_steps_error"] = str(e)

    to_ll = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    coords_ll = []
    for x, y in coords_xy:
        lo, la = to_ll.transform(x, y)
        coords_ll.append((la, lo))
    return coords_xy, coords_ll, stats
