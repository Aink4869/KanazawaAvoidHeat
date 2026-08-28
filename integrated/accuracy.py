"""高速化前後のルート精度差分（記録・ベンチマーク用）。"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pyproj import Transformer

from integrated.route_ext import (
    REFERENCE_CS_CANDIDATES,
    REFERENCE_SHADOW_SAMPLES_PER_EDGE,
    SHADOW_SAMPLES_PER_EDGE,
    AUTO_CS_CANDIDATES,
    RouteFactorsIntegrated,
    _get_prepared_graph,
    _pick_auto_visit_cs,
    _precompute_cs_nodes,
    _stats_snapshot,
    get_walk_graph_projected,
    route_shadow_aware_integrated,
)
from nav_shadow import ShadowGrid, load_shadow_grid, nearest_node, time_index

REPO = Path(__file__).resolve().parent.parent
BENCHMARK_JSON = REPO / "data" / "route_accuracy_benchmark.json"
ACCURACY_MD = Path(__file__).resolve().parent / "ACCURACY.md"

# ベンチマーク用 OD（金沢市内・代表例）
BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": "station_to_korinbo",
        "label": "駅付近 → 香林坊",
        "orig_lon": 136.656,
        "orig_lat": 36.561,
        "dest_lon": 136.668,
        "dest_lat": 36.572,
    },
    {
        "id": "kenrokuen_to_station",
        "label": "兼六園口 → 駅付近",
        "orig_lon": 136.662,
        "orig_lat": 36.562,
        "dest_lon": 136.650,
        "dest_lat": 36.558,
    },
    {
        "id": "katamachi_to_higashichaya",
        "label": "片町 → 東茶屋",
        "orig_lon": 136.660,
        "orig_lat": 36.568,
        "dest_lon": 136.672,
        "dest_lat": 36.572,
    },
    {
        "id": "west_to_musashigatsuji",
        "label": "西側 → 武蔵ヶ辻",
        "orig_lon": 136.648,
        "orig_lat": 36.565,
        "dest_lon": 136.665,
        "dest_lat": 36.570,
    },
    {
        "id": "long_cross_city",
        "label": "市内横断（やや長め）",
        "orig_lon": 136.642,
        "orig_lat": 36.555,
        "dest_lon": 136.675,
        "dest_lat": 36.578,
    },
]


def _pct_delta(fast: float, ref: float) -> float | None:
    if ref == 0 or ref != ref:
        return None
    return round(100.0 * (fast - ref) / ref, 4)


def _abs_delta(fast: float, ref: float) -> float:
    return round(float(fast - ref), 6)


def diff_route_snapshots(
    fast: dict,
    ref: dict,
    *,
    profile_fast: dict | None = None,
    profile_reference: dict | None = None,
) -> dict:
    """本番（fast）と基準（reference）のルート指標差分。"""
    same_cs = fast.get("cs_id") == ref.get("cs_id")
    out: dict[str, Any] = {
        "same_cs": same_cs,
        "fast_cs_id": fast.get("cs_id"),
        "fast_cs_name": fast.get("cs_name"),
        "reference_cs_id": ref.get("cs_id"),
        "reference_cs_name": ref.get("cs_name"),
        "path_length_m": {
            "fast": fast.get("path_length_geometric_m"),
            "reference": ref.get("path_length_geometric_m"),
            "delta_m": _abs_delta(
                fast.get("path_length_geometric_m", 0),
                ref.get("path_length_geometric_m", 0),
            ),
            "delta_pct": _pct_delta(
                fast.get("path_length_geometric_m", 0),
                ref.get("path_length_geometric_m", 0),
            ),
        },
        "path_cost_total": {
            "fast": fast.get("path_cost_total"),
            "reference": ref.get("path_cost_total"),
            "delta": _abs_delta(
                fast.get("path_cost_total", 0), ref.get("path_cost_total", 0)
            ),
            "delta_pct": _pct_delta(
                fast.get("path_cost_total", 0), ref.get("path_cost_total", 0)
            ),
        },
        "mean_shadow_fraction": {
            "fast": fast.get("mean_shadow_fraction_on_path"),
            "reference": ref.get("mean_shadow_fraction_on_path"),
            "delta_abs": round(
                float(fast.get("mean_shadow_fraction_on_path", 0))
                - float(ref.get("mean_shadow_fraction_on_path", 0)),
                6,
            ),
        },
        "selection_cost_estimate": {
            "fast": fast.get("selection_cost_estimate"),
            "reference": ref.get("selection_cost_estimate"),
        },
    }
    if profile_fast:
        out["profile_fast"] = profile_fast
    if profile_reference:
        out["profile_reference"] = profile_reference
    return out


def _bbox_for_od(orig_xy, dest_xy) -> tuple:
    to_wgs = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    olon, olat = to_wgs.transform(orig_xy[0], orig_xy[1])
    dlon, dlat = to_wgs.transform(dest_xy[0], dest_xy[1])
    pad = 0.008
    lat_max, lat_min = max(olat, dlat) + pad, min(olat, dlat) - pad
    lon_max, lon_min = max(olon, dlon) + pad, min(olon, dlon) - pad
    bbox_key = (round(lon_min, 4), round(lat_min, 4), round(lon_max, 4), round(lat_max, 4))
    return lon_min, lat_min, lon_max, lat_max, bbox_key


def compare_cs_shortlist_on_graph(
    Gp,
    grid: ShadowGrid,
    mask,
    factors: RouteFactorsIntegrated,
    bbox_key: tuple,
    ti: int,
    orig_xy: tuple[float, float],
    dest_xy: tuple[float, float],
    *,
    shadow_samples: int = SHADOW_SAMPLES_PER_EDGE,
) -> dict:
    """同一コストグラフ上で CS 候補数のみ比較（8 vs 57）。"""
    Gp = _get_prepared_graph(
        Gp, mask, grid, factors, bbox_key, ti, shadow_samples=shadow_samples
    )
    on = nearest_node(Gp, orig_xy[0], orig_xy[1])
    dn = nearest_node(Gp, dest_xy[0], dest_xy[1])
    from integrated.cs_spatial import load_cs_catalog

    catalog = load_cs_catalog()
    cs_nodes = _precompute_cs_nodes(Gp, catalog, bbox_key)
    dwell = factors.weight_cooling_shelter * factors.cs_dwell_time_s

    fast = _pick_auto_visit_cs(
        Gp,
        on,
        dn,
        orig_xy,
        dest_xy,
        cs_nodes,
        dwell_cost=dwell,
        max_candidates=AUTO_CS_CANDIDATES,
        selection="directed",
    )
    ref = _pick_auto_visit_cs(
        Gp,
        on,
        dn,
        orig_xy,
        dest_xy,
        cs_nodes,
        dwell_cost=dwell,
        max_candidates=REFERENCE_CS_CANDIDATES,
        selection="directed",
    )
    undir = _pick_auto_visit_cs(
        Gp,
        on,
        dn,
        orig_xy,
        dest_xy,
        cs_nodes,
        dwell_cost=dwell,
        max_candidates=AUTO_CS_CANDIDATES,
        selection="undirected",
    )

    result: dict[str, Any] = {"shadow_samples_on_graph": shadow_samples}
    if fast and ref:
        fp, _, fs, fm = fast
        rp, _, rs, rm = ref
        result["shortlist_8_vs_57"] = diff_route_snapshots(
            _stats_snapshot(fp, Gp, fs, dwell, factors, fm),
            _stats_snapshot(rp, Gp, rs, dwell, factors, rm),
            profile_fast={"cs_candidates": AUTO_CS_CANDIDATES},
            profile_reference={"cs_candidates": REFERENCE_CS_CANDIDATES},
        )
    if fast and undir:
        fp, _, fs, fm = fast
        up, _, us, um = undir
        result["directed_vs_undirected_selection_8"] = diff_route_snapshots(
            _stats_snapshot(fp, Gp, fs, dwell, factors, fm),
            _stats_snapshot(up, Gp, us, dwell, factors, um),
            profile_fast={"selection": "directed"},
            profile_reference={"selection": "undirected"},
        )
    return result


def compare_shadow_samples(
    Gp,
    grid: ShadowGrid,
    mask,
    factors: RouteFactorsIntegrated,
    bbox_key: tuple,
    ti: int,
    orig_xy: tuple[float, float],
    dest_xy: tuple[float, float],
) -> dict:
    """日陰サンプル 3 vs 8（各グラフで CS57 選定）。"""
    dwell = factors.weight_cooling_shelter * factors.cs_dwell_time_s
    on = nearest_node(Gp, orig_xy[0], orig_xy[1])
    dn = nearest_node(Gp, dest_xy[0], dest_xy[1])
    from integrated.cs_spatial import load_cs_catalog

    catalog = load_cs_catalog()
    cs_nodes = _precompute_cs_nodes(Gp, catalog, bbox_key)

    def _pick(samples: int):
        Gs = _get_prepared_graph(
            Gp, mask, grid, factors, bbox_key, ti, shadow_samples=samples
        )
        return _pick_auto_visit_cs(
            Gs,
            on,
            dn,
            orig_xy,
            dest_xy,
            cs_nodes,
            dwell_cost=dwell,
            max_candidates=REFERENCE_CS_CANDIDATES,
            selection="directed",
        )

    fast_g = _pick(SHADOW_SAMPLES_PER_EDGE)
    ref_g = _pick(REFERENCE_SHADOW_SAMPLES_PER_EDGE)
    if not fast_g or not ref_g:
        return {"error": "shadow comparison path not found"}
    fp, _, fs, fm = fast_g
    rp, _, rs, rm = ref_g
    G_fast = _get_prepared_graph(
        Gp, mask, grid, factors, bbox_key, ti, shadow_samples=SHADOW_SAMPLES_PER_EDGE
    )
    G_ref = _get_prepared_graph(
        Gp,
        mask,
        grid,
        factors,
        bbox_key,
        ti,
        shadow_samples=REFERENCE_SHADOW_SAMPLES_PER_EDGE,
    )
    return diff_route_snapshots(
        _stats_snapshot(fp, G_fast, fs, dwell, factors, fm),
        _stats_snapshot(rp, G_ref, rs, dwell, factors, rm),
        profile_fast={"shadow_samples_per_edge": SHADOW_SAMPLES_PER_EDGE},
        profile_reference={
            "shadow_samples_per_edge": REFERENCE_SHADOW_SAMPLES_PER_EDGE
        },
    )


def run_benchmark_case(
    grid: ShadowGrid,
    when: date,
    hour: int,
    case: dict,
    factors: RouteFactorsIntegrated | None = None,
) -> dict:
    if factors is None:
        factors = RouteFactorsIntegrated(use_shadow=True, prefer_shade=True)

    tf = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    ox0, oy0 = tf.transform(case["orig_lon"], case["orig_lat"])
    dx0, dy0 = tf.transform(case["dest_lon"], case["dest_lat"])
    orig_xy, dest_xy = (ox0, oy0), (dx0, dy0)

    ti = time_index(grid, when, hour)
    mask = grid.masks[ti]
    lon_min, lat_min, lon_max, lat_max, bbox_key = _bbox_for_od(orig_xy, dest_xy)
    Gp = get_walk_graph_projected(lon_min, lat_min, lon_max, lat_max)

    import time as _time

    t0 = _time.perf_counter()
    shortlist = compare_cs_shortlist_on_graph(
        Gp,
        grid,
        mask,
        factors,
        bbox_key,
        ti,
        orig_xy,
        dest_xy,
        shadow_samples=SHADOW_SAMPLES_PER_EDGE,
    )
    t1 = _time.perf_counter()
    shadow = compare_shadow_samples(
        Gp, grid, mask, factors, bbox_key, ti, orig_xy, dest_xy
    )
    t2 = _time.perf_counter()

    _, _, prod_stats = route_shadow_aware_integrated(
        grid,
        when,
        hour,
        orig_xy,
        dest_xy,
        factors=factors,
        compare_accuracy=True,
    )

    return {
        "case_id": case["id"],
        "label": case["label"],
        "orig": {"lon": case["orig_lon"], "lat": case["orig_lat"]},
        "dest": {"lon": case["dest_lon"], "lat": case["dest_lat"]},
        "route_date": when.isoformat(),
        "hour": hour,
        "cs_shortlist_on_same_graph": shortlist.get("shortlist_8_vs_57"),
        "selection_directed_vs_undirected": shortlist.get(
            "directed_vs_undirected_selection_8"
        ),
        "shadow_samples_3_vs_8": shadow,
        "api_compare_accuracy": prod_stats.get("accuracy_vs_reference"),
        "benchmark_timing_s": {
            "cs_shortlist_s": round(t1 - t0, 3),
            "shadow_compare_s": round(t2 - t1, 3),
            "full_route_compare_s": round(
                prod_stats.get("timing_s", {}).get("accuracy_compare_s", 0), 3
            ),
        },
    }


def _aggregate(cases: list[dict]) -> dict:
    def _collect(key_path: str):
        vals = []
        same = 0
        n = 0
        for c in cases:
            node = c
            for k in key_path.split("."):
                node = (node or {}).get(k) if node else None
            if not node:
                continue
            n += 1
            if node.get("same_cs"):
                same += 1
            p = (node.get("path_cost_total") or {}).get("delta_pct")
            if p is not None:
                vals.append(abs(p))
        return {
            "cases_with_data": n,
            "same_cs_rate": round(same / n, 4) if n else None,
            "path_cost_total_abs_delta_pct": {
                "median": round(float(np_median(vals)), 4) if vals else None,
                "max": round(float(max(vals)), 4) if vals else None,
            },
        }

    import statistics

    def np_median(vals):
        return statistics.median(vals)

    return {
        "cs_shortlist_8_vs_57": _collect("cs_shortlist_on_same_graph"),
        "shadow_samples_3_vs_8": _collect("shadow_samples_3_vs_8"),
        "api_compare_accuracy": _collect("api_compare_accuracy"),
    }


def run_full_benchmark(
    *,
    when: date | None = None,
    hour: int = 14,
    year: int = 2026,
    cases: list[dict] | None = None,
) -> dict:
    when = when or date(2026, 8, 5)
    cases = cases or BENCHMARK_CASES
    grid = load_shadow_grid(REPO / "data", year)
    factors = RouteFactorsIntegrated(use_shadow=True, prefer_shade=True)

    results = [
        run_benchmark_case(grid, when, hour, case, factors) for case in cases
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_date": when.isoformat(),
        "hour": hour,
        "profiles": {
            "production": {
                "cs_candidates": AUTO_CS_CANDIDATES,
                "shadow_samples_per_edge": SHADOW_SAMPLES_PER_EDGE,
                "cs_selection": "directed",
            },
            "reference": {
                "cs_candidates": REFERENCE_CS_CANDIDATES,
                "shadow_samples_per_edge": REFERENCE_SHADOW_SAMPLES_PER_EDGE,
                "cs_selection": "directed",
            },
        },
        "cases": results,
        "aggregate": _aggregate(results),
    }
    return payload


def write_benchmark_artifacts(payload: dict | None = None) -> Path:
    payload = payload or run_full_benchmark()
    BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ACCURACY_MD.write_text(render_accuracy_markdown(payload), encoding="utf-8")
    return BENCHMARK_JSON


def render_accuracy_markdown(payload: dict) -> str:
    agg = payload.get("aggregate", {})
    prof = payload.get("profiles", {})
    lines = [
        "# ルート高速化と精度（記録）",
        "",
        f"- 記録日時（UTC）: `{payload.get('generated_at', '—')}`",
        f"- 対象日・時刻: `{payload.get('route_date')}` {payload.get('hour')} 時",
        f"- 生データ: [`data/route_accuracy_benchmark.json`](../data/route_accuracy_benchmark.json)",
        "",
        "## 比較プロファイル",
        "",
        "| 項目 | 本番（高速） | 基準（精度） |",
        "|------|-------------|-------------|",
        f"| CS 候補数 | {prof.get('production', {}).get('cs_candidates')} | {prof.get('reference', {}).get('cs_candidates')} |",
        f"| 日陰サンプル/辺 | {prof.get('production', {}).get('shadow_samples_per_edge')} | {prof.get('reference', {}).get('shadow_samples_per_edge')} |",
        f"| CS 選定 | {prof.get('production', {}).get('cs_selection')} | {prof.get('reference', {}).get('cs_selection')} |",
        "",
        "## 集計（ケース平均）",
        "",
    ]

    def _agg_row(title: str, key: str):
        a = agg.get(key, {})
        rate = a.get("same_cs_rate")
        med = (a.get("path_cost_total_abs_delta_pct") or {}).get("median")
        mx = (a.get("path_cost_total_abs_delta_pct") or {}).get("max")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- 同一 CS 率: **{rate * 100:.1f}%**" if rate is not None else "- 同一 CS 率: —")
        lines.append(
            f"- 総コスト差 |Δ%| 中央値: **{med}%**" if med is not None else "- 総コスト差: —"
        )
        lines.append(f"- 総コスト差 |Δ%| 最大: **{mx}%**" if mx is not None else "")
        lines.append("")

    _agg_row("CS 候補 8 vs 57（同一グラフ）", "cs_shortlist_8_vs_57")
    _agg_row("日陰サンプル 3 vs 8", "shadow_samples_3_vs_8")
    _agg_row("API compare_accuracy（本番ルート）", "api_compare_accuracy")

    lines.extend(
        [
            "## ケース別",
            "",
            "| ID | 同一CS | コストΔ% | 距離Δ% | 日陰率Δ |",
            "|----|--------|---------|--------|---------|",
        ]
    )
    for c in payload.get("cases", []):
        d = c.get("cs_shortlist_on_same_graph") or {}
        pc = (d.get("path_cost_total") or {}).get("delta_pct")
        pl = (d.get("path_length_m") or {}).get("delta_pct")
        sh = (d.get("mean_shadow_fraction") or {}).get("delta_abs")
        lines.append(
            f"| {c.get('case_id')} | {'✓' if d.get('same_cs') else '✗'} | "
            f"{pc if pc is not None else '—'} | {pl if pl is not None else '—'} | "
            f"{sh if sh is not None else '—'} |"
        )

    lines.extend(
        [
            "",
            "## 再計測",
            "",
            "```powershell",
            "py -3 scripts/benchmark_route_accuracy.py",
            "```",
            "",
            "API で1リクエストごとに差分を付ける: `POST /api/route_latlon` に `compare_accuracy: true`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    path = write_benchmark_artifacts()
    print(f"Wrote {path}")
