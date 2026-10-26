"""経路ノード列からターンバイターン案内（曲がる方向・道路名・距離・日陰率）を組み立てる。"""
from __future__ import annotations

import math
from typing import Any

import networkx as nx

TURN_STRAIGHT_DEG = 25.0
TURN_SLIGHT_DEG = 55.0
TURN_SHARP_DEG = 135.0
MIN_STEP_M = 12.0

_HIGHWAY_JA = {
    "footway": "歩道",
    "pedestrian": "歩行者道",
    "path": "小道",
    "steps": "階段",
    "service": "路地",
    "living_street": "生活道路",
    "residential": "住宅街の道",
    "unclassified": "一般道",
    "tertiary": "市道",
    "secondary": "県道",
    "primary": "幹線道路",
    "trunk": "国道",
    "cycleway": "自転車道",
    "track": "農道",
}
_HIGHWAY_EN = {
    "footway": "footpath",
    "pedestrian": "pedestrian street",
    "path": "path",
    "steps": "stairs",
    "service": "service road",
    "living_street": "living street",
    "residential": "residential street",
    "unclassified": "local road",
    "tertiary": "city road",
    "secondary": "prefectural road",
    "primary": "main road",
    "trunk": "national road",
    "cycleway": "cycleway",
    "track": "track",
}

_MODIFIER_JA = {
    "straight": "直進",
    "slight right": "やや右",
    "right": "右折",
    "sharp right": "大きく右",
    "slight left": "やや左",
    "left": "左折",
    "sharp left": "大きく左",
    "uturn": "Uターン",
}
_MODIFIER_EN = {
    "straight": "Continue straight",
    "slight right": "Bear right",
    "right": "Turn right",
    "sharp right": "Turn sharp right",
    "slight left": "Bear left",
    "left": "Turn left",
    "sharp left": "Turn sharp left",
    "uturn": "Make a U-turn",
}


def _first(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v


def _clean_name(v: Any) -> str | None:
    v = _first(v)
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _best_edge(G: nx.MultiDiGraph, u, v) -> dict:
    datas = G.get_edge_data(u, v) or G.get_edge_data(v, u) or {}
    if not datas:
        return {}
    return min(datas.values(), key=lambda d: float(d.get("nav_cost", d.get("length", 1e18))))


def _bearing(x0: float, y0: float, x1: float, y1: float) -> float:
    """投影座標（y=北）での方位角 [deg]、北=0・時計回り。"""
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def _edge_bearings(G: nx.MultiDiGraph, u, v, data: dict) -> tuple[float, float]:
    """辺の出発方位と到着方位（曲がった道でも交差点付近の向きで判定）。"""
    nu, nv = G.nodes[u], G.nodes[v]
    geom = data.get("geometry")
    if geom is not None:
        pts = list(geom.coords)
        if len(pts) >= 2:
            d_start = math.hypot(pts[0][0] - nu["x"], pts[0][1] - nu["y"])
            d_end = math.hypot(pts[-1][0] - nu["x"], pts[-1][1] - nu["y"])
            if d_end < d_start:
                pts = pts[::-1]
            b_out = _bearing(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
            b_in = _bearing(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1])
            return b_out, b_in
    b = _bearing(nu["x"], nu["y"], nv["x"], nv["y"])
    return b, b


def _modifier(delta: float) -> str:
    a = abs(delta)
    if a < TURN_STRAIGHT_DEG:
        return "straight"
    side = "right" if delta > 0 else "left"
    if a < TURN_SLIGHT_DEG:
        return f"slight {side}"
    if a < TURN_SHARP_DEG:
        return side
    if a < 170.0:
        return f"sharp {side}"
    return "uturn"


def _road_label(name: str | None, highway: str | None, lang: str) -> str:
    if name:
        return name
    table = _HIGHWAY_EN if lang == "en" else _HIGHWAY_JA
    if highway and highway in table:
        return table[highway]
    return "road" if lang == "en" else "道路"


def _instruction(kind: str, modifier: str, road: str, lang: str, cs_name: str | None = None) -> str:
    if lang == "en":
        if kind == "depart":
            return f"Head out on {road}"
        if kind == "arrive":
            return "Arrive at your destination"
        if kind == "waypoint":
            return f"Stop at cooling shelter: {cs_name or ''}".strip()
        if kind == "continue":
            return f"Continue onto {road}"
        return f"{_MODIFIER_EN.get(modifier, 'Continue')} onto {road}"
    if kind == "depart":
        return f"{road}を進む"
    if kind == "arrive":
        return "目的地に到着"
    if kind == "waypoint":
        return f"クーリングシェルター「{cs_name or ''}」に立ち寄り"
    if kind == "continue":
        return f"道なりに{road}へ"
    return f"{_MODIFIER_JA.get(modifier, '直進')}して{road}へ"


def build_nav_steps(
    G: nx.MultiDiGraph,
    path: list,
    *,
    waypoint_node: Any = None,
    waypoint_name: str | None = None,
    lang: str = "ja",
) -> list[dict]:
    """
    path（ノード列）→ 案内ステップ列。

    各ステップ: type / modifier / road / instruction / distance_m（このステップで歩く距離）/
    shade_frac / start_index（coordinates_latlon 上の開始点）/ bearing_after
    """
    if len(path) < 2:
        return []

    edges = []
    for i, (u, v) in enumerate(zip(path[:-1], path[1:])):
        d = _best_edge(G, u, v)
        b_out, b_in = _edge_bearings(G, u, v, d)
        edges.append(
            {
                "i": i,
                "name": _clean_name(d.get("name")),
                "highway": _clean_name(d.get("highway")),
                "length": float(d.get("length", 0.0) or 0.0),
                "shade": float(d.get("shadow_fraction", 0.0) or 0.0),
                "b_out": b_out,
                "b_in": b_in,
            }
        )

    wp_index = None
    if waypoint_node is not None:
        for k, n in enumerate(path):
            if n == waypoint_node and 0 < k < len(path) - 1:
                wp_index = k
                break

    steps: list[dict] = []

    def _open(kind: str, modifier: str, e: dict) -> dict:
        return {
            "type": kind,
            "modifier": modifier,
            "name": e["name"],
            "highway": e["highway"],
            "start_index": e["i"],
            "bearing_after": round(e["b_out"], 1),
            "_len": 0.0,
            "_shade_len": 0.0,
        }

    cur = _open("depart", "straight", edges[0])
    for k, e in enumerate(edges):
        if k > 0:
            prev = edges[k - 1]
            delta = ((e["b_out"] - prev["b_in"] + 540.0) % 360.0) - 180.0
            mod = _modifier(delta)
            name_changed = (e["name"] or e["highway"]) != (cur["name"] or cur["highway"])
            at_waypoint = wp_index is not None and e["i"] == wp_index
            if at_waypoint:
                steps.append(cur)
                wp = _open("waypoint", "straight", e)
                wp["cs_name"] = waypoint_name
                steps.append(wp)
                cur = _open("depart" if mod == "straight" else "turn", mod, e)
            elif mod != "straight" and (name_changed or abs(delta) >= TURN_SLIGHT_DEG):
                steps.append(cur)
                cur = _open("turn", mod, e)
            elif name_changed and e["name"] and cur["name"]:
                steps.append(cur)
                cur = _open("continue", "straight", e)
        cur["_len"] += e["length"]
        cur["_shade_len"] += e["length"] * e["shade"]
    steps.append(cur)

    merged: list[dict] = []
    for s in steps:
        if (
            merged
            and s["type"] in ("turn", "continue")
            and s["_len"] < MIN_STEP_M
            and merged[-1]["type"] != "waypoint"
        ):
            merged[-1]["_len"] += s["_len"]
            merged[-1]["_shade_len"] += s["_shade_len"]
            continue
        merged.append(s)

    last = edges[-1]
    merged.append(
        {
            "type": "arrive",
            "modifier": "straight",
            "name": None,
            "highway": None,
            "start_index": len(path) - 1,
            "bearing_after": round(last["b_in"], 1),
            "_len": 0.0,
            "_shade_len": 0.0,
        }
    )

    out = []
    for s in merged:
        road = _road_label(s["name"], s["highway"], lang)
        L = s.pop("_len")
        SL = s.pop("_shade_len")
        s["road"] = road
        s["distance_m"] = round(L, 1)
        s["shade_frac"] = round(SL / L, 3) if L > 0 else 0.0
        s["instruction"] = _instruction(s["type"], s["modifier"], road, lang, s.get("cs_name"))
        out.append(s)
    return out
