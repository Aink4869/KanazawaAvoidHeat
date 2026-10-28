"""
ルート探索統合版（日陰・CS をコスト要素に含む。既存 nav_app / static とは別ポート）。

起動（リポジトリルートで）:
  py -3 -m uvicorn integrated.app:app --reload --host 127.0.0.1 --port 8766

ブラウザ:
  http://127.0.0.1:8766/          … 旧システム（軽量モデルの日陰のみ）
  http://127.0.0.1:8766/new       … 最新システム（樹木日陰を加算）
  http://127.0.0.1:8766/compare   … 旧と最新のルート比較
  http://127.0.0.1:8766/en        … English (redirects to /?lang=en)
  http://127.0.0.1:8766/?lang=en  … English UI
"""
from __future__ import annotations

import json
import math
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from pyproj import Transformer

from nav_app import (
    get_grid,
    grid_bounds_leaflet,
    mask_to_png_bytes,
    mask_to_png_bytes_hybrid,
    time_index,
    _shadow_masks_for_overlay,
    with_tree_flag,
    compare_route_delta,
)
from integrated.cs_spatial import _load_cs_geojson
from integrated.graph_cache import warmup_bbox
from integrated.cs_i18n import localize_geojson_feature, localize_visit_block
from integrated.route_ext import RouteFactorsIntegrated, route_shadow_aware_integrated

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
STATIC = Path(__file__).resolve().parent / "static"
CS_GEOJSON = REPO / "ジオコーディング" / "cooling_shelters.geojson"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    def _warm():
        try:
            warmup_bbox()
        except Exception:
            pass

    threading.Thread(target=_warm, name="osm-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="Kanazawa Integrated Route (shadow + cooling shelter factors)",
    version="1.2.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteFactorsIntegratedModel(BaseModel):
    use_distance: bool = True
    use_time: bool = True
    use_slope: bool = True
    use_shadow: bool = True
    include_tree_shade: bool = False
    use_cooling_shelter: bool = True
    weight_distance: float = Field(1.0, ge=0.0, le=100.0)
    weight_time: float = Field(1.0, ge=0.0, le=100.0)
    weight_slope: float = Field(3.0, ge=0.0, le=100.0)
    weight_shadow: float = Field(4.0, ge=0.0, le=100.0)
    weight_cooling_shelter: float = Field(2.5, ge=0.0, le=100.0)
    walk_speed_m_s: float = Field(1.2, ge=0.5, le=3.0)
    prefer_shade: bool = Field(
        True,
        description="True=日陰の多い道を選びやすい（コストに 1-日陰率）",
    )
    cs_dwell_time_s: float = Field(
        600.0,
        ge=0.0,
        le=7200.0,
        description="CS 立ち寄りの滞在時間 [s]（コスト = weight_cooling_shelter × 本値）",
    )

    @model_validator(mode="after")
    def integrated_rules(self):
        if not any(
            [self.use_distance, self.use_time, self.use_slope, self.use_shadow]
        ):
            raise ValueError(
                "歩行区間のコスト（距離・時間・勾配・日陰）のうち、少なくとも1つを有効にしてください。"
            )
        return self

    def to_route_factors_integrated(self) -> RouteFactorsIntegrated:
        return RouteFactorsIntegrated(
            use_distance=self.use_distance,
            use_time=self.use_time,
            use_slope=self.use_slope,
            use_shadow=self.use_shadow,
            include_tree_shade=self.include_tree_shade,
            use_cooling_shelter=self.use_cooling_shelter,
            weight_distance=self.weight_distance,
            weight_time=self.weight_time,
            weight_slope=self.weight_slope,
            weight_shadow=self.weight_shadow,
            weight_cooling_shelter=self.weight_cooling_shelter,
            walk_speed_m_s=self.walk_speed_m_s,
            prefer_shade=self.prefer_shade,
            cs_dwell_time_s=self.cs_dwell_time_s,
        )


class RouteLatLonIntegratedRequest(BaseModel):
    orig_lat: float
    orig_lon: float
    dest_lat: float
    dest_lon: float
    route_date: str
    hour: int = Field(14, ge=6, le=18)
    year: int = 2026
    visit_cs_id: int | None = Field(
        None,
        description="手動指定時のみ。未指定なら全 CS を評価して総コスト最小を自動選定",
    )
    compare_accuracy: bool = Field(
        False,
        description="true のとき本番 CS 選定と全57件基準の差分を stats.accuracy_vs_reference に付与",
    )
    lang: str = Field("ja", description="UI language for CS labels in response: ja | en")
    shadow_samples: int = Field(3, ge=1, le=16, description="辺あたりの日陰観測点。3 は従来の 0.2/0.5/0.8")
    shadow_cell_m: float = Field(12.0, description="12 または 2。2 は樹木影を 2m で引き直し、建物影は 12m のまま載せる")
    factors: RouteFactorsIntegratedModel = Field(default_factory=RouteFactorsIntegratedModel)


def _clean_json_nan(obj):
    if isinstance(obj, dict):
        return {k: _clean_json_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _load_cooling_shelters_geojson() -> dict:
    if not CS_GEOJSON.is_file():
        raise FileNotFoundError(f"GeoJSON がありません: {CS_GEOJSON}")
    return _clean_json_nan(_load_cs_geojson())


@app.get("/")
def integrated_ui():
    p = STATIC / "index.html"
    if not p.is_file():
        raise HTTPException(404, "integrated/static/index.html がありません")
    return FileResponse(p)


@app.get("/new")
def integrated_ui_new():
    p = STATIC / "index.html"
    if not p.is_file():
        raise HTTPException(404, "integrated/static/index.html がありません")
    return FileResponse(p)


@app.get("/compare")
def integrated_ui_compare():
    p = STATIC / "compare.html"
    if not p.is_file():
        raise HTTPException(404, "integrated/static/compare.html がありません")
    return FileResponse(p)


@app.get("/navi")
@app.get("/navi/new")
def integrated_ui_navi(request: Request):
    # 以前の HTML がブラウザに残ると、シートの見た目が更新されない。
    if request.query_params.get("_sheet") not in ("1", "2"):
        pairs = [(k, v) for k, v in request.query_params.multi_items() if k != "_sheet"]
        pairs.append(("_sheet", "1"))
        return RedirectResponse(url=f"{request.url.path}?{urlencode(pairs)}", status_code=307)
    p = STATIC / "navi.html"
    if not p.is_file():
        raise HTTPException(404, "integrated/static/navi.html がありません")
    return FileResponse(p, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})


@app.get("/en")
def integrated_ui_en():
    return RedirectResponse(url="/?lang=en", status_code=307)


@app.get("/route/en")
def integrated_ui_en_alias():
    return RedirectResponse(url="/?lang=en", status_code=307)


@app.get("/api/health")
def api_health():
    cs_ok = CS_GEOJSON.is_file()
    geom_npz = DATA / "summer_shadow_geometry_2026.npz"
    flat_npz = DATA / "summer_shadow_2026.npz"
    return {
        "ok": True,
        "service": "integrated_app",
        "port_hint": 8766,
        "cooling_shelters_geojson": cs_ok,
        "summer_shadow_npz": flat_npz.is_file(),
        "summer_shadow_geometry_npz": geom_npz.is_file(),
        "default_shadow_model": "geometry（旧） / geometry + tree residual（最新）",
        "features": [
            "osm_route_multi_factor",
            "cooling_shelter_visit_waypoint",
            "prefer_shade_routing",
            "shadow_overlay_api_optional",
            "geometry_dem_shadow_routing",
            "tree_residual_from_hp_model",
        ],
        "route_ui": {
            "old": "http://127.0.0.1:8766/",
            "new": "http://127.0.0.1:8766/new",
            "compare": "http://127.0.0.1:8766/compare",
            "navi": "http://127.0.0.1:8766/navi",
            "navi_new": "http://127.0.0.1:8766/navi?sys=new",
        },
        "ui_en": "http://127.0.0.1:8766/?lang=en",
        "legacy_nav_app": "http://127.0.0.1:8765/ (別プロセス)",
    }


@app.get("/api/meta")
def api_meta(year: int = Query(2026), shadow_model: str | None = Query(None, alias="model")):
    try:
        g = get_grid(year, shadow_model)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    from nav_app import _shadow_model_note

    return {
        "year": g.year,
        "shadow_model": g.model,
        "date_start": g.date_start,
        "date_end": g.date_end,
        "hours": g.hours,
        "bounds": {"xmin": g.xmin, "xmax": g.xmax, "ymin": g.ymin, "ymax": g.ymax},
        "grid": {"nx": g.nx, "ny": g.ny},
        "n_timesteps": int(g.masks.shape[0]),
        "elev_grid_available": g.elev_m is not None,
        "approximation_note": _shadow_model_note(g.model),
    }


@app.get("/api/shadow/query")
def api_shadow_query(
    route_date: str = Query(...),
    hour: int = Query(14, ge=6, le=18),
    year: int = Query(2026),
    include_tree_shade: bool = Query(False),
):
    try:
        g = get_grid(year)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    try:
        d = date.fromisoformat(route_date)
    except ValueError as e:
        raise HTTPException(400, "route_date は YYYY-MM-DD") from e
    try:
        combined, residual, tree_info, ti = _shadow_masks_for_overlay(g, d, hour, include_tree_shade)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    frac = float(np.mean(combined > 0)) if combined.size else 0.0
    return {
        "year": year,
        "route_date": route_date,
        "hour": hour,
        "timestep_index": int(ti),
        "bounds_latlon": grid_bounds_leaflet(g),
        "shadow_cell_fraction": frac,
        "overlay_png_path": "/api/shadow/overlay.png",
        "tree_shade": tree_info,
    }


@app.get("/api/shadow/overlay.png")
def api_shadow_overlay_png(
    route_date: str = Query(...),
    hour: int = Query(14, ge=6, le=18),
    year: int = Query(2026),
    include_tree_shade: bool = Query(False),
):
    try:
        g = get_grid(year)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    try:
        d = date.fromisoformat(route_date)
    except ValueError as e:
        raise HTTPException(400, "route_date は YYYY-MM-DD") from e
    try:
        combined, residual, tree_info, _ti = _shadow_masks_for_overlay(g, d, hour, include_tree_shade)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    png = (
        mask_to_png_bytes_hybrid((combined > 0) & (residual == 0), residual)
        if tree_info.get("tree_shade_applied")
        else mask_to_png_bytes(combined)
    )
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/cooling-shelters")
def api_cooling_shelters(lang: str = Query("ja", description="ja | en")):
    try:
        geo = _load_cooling_shelters_geojson()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    features = [
        localize_geojson_feature(f, lang) for f in geo.get("features", [])
    ]
    n = len(features)
    return JSONResponse(
        {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "count": n,
                "lang": lang,
                "source": str(CS_GEOJSON.relative_to(REPO)),
                "geocoder": "国土地理院 AddressSearch API (geocode_gsi.py)",
            },
        }
    )


@app.post("/api/route_latlon")
def api_route_latlon(req: RouteLatLonIntegratedRequest):
    try:
        g = get_grid(req.year)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    try:
        d = date.fromisoformat(req.route_date)
    except ValueError as e:
        raise HTTPException(400, "route_date は YYYY-MM-DD") from e
    tf = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    ox0, oy0 = tf.transform(req.orig_lon, req.orig_lat)
    dx0, dy0 = tf.transform(req.dest_lon, req.dest_lat)
    try:
        factors = req.factors.to_route_factors_integrated()
        xy, ll, stats = route_shadow_aware_integrated(
            g,
            d,
            req.hour,
            (ox0, oy0),
            (dx0, dy0),
            factors=factors,
            visit_cs_id=req.visit_cs_id,
            compare_accuracy=req.compare_accuracy,
            data_dir=DATA,
            shadow_samples=req.shadow_samples,
            shadow_cell_m=req.shadow_cell_m,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    lang = (req.lang or "ja").lower()
    if lang == "en" and stats.get("cooling_shelter_visit"):
        stats = dict(stats)
        stats["cooling_shelter_visit"] = localize_visit_block(
            stats["cooling_shelter_visit"], lang
        )
    return {"coordinates_xy": xy, "coordinates_latlon": ll, "stats": stats}


@app.post("/api/route_latlon_compare")
def api_route_latlon_compare(req: RouteLatLonIntegratedRequest):
    """同一起終点・日時で旧（樹木なし）と最新（樹木あり）を比較。立ち寄りCSは旧側に揃える。"""
    old_req = req.model_copy(update={"factors": with_tree_flag(req.factors, False)})
    old = api_route_latlon(old_req)
    cs_id = req.visit_cs_id
    if cs_id is None:
        cv = (old.get("stats") or {}).get("cooling_shelter_visit") or {}
        if cv.get("cs_id") is not None:
            cs_id = int(cv["cs_id"])
    new_req = req.model_copy(
        update={"factors": with_tree_flag(req.factors, True), "visit_cs_id": cs_id}
    )
    new = api_route_latlon(new_req)
    delta = compare_route_delta(old, new)
    cv_old = (old.get("stats") or {}).get("cooling_shelter_visit") or {}
    cv_new = (new.get("stats") or {}).get("cooling_shelter_visit") or {}
    delta["cs_id_old"] = cv_old.get("cs_id")
    delta["cs_id_new"] = cv_new.get("cs_id")
    delta["cs_name_old"] = cv_old.get("cs_name")
    delta["cs_name_new"] = cv_new.get("cs_name")
    delta["cs_fixed_from_old"] = req.visit_cs_id is None and cs_id is not None
    return {"old": old, "new": new, "delta": delta}


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static_integrated")
