"""
同一データ（summer_shadow_*.npz / summer_shadow_geometry_*.npz）で日陰オーバーレイ表示とルート提案を提供する FastAPI。

既定は幾何+DEM モデル（SHADOW_MODEL=geometry）。平坦地近似に戻す場合は SHADOW_MODEL=flat。

起動例:
  py -3 -m uvicorn nav_app:app --reload --host 127.0.0.1 --port 8765

- http://127.0.0.1:8765/ … 日陰グリッド表示（解析）
- http://127.0.0.1:8765/route … 旧システム（軽量モデルの日陰のみ）
- http://127.0.0.1:8765/route/new … 最新システム（樹木日陰を加算）
- http://127.0.0.1:8765/route/compare … 旧と最新のルート比較
- http://127.0.0.1:8765/route/en … English route UI (API connected)
- static/index_en.html … open locally; API defaults to http://127.0.0.1:8765
"""
from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from pyproj import Transformer
from PIL import Image

from nav_shadow import (
    RouteFactors,
    SHADOW_MODEL_FLAT,
    SHADOW_MODEL_GEOMETRY,
    SHADOW_MODEL_GEOMETRY_TREES,
    ShadowGrid,
    load_shadow_grid,
    resolve_shadow_model,
    route_shadow_aware,
    time_index,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATIC = ROOT / "static"
CS_GEOJSON = ROOT / "ジオコーディング" / "cooling_shelters.geojson"

app = FastAPI(title="Kanazawa Shadow Map & Navigator", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# StaticFiles はルート定義の後で mount する（末尾の if __name__ ブロック相当の位置）


class RouteFactorsModel(BaseModel):
    """ルートコストの要素。チェックを外した項はコストに入りません。"""

    use_distance: bool = True
    use_time: bool = True
    use_slope: bool = True
    use_shadow: bool = True
    include_tree_shade: bool = False
    weight_distance: float = Field(1.0, ge=0.0, le=100.0)
    weight_time: float = Field(1.0, ge=0.0, le=100.0)
    weight_slope: float = Field(3.0, ge=0.0, le=100.0)
    # 負値=日陰を多く通るルートを優先（コストを割引）／正値=日陰を避け日当たり優先
    weight_shadow: float = Field(4.0, ge=-100.0, le=100.0)
    walk_speed_m_s: float = Field(1.2, ge=0.5, le=3.0)

    @model_validator(mode="after")
    def at_least_one(self):
        if not any(
            [
                self.use_distance,
                self.use_time,
                self.use_slope,
                self.use_shadow,
            ]
        ):
            raise ValueError("距離・時間・勾配・日陰のうち、少なくとも1つを有効にしてください")
        return self

    def to_route_factors(self) -> RouteFactors:
        return RouteFactors(
            use_distance=self.use_distance,
            use_time=self.use_time,
            use_slope=self.use_slope,
            use_shadow=self.use_shadow,
            include_tree_shade=self.include_tree_shade,
            weight_distance=self.weight_distance,
            weight_time=self.weight_time,
            weight_slope=self.weight_slope,
            weight_shadow=self.weight_shadow,
            walk_speed_m_s=self.walk_speed_m_s,
        )


class RouteRequest(BaseModel):
    """リクエストボディ（投影座標メートル）。"""

    orig_x: float = Field(..., description="起点 X [m] EPSG:6670")
    orig_y: float = Field(..., description="起点 Y [m]")
    dest_x: float = Field(..., description="終点 X [m]")
    dest_y: float = Field(..., description="終点 Y [m]")
    route_date: str = Field(..., description="YYYY-MM-DD")
    hour: int = Field(14, ge=6, le=18, description="ローカル時刻（時）6〜18")
    year: int = Field(2026, description="summer_shadow_{year}.npz の年")
    factors: RouteFactorsModel = Field(default_factory=RouteFactorsModel)


class RouteLatLonRequest(BaseModel):
    """緯度経度（WGS84）で指定。"""

    orig_lat: float
    orig_lon: float
    dest_lat: float
    dest_lon: float
    route_date: str
    hour: int = Field(14, ge=6, le=18)
    year: int = 2026
    factors: RouteFactorsModel = Field(default_factory=RouteFactorsModel)


_grid_cache = {}


def _shadow_model_note(model: str) -> str:
    notes = {
        SHADOW_MODEL_FLAT: "masks は compute_summer_shadows.py の平坦地・平行移動近似です。",
        SHADOW_MODEL_GEOMETRY: "masks は compute_geometry_shadow_npz.py の幾何+DEM（壁面影）です。",
        SHADOW_MODEL_GEOMETRY_TREES: "masks は compute_geometry_shadow_npz.py --with-trees の幾何+DEM+樹木です。",
        "geometry+tree_residual": "軽量モデルの日陰に、高精度（樹木あり）だけが持つ日陰を加算しています。",
    }
    return notes.get(model, f"shadow model: {model}")


def get_grid(year: int, model: str | None = None):
    resolved = resolve_shadow_model(DATA, year, model)
    key = (year, resolved)
    if key not in _grid_cache:
        _grid_cache[key] = load_shadow_grid(DATA, year, resolved)
    return _grid_cache[key]


def grid_bounds_leaflet(g: ShadowGrid) -> list[list[float]]:
    """Leaflet imageOverlay 用 [[south, west], [north, east]]（WGS84）。"""
    tf = Transformer.from_crs("EPSG:6670", "EPSG:4326", always_xy=True)
    corners = [
        (g.xmin, g.ymin),
        (g.xmin, g.ymax),
        (g.xmax, g.ymin),
        (g.xmax, g.ymax),
    ]
    lats: list[float] = []
    lons: list[float] = []
    for x, y in corners:
        lon, lat = tf.transform(float(x), float(y))
        lats.append(float(lat))
        lons.append(float(lon))
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    return [[south, west], [north, east]]


# 日陰マップ用: グリッドセル単位の簡易2色区分（ルート用マスクの0/1をそのまま色塗り）
_SHADOW_MAP_RGBA = np.array([45, 58, 120, 255], dtype=np.uint8)  # 日陰（軽量）
_TREE_RESIDUAL_RGBA = np.array([46, 125, 50, 255], dtype=np.uint8)  # 高精度樹木の追加日陰
_SUN_MAP_RGBA = np.array([255, 244, 214, 255], dtype=np.uint8)  # 非日陰（日当たり側・簡易）


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """
    masks の1スライス (ny, nx) uint8 を、セルごとに2色で塗り分けた PNG（RGBA）にする。
    建物影の幾何は粗いグリッド近似（compute_summer_shadows と同じデータ）の可視化。
    """
    ny, nx = mask.shape
    rgba = np.empty((ny, nx, 4), dtype=np.uint8)
    rgba[:] = _SUN_MAP_RGBA
    rgba[mask > 0] = _SHADOW_MAP_RGBA
    img = Image.fromarray(rgba, "RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def mask_to_png_bytes_hybrid(light: np.ndarray, residual: np.ndarray) -> bytes:
    """軽量日陰（青）＋樹木残差（緑）。"""
    ny, nx = light.shape
    rgba = np.empty((ny, nx, 4), dtype=np.uint8)
    rgba[:] = _SUN_MAP_RGBA
    rgba[light > 0] = _SHADOW_MAP_RGBA
    rgba[residual > 0] = _TREE_RESIDUAL_RGBA
    img = Image.fromarray(rgba, "RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _shadow_masks_for_overlay(g: ShadowGrid, d: date, hour: int, include_tree_shade: bool):
    from nav_tree_shade import apply_tree_residual

    ti = time_index(g, d, hour)
    light = g.masks[ti]
    combined, residual, info = apply_tree_residual(
        light, g, d, hour, include_tree_shade=include_tree_shade, root=ROOT
    )
    return combined, residual, info, ti


def _file_shadow_html():
    p = STATIC / "shadow.html"
    if not p.is_file():
        raise HTTPException(404, "static/shadow.html がありません")
    return FileResponse(p)


def _file_route_html():
    index_path = STATIC / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            {
                "message": "static/index.html がありません。",
                "api": "/api/route",
            }
        )
    return FileResponse(index_path)

def _file_route_html_en():
    index_path = STATIC / "index_en.html"
    if not index_path.is_file():
        raise HTTPException(404, "static/index_en.html not found")
    return FileResponse(index_path)


def _file_static_html(name: str):
    p = STATIC / name
    if not p.is_file():
        raise HTTPException(404, f"static/{name} がありません")
    return FileResponse(p)


def with_tree_flag(factors: BaseModel, include: bool):
    """経路コスト用 factors の樹木日陰フラグだけを差し替える。"""
    if hasattr(factors, "model_copy"):
        return factors.model_copy(update={"include_tree_shade": include})
    return factors.copy(update={"include_tree_shade": include})


def _stat_float(stats: dict, key: str) -> float:
    try:
        return float((stats or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _routes_identical(ll_a, ll_b, tol: float = 1e-6) -> bool:
    if not ll_a or not ll_b or len(ll_a) != len(ll_b):
        return False
    for a, b in zip(ll_a, ll_b):
        if abs(float(a[0]) - float(b[0])) > tol or abs(float(a[1]) - float(b[1])) > tol:
            return False
    return True


def compare_route_delta(old: dict, new: dict) -> dict:
    os_ = old.get("stats") or {}
    ns_ = new.get("stats") or {}
    lo = _stat_float(os_, "path_length_geometric_m")
    ln = _stat_float(ns_, "path_length_geometric_m")
    so = _stat_float(os_, "mean_shadow_fraction_on_path")
    sn = _stat_float(ns_, "mean_shadow_fraction_on_path")
    co = _stat_float(os_, "path_cost_total")
    cn = _stat_float(ns_, "path_cost_total")
    return {
        "length_m_old": lo,
        "length_m_new": ln,
        "length_m_delta": ln - lo,
        "shade_frac_old": so,
        "shade_frac_new": sn,
        "shade_frac_delta": sn - so,
        "cost_old": co,
        "cost_new": cn,
        "cost_delta": cn - co,
        "routes_identical": _routes_identical(
            old.get("coordinates_latlon"), new.get("coordinates_latlon")
        ),
        "tree_shade_old": bool((os_.get("factors") or {}).get("include_tree_shade")),
        "tree_shade_new": bool((ns_.get("factors") or {}).get("include_tree_shade")),
    }


@app.get("/")
def shadow_viewer():
    """日陰グリッド表示（ナビと同一の summer_shadow データ）。"""
    return _file_shadow_html()


@app.get("/shadow-map")
def shadow_viewer_alias():
    """日陰マップ（`/` と同一。別 URL で 404 になる環境向け）。"""
    return _file_shadow_html()



@app.get("/en")
def route_ui_en_short():
    return RedirectResponse(url="/route/en", status_code=307)


@app.get("/route/en")
def route_ui_en():
    """English route search UI."""
    return _file_route_html_en()


@app.get("/route")
def route_ui():
    """旧システム：軽量モデルの日陰のみ。"""
    return _file_route_html()


@app.get("/route/")
def route_ui_trailing_slash():
    return RedirectResponse(url="/route", status_code=307)


@app.get("/route/new")
def route_ui_new():
    """最新システム：樹木日陰（高精度の追加分）を加算。"""
    return _file_route_html()


@app.get("/route/compare")
def route_ui_compare():
    """旧システムと最新システムのルート比較。"""
    return _file_static_html("route_compare.html")


@app.get("/nav")
def route_ui_alias():
    """ルート検索（`/route` と同一。ブロックされる場合向け）。"""
    return _file_route_html()


@app.get("/nav/new")
def route_ui_new_alias():
    return _file_route_html()


@app.get("/nav/compare")
def route_ui_compare_alias():
    return _file_static_html("route_compare.html")


@app.get("/api/health")
def api_health():
    """フロントの接続確認用。起動していれば 200。"""
    return {
        "ok": True,
        "service": "nav_app",
        "shadow_ui": ["/", "/shadow-map"],
        "route_ui": {
            "old": ["/route", "/nav"],
            "new": ["/route/new", "/nav/new"],
            "compare": ["/route/compare", "/nav/compare"],
            "en": ["/route/en"],
        },
        "open_url": "http://127.0.0.1:8765/",
        "note": "このホストで nav_app を起動しているか確認してください。/route が 404 のときは /nav を試してください。HTML を file:// で開くと API に接続できません。",
        "cooling_shelters_geojson": CS_GEOJSON.is_file(),
    }


@app.get("/api/cooling-shelters")
def api_cooling_shelters():
    """地図表示用。ナビ（8765）の経路は立ち寄り必須ではない。立ち寄り付きは 8766。"""
    try:
        from integrated.cs_spatial import _load_cs_geojson
    except Exception as e:
        raise HTTPException(503, f"CS 読込モジュールを読み込めません: {e}") from e
    if not CS_GEOJSON.is_file():
        raise HTTPException(503, f"GeoJSON がありません: {CS_GEOJSON}")
    geo = _load_cs_geojson()
    features = geo.get("features", [])
    return JSONResponse(
        {
            "type": "FeatureCollection",
            "features": features,
            "meta": {"count": len(features), "lang": "ja", "source": str(CS_GEOJSON.relative_to(ROOT))},
        }
    )


@app.get("/api/meta")
def api_meta(year: int = Query(2026), shadow_model: str | None = Query(None, alias="model")):
    try:
        g = get_grid(year, shadow_model)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
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
    route_date: str = Query(..., description="YYYY-MM-DD"),
    hour: int = Query(14, ge=6, le=18),
    year: int = Query(2026),
    include_tree_shade: bool = Query(False),
):
    """
    指定日時のグリッドインデックス・表示境界・日陰セル割合。
    画像 URL は `/api/shadow/overlay.png` に同一クエリを付与。
    """
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
        "grid_nx": g.nx,
        "grid_ny": g.ny,
        "shadow_cell_fraction": frac,
        "shadow_model": g.model,
        "approximation_note": _shadow_model_note(g.model),
        "overlay_png_path": "/api/shadow/overlay.png",
        "tree_shade": tree_info,
        "legend": {
            "mode": "hybrid_cells" if tree_info.get("tree_shade_applied") else "two_color_cells",
            "shadow_label": "日陰（軽量モデル）",
            "tree_residual_label": "高精度樹木の追加日陰",
            "sun_label": "非日陰",
            "shadow_rgba": _SHADOW_MAP_RGBA.tolist(),
            "tree_residual_rgba": _TREE_RESIDUAL_RGBA.tolist(),
            "sun_rgba": _SUN_MAP_RGBA.tolist(),
        },
    }


@app.get("/api/shadow/overlay.png")
def api_shadow_overlay_png(
    route_date: str = Query(..., description="YYYY-MM-DD"),
    hour: int = Query(14, ge=6, le=18),
    year: int = Query(2026),
    include_tree_shade: bool = Query(False),
):
    """日陰マスク1枚を PNG（RGBA）で返す。Leaflet ImageOverlay 向け。"""
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
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/route")
def api_route(req: RouteRequest):
    try:
        g = get_grid(req.year)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    try:
        d = date.fromisoformat(req.route_date)
    except ValueError as e:
        raise HTTPException(400, "route_date は YYYY-MM-DD") from e
    try:
        factors = req.factors.to_route_factors()
        xy, ll, stats = route_shadow_aware(
            g,
            d,
            req.hour,
            (req.orig_x, req.orig_y),
            (req.dest_x, req.dest_y),
            factors=factors,
            data_dir=DATA,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    return {"coordinates_xy": xy, "coordinates_latlon": ll, "stats": {**stats, "shadow_model": g.model}}


@app.post("/api/route_latlon")
def api_route_latlon(req: RouteLatLonRequest):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)
    ox0, oy0 = tf.transform(req.orig_lon, req.orig_lat)
    dx0, dy0 = tf.transform(req.dest_lon, req.dest_lat)
    body = RouteRequest(
        orig_x=ox0,
        orig_y=oy0,
        dest_x=dx0,
        dest_y=dy0,
        route_date=req.route_date,
        hour=req.hour,
        year=req.year,
        factors=req.factors,
    )
    return api_route(body)


@app.post("/api/route_latlon_compare")
def api_route_latlon_compare(req: RouteLatLonRequest):
    """同一起終点・日時で、旧（樹木なし）と最新（樹木あり）のルートを返す。"""
    old_req = req.model_copy(update={"factors": with_tree_flag(req.factors, False)})
    new_req = req.model_copy(update={"factors": with_tree_flag(req.factors, True)})
    old = api_route_latlon(old_req)
    new = api_route_latlon(new_req)
    return {"old": old, "new": new, "delta": compare_route_delta(old, new)}


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
