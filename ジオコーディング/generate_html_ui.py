import json

# GeoJSONファイルを読み込む
with open("cooling_shelters.geojson", "r", encoding="utf-8") as f:
    geojson = json.load(f)

# NaN値をnullに変換
def clean_geojson(obj):
    if isinstance(obj, dict):
        return {k: clean_geojson(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_geojson(item) for item in obj]
    elif isinstance(obj, float) and str(obj) == "nan":
        return None
    return obj

geojson_cleaned = clean_geojson(geojson)

# GeoJSONをJSON文字列に変換（インデント付き）
geojson_str = json.dumps(geojson_cleaned, ensure_ascii=False, indent=2)

# HTMLテンプレート
html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>金沢クーリングシェルター誘導システム</title>
<link
  rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{
  --bg:#0b1020;
  --fg:#f2f5ff;
  --muted:#9aa3b2;
  --panel:#0c132a;
  --border:#1f2a4d;
  --accent:#2ea0ff;
  --cs:#2ea0ff;
}}

*{{box-sizing:border-box}}

html,body{{
  height:100%;
  margin:0;
  background:var(--bg);
  color:var(--fg);
  font-family:system-ui,-apple-system,Segoe UI,Roboto,"Noto Sans JP",sans-serif;
}}

/* ===== Header ===== */
.appbar{{
  position:sticky;
  top:0;
  z-index:5;
  display:flex;
  justify-content:space-between;
  gap:12px;
  align-items:center;
  padding:10px 12px;
  border-bottom:1px solid var(--border);
  background:linear-gradient(90deg,#0b1020,#0d1634);
}}

.brand h1{{
  margin:0;
  font-size:18px;
}}

.brand .sub{{
  font-size:12px;
  color:var(--muted);
}}

.controls{{
  display:flex;
  gap:8px;
  align-items:center;
  flex-wrap:wrap;
}}

.input{{
  background:#0b1430;
  color:var(--fg);
  border:1px solid var(--border);
  border-radius:10px;
  padding:8px 10px;
  min-width:220px;
}}

.chk{{
  display:flex;
  gap:6px;
  align-items:center;
  font-size:14px;
}}

.btn{{
  background:var(--accent);
  border:1px solid #1b5fa8;
  color:#061222;
  font-weight:600;
  border-radius:10px;
  padding:8px 12px;
  cursor:pointer;
  border:none;
}}

.btn.ghost{{
  background:#0b1430;
  color:#cfe3ff;
}}

.btn:hover{{
  opacity:0.9;
}}

/* ===== Layout ===== */
.main{{
  display:grid;
  grid-template-columns:360px 1fr;
  height:calc(100vh - 66px);
}}

.panel{{
  border-right:1px solid var(--border);
  background:var(--panel);
  padding:10px;
  overflow:auto;
}}

.panel h2{{
  margin:6px 0 8px;
  font-size:15px;
}}

.small{{
  font-size:12px;
  color:var(--muted);
}}

/* ===== List ===== */
.list{{
  display:flex;
  flex-direction:column;
  gap:8px;
  max-height:50vh;
  overflow:auto;
}}

.item{{
  border:1px solid var(--border);
  border-radius:10px;
  background:#0e1736;
  padding:8px;
  cursor:pointer;
  transition:all 0.2s;
}}

.item:hover{{
  background:#121a3e;
  border-color:var(--accent);
}}

.item.active{{
  background:#0d1a3a;
  border-color:var(--accent);
}}

.item .name{{
  font-weight:700;
}}

.item .addr{{
  font-size:12px;
  color:var(--muted);
  margin-top:2px;
}}

.badge{{
  padding:2px 6px;
  border-radius:999px;
  font-size:12px;
  border:1px solid var(--border);
  background:#0b1430;
  display:inline-block;
  margin-top:4px;
}}

/* ===== Route panel ===== */
.route-panel{{
  margin-top:10px;
  border:1px solid var(--border);
  border-radius:10px;
  background:#0e1736;
  padding:8px;
  display:none;
}}

.route-badges{{
  display:flex;
  gap:6px;
  flex-wrap:wrap;
  margin-top:6px;
}}

.route-badge{{
  background:#0b1430;
  border:1px solid #1b5fa8;
  border-radius:999px;
  padding:4px 8px;
  font-size:12px;
}}

/* ===== Map ===== */
.mapwrap{{
  position:relative;
}}

#map{{
  position:absolute;
  inset:0;
  background:#081020;
}}

/* ===== Footer ===== */
.footer{{
  border-top:1px solid var(--border);
  padding:8px 12px;
  color:var(--muted);
  font-size:12px;
  text-align:center;
}}

/* ===== Responsive ===== */
@media (max-width:900px){{
  .main{{
    grid-template-columns:1fr;
  }}
  .panel{{
    height:42vh;
    order:2;
  }}
  .mapwrap{{
    height:58vh;
  }}
}}
</style>
</head>
<body>
<header class="appbar">
  <div class="brand">
    <h1>金沢クーリングシェルター誘導システム</h1>
    <div class="sub">CSマップ（{len(geojson_cleaned['features'])}件）</div>
  </div>
  <div class="controls">
    <input class="input" type="text" id="searchInput" placeholder="名称・住所で検索">
    <label class="chk"><input type="checkbox" id="csFilter" checked> CS</label>
    <button class="btn" id="nearestBtn">最寄り</button>
  </div>
</header>

<main class="main">
  <aside class="panel">
    <h2>検索結果</h2>
    <div class="list" id="listContainer"></div>
    <div class="route-panel" id="routePanel">
      <strong>ルート情報</strong>
      <div class="route-badges">
        <span class="route-badge" id="distanceBadge">距離 -</span>
        <span class="route-badge" id="timeBadge">時間 -</span>
      </div>
    </div>
    <div class="small" style="margin-top:10px;">
      ※ マーカーをクリックすると詳細情報が表示されます<br>
      ※ リスト項目をクリックすると地図上で該当施設に移動します
    </div>
  </aside>

  <section class="mapwrap">
    <div id="map"></div>
  </section>
</main>

<footer class="footer">
  UI Template for Cooling Shelter Map
</footer>

<!-- GeoJSONデータ -->
<script id="CS_GEOJSON" type="application/json">
{geojson_str}
</script>

<script>
// データ読み込み
const geojson = JSON.parse(document.getElementById("CS_GEOJSON").textContent);
let map, markers = [], layers = [];
let currentLocation = null;
let currentLocationMarker = null;
let selectedFeature = null;

// 地図初期化
map = L.map("map").setView([36.5613, 136.6562], 13);

// ダークテーマのタイルレイヤー（オプション）
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
  attribution: "© OpenStreetMap contributors"
}}).addTo(map);

// マーカー作成
function createMarker(feature) {{
  const [lon, lat] = feature.geometry.coordinates;
  const marker = L.circleMarker([lat, lon], {{
    radius: 6,
    color: "#2ea0ff",
    weight: 2,
    fillOpacity: 0.8
  }});

  const p = feature.properties;
  marker.bindPopup(`
    <strong>${{p.cs_name || "-"}}</strong><br>
    ${{p.address || "-"}}<br>
    時間：${{p.open_time || "-"}}<br>
    定員：${{p.capacity || "-"}}<br>
    備考：${{p.notes || "-"}}
  `);

  marker.on("click", () => {{
    highlightFeature(feature);
    scrollToItem(feature);
  }});

  return marker;
}}

// フィーチャーをハイライト
function highlightFeature(feature) {{
  // 既存のハイライトを解除
  document.querySelectorAll(".item").forEach(item => {{
    item.classList.remove("active");
  }});

  // リスト項目をハイライト
  const item = document.querySelector(`[data-id="${{feature.properties.cs_id}}"]`);
  if (item) {{
    item.classList.add("active");
    item.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
  }}

  selectedFeature = feature;
}}

// リスト項目までスクロール
function scrollToItem(feature) {{
  const item = document.querySelector(`[data-id="${{feature.properties.cs_id}}"]`);
  if (item) {{
    item.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
  }}
}}

// リスト表示
function renderList(features) {{
  const container = document.getElementById("listContainer");
  container.innerHTML = "";

  features.forEach(feature => {{
    const p = feature.properties;
    const item = document.createElement("div");
    item.className = "item";
    item.setAttribute("data-id", p.cs_id);
    item.innerHTML = `
      <div class="name">${{p.cs_name || "-"}}</div>
      <div class="addr">${{p.address || "-"}}</div>
      <span class="badge">CS</span>
    `;

    item.addEventListener("click", () => {{
      const [lon, lat] = feature.geometry.coordinates;
      map.setView([lat, lon], 15);
      highlightFeature(feature);
      
      // マーカーのポップアップを開く
      layers.forEach(layer => {{
        if (layer.feature && layer.feature.properties.cs_id === p.cs_id) {{
          layer.openPopup();
        }}
      }});
    }});

    container.appendChild(item);
  }});
}}

// 検索機能
function filterFeatures(query) {{
  if (!query) {{
    return geojson.features;
  }}

  const lowerQuery = query.toLowerCase();
  return geojson.features.filter(feature => {{
    const p = feature.properties;
    return (p.cs_name && p.cs_name.toLowerCase().includes(lowerQuery)) ||
           (p.address && p.address.toLowerCase().includes(lowerQuery));
  }});
}}

// 検索入力
document.getElementById("searchInput").addEventListener("input", (e) => {{
  const query = e.target.value;
  const filtered = filterFeatures(query);
  renderList(filtered);
  updateMarkers(filtered);
}});

// マーカー更新
function updateMarkers(features) {{
  // 既存のマーカーを削除
  layers.forEach(layer => map.removeLayer(layer));
  layers = [];

  // 新しいマーカーを追加
  features.forEach(feature => {{
    const layer = createMarker(feature);
    layer.feature = feature;
    layer.addTo(map);
    layers.push(layer);
  }});
}}

// 最寄り検索
document.getElementById("nearestBtn").addEventListener("click", () => {{
  if (!navigator.geolocation) {{
    alert("位置情報サービスが利用できません");
    return;
  }}

  navigator.geolocation.getCurrentPosition(
    (position) => {{
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      currentLocation = {{ lat, lon }};

      // 現在地マーカー
      if (currentLocationMarker) {{
        map.removeLayer(currentLocationMarker);
      }}
      currentLocationMarker = L.marker([lat, lon], {{
        icon: L.divIcon({{
          className: "current-location-marker",
          html: '<div style="width:20px;height:20px;border-radius:50%;background:#ff4444;border:3px solid white;box-shadow:0 0 10px rgba(0,0,0,0.5);"></div>',
          iconSize: [20, 20]
        }})
      }}).addTo(map);

      map.setView([lat, lon], 14);

      // 距離計算
      const featuresWithDistance = geojson.features.map(feature => {{
        const [fLon, fLat] = feature.geometry.coordinates;
        const distance = calculateDistance(lat, lon, fLat, fLon);
        return {{ feature, distance }};
      }});

      // 距離でソート
      featuresWithDistance.sort((a, b) => a.distance - b.distance);
      const nearest = featuresWithDistance[0];

      if (nearest) {{
        const [lon, lat] = nearest.feature.geometry.coordinates;
        map.setView([lat, lon], 15);
        highlightFeature(nearest.feature);
        
        // ルート情報表示
        document.getElementById("routePanel").style.display = "block";
        document.getElementById("distanceBadge").textContent = `距離 ${{nearest.distance.toFixed(1)}}km`;
        document.getElementById("timeBadge").textContent = `時間 約${{Math.round(nearest.distance * 15)}}分（徒歩）`;
      }}
    }},
    (error) => {{
      alert("位置情報の取得に失敗しました");
    }}
  );
}});

// 距離計算（ハーバーサイン公式）
function calculateDistance(lat1, lon1, lat2, lon2) {{
  const R = 6371; // 地球の半径（km）
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}}

// 初期表示
renderList(geojson.features);
updateMarkers(geojson.features);
</script>
</body>
</html>"""

# HTMLファイルに保存
with open("cooling_shelter.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"完了: cooling_shelter.html ({len(geojson_cleaned['features'])}件のフィーチャー)")

