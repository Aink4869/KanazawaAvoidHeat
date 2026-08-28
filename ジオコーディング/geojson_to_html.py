import json

# GeoJSONファイルを読み込む
with open("cooling_shelters.geojson", "r", encoding="utf-8") as f:
    geojson = json.load(f)

# NaN値をnullに変換（JSONで正しく扱えるように）
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
html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>金沢市 クーリングシェルター（CS）</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link
  rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body, html {{ margin:0; padding:0; height:100%; }}
  #map {{ width:100%; height:100%; }}
</style>
</head>
<body>
<div id="map"></div>

<!-- ▼ CSVから変換したGeoJSONをここに埋め込む -->
<script id="CS_GEOJSON" type="application/json">
{geojson_str}
</script>

<script>
const map = L.map("map").setView([36.5613, 136.6562], 13);

L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
  attribution: "© OpenStreetMap contributors"
}}).addTo(map);

// GeoJSON読み込み
const geojson = JSON.parse(
  document.getElementById("CS_GEOJSON").textContent
);

L.geoJSON(geojson, {{
  pointToLayer: (feature, latlng) =>
    L.circleMarker(latlng, {{
      radius: 6,
      color: "#0066cc",
      weight: 2,
      fillOpacity: 0.8
    }}),
  onEachFeature: (feature, layer) => {{
    const p = feature.properties;
    layer.bindPopup(`
      <strong>${{p.cs_name || p.name || "-"}}</strong><br>
      ${{p.address || "-"}}<br>
      時間：${{p.open_time || p.open_hours || "-"}}<br>
      定員：${{p.capacity || "-"}}<br>
      備考：${{p.notes || "-"}}
    `);
  }}
}}).addTo(map);
</script>
</body>
</html>"""

# HTMLファイルに保存
with open("cooling_shelter.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"完了: cooling_shelter.html ({len(geojson_cleaned['features'])}件のフィーチャー)")

