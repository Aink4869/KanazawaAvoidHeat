import pandas as pd
import json

df = pd.read_csv("cooling_shelters_geocoded_gsi.csv")

features = []

for _, r in df.iterrows():
    if pd.isna(r["lat"]) or pd.isna(r["lon"]):
        continue

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [r["lon"], r["lat"]]
        },
        "properties": r.drop(["lat", "lon"]).to_dict()
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("cooling_shelters.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"完了: cooling_shelters.geojson ({len(features)}件のフィーチャー)")

