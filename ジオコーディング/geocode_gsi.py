import pandas as pd
import requests
import time

# 入力CSV
INPUT_CSV = "cooling_shelters.csv"

# 出力CSV
OUTPUT_CSV = "cooling_shelters_geocoded_gsi.csv"

df = pd.read_csv(INPUT_CSV)

lats = []
lons = []

total = len(df)
print(f"総件数: {total}件")

for idx, addr in enumerate(df["address"], 1):
    if pd.isna(addr):
        lats.append(None)
        lons.append(None)
        print(f"[{idx}/{total}] スキップ: 住所が空です")
        continue

    url = "https://msearch.gsi.go.jp/address-search/AddressSearch"
    params = {"q": addr}

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if len(data) > 0:
            # geometry.coordinates = [lon, lat]
            lon, lat = data[0]["geometry"]["coordinates"]
            lats.append(lat)
            lons.append(lon)
            print(f"[{idx}/{total}] 成功: {addr} -> ({lat}, {lon})")
        else:
            lats.append(None)
            lons.append(None)
            print(f"[{idx}/{total}] 結果なし: {addr}")

    except Exception as e:
        print(f"[{idx}/{total}] 失敗: {addr} ({e})")
        lats.append(None)
        lons.append(None)

    time.sleep(0.2)  # 連続アクセス防止（重要）

df["lat"] = lats
df["lon"] = lons
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n完了: {OUTPUT_CSV}")

