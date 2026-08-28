# 統合ルート版（日陰 + CS 立ち寄り）

`static/index.html`（ルート画面）と同系 UI。  
**クーリングシェルター（CS）は最低1か所への立ち寄りが必須**です（経由なしの直行ルートはありません）。

## 起動

```powershell
py -3 -m uvicorn integrated.app:app --reload --host 127.0.0.1 --port 8766
```

http://127.0.0.1:8766/

## ルートの考え方

| 部分 | 内容 |
|------|------|
| 歩行区間 | 起点→CS、CS→終点（または起点→終点）の各 OSM 辺に、距離・時間・勾配・日陰の重み付きコスト |
| CS 要素 | **経由 waypoint** … 指定 CS、終点近傍の CS、または自動選択（総コスト最小） |
| 立ち寄りコスト | `weight_cooling_shelter × cs_dwell_time_s`（滞在時間のペナルティ／休憩の明示） |

**立ち寄り CS は既定で自動選定**です。折り返し距離の近い **上位8候補** を評価し、総コスト最小の1か所を選びます。手動指定は `visit_cs_id` または UI の③。

## 計算時間の目安

| 要因 | 内容 |
|------|------|
| API 起動直後 | バックグラウンドで金沢中心の歩道グラフをプリロード（初回ルートが速くなりやすい） |
| 初回（同一エリア・ディスク無） | Overpass 取得で **数十秒**。取得後は `data/osm_walk_cache/*.graphml` に保存 |
| 2回目以降 | メモリ／ディスクからグラフ再利用 + コスト付きグラフキャッシュ |
| CS 自動選定 | 全源ダイクストラ 2 回 + 候補8件のみ経路復元（失敗時 15→30 件） |
| その他 | CS 最近傍ノード一括計算、目的ノード限定ダイクストラ、日陰オフ時はサンプル省略 |
| 応答 | `stats.timing_s`、`stats.graph_load_source`（memory/disk/network） |

候補数は `route_ext.AUTO_CS_CANDIDATES`（既定 8）で変更できます。

## 精度の記録（高速化前後）

| ファイル | 内容 |
|----------|------|
| [`integrated/ACCURACY.md`](ACCURACY.md) | ベンチマーク集計の読み物版 |
| [`data/route_accuracy_benchmark.json`](../data/route_accuracy_benchmark.json) | ケース別の数値（JSON） |

再計測:

```powershell
py -3 scripts/benchmark_route_accuracy.py
```

API で1リクエストごとに差分だけ欲しい場合は `compare_accuracy: true`（自動 CS 時のみ。全57件評価が追加で走る）。

## API

`POST /api/route_latlon` ボディ例:

```json
{
  "orig_lat": 36.57, "orig_lon": 136.65,
  "dest_lat": 36.56, "dest_lon": 136.66,
  "route_date": "2026-08-05", "hour": 14,
  "visit_cs_id": null,
  "factors": {
    "use_cooling_shelter": true,
    "cs_dwell_time_s": 600,
    "weight_cooling_shelter": 2.5
  }
}
```

- `visit_cs_id`: 立ち寄り CS の `cs_id`。`null` で自動（終点が CS 近傍ならその CS を優先）
- 応答 `stats.cooling_shelter_visit` に立ち寄り先名称・区間数（1 or 2）が入ります

## データ

- `ジオコーディング/cooling_shelters.geojson`
- `data/summer_shadow_2026.npz`
