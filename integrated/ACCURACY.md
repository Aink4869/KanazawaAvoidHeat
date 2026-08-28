# ルート高速化と精度（記録）

- 記録日時（UTC）: `2026-05-18T06:53:03.567944+00:00`
- 対象日・時刻: `2026-08-05` 14 時
- 生データ: [`data/route_accuracy_benchmark.json`](../data/route_accuracy_benchmark.json)

## 比較プロファイル

| 項目 | 本番（高速） | 基準（精度） |
|------|-------------|-------------|
| CS 候補数 | 8 | 57 |
| 日陰サンプル/辺 | 3 | 8 |
| CS 選定 | directed | directed |

## 集計（ケース平均）

### CS 候補 8 vs 57（同一グラフ）

- 同一 CS 率: **80.0%**
- 総コスト差 |Δ%| 中央値: **0.0%**
- 総コスト差 |Δ%| 最大: **0.0629%**

### 日陰サンプル 3 vs 8

- 同一 CS 率: **100.0%**
- 総コスト差 |Δ%| 中央値: **0.909%**
- 総コスト差 |Δ%| 最大: **1.2725%**

### API compare_accuracy（本番ルート）

- 同一 CS 率: **80.0%**
- 総コスト差 |Δ%| 中央値: **0.0%**
- 総コスト差 |Δ%| 最大: **0.0629%**

## ケース別

| ID | 同一CS | コストΔ% | 距離Δ% | 日陰率Δ |
|----|--------|---------|--------|---------|
| station_to_korinbo | ✓ | 0.0 | 0.0 | 0.0 |
| kenrokuen_to_station | ✓ | 0.0 | 0.0 | 0.0 |
| katamachi_to_higashichaya | ✓ | 0.0 | 0.0 | 0.0 |
| west_to_musashigatsuji | ✓ | 0.0 | 0.0 | 0.0 |
| long_cross_city | ✗ | 0.0629 | 1.8703 | 0.023489 |

## 再計測

```powershell
py -3 scripts/benchmark_route_accuracy.py
```

API で1リクエストごとに差分を付ける: `POST /api/route_latlon` に `compare_accuracy: true`
