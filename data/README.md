# data/

統合 API（`http://127.0.0.1:8766/`）で必要なファイルです。

## 必須（Git に含める）

- `summer_shadow_2026.npz` — 日陰マスク（時系列）
- `summer_shadow_2026_meta.json` — グリッド・日付範囲
- `summer_shadow_2026_times.json` — 時刻インデックス

## 推奨（初回ルートを速くする）

- `osm_walk_cache/*.graphml` — Overpass 取得済み歩道グラフ

無い場合は初回ルート計算時にネットワークから取得します（数十秒かかることがあります）。

## 任意

- `route_accuracy_benchmark.json` — 精度ベンチマーク結果（`scripts/benchmark_route_accuracy.py` で再生成可）

## 再生成（開発者向け）

日陰 npz は `plateaudata/` と `compute_summer_shadows.py` で作ります（`plateaudata` は Git 対象外）。
