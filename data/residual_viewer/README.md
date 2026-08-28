# 残差樹木日陰マップ（ビューア）

地理院写真の上に、残差分解レイヤを重ねて確認する Web UI です。

## 起動

```powershell
# 初回またはシナリオ追加時: キャッシュ生成（数分）
py -3 residual_tree_shade_aerial_time.py --cell-m 2 --only ccb2007_nendo_1045,s2_20230724_1047

# ビューア
py -3 -m uvicorn residual_viewer_app:app --host 127.0.0.1 --port 8767
```

ブラウザ: http://127.0.0.1:8767/

| ポート | アプリ |
|--------|--------|
| 8765 | ナビ（日陰+ルート） |
| 8766 | 統合（CS立ち寄り） |
| **8767** | **残差樹木日陰マップ** |

## できること

- シナリオ切替（CCB2007 10:45 / Sentinel-2 10:47 など）
- プリセット: 分解カラー / S+B / 樹木成分 / 再構成 / 残差のみ
- レイヤ個別 ON/OFF・透明度・道路セル限定
- F1・セル数の表示

## データ

`data/residual_viewer/*.npz` + `*.json`（`residual_tree_shade_aerial_time.py` が出力）
