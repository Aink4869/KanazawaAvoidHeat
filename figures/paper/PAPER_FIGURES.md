# 論文用図一覧（自動生成）

生成日時の入力ファイル:
- `field_road_tran_compare_2023-08-05_20260511_142957_maps.png`
- `field_road_tran_compare_2023-08-05_20260511_142957_agreement.png`
- `field_baseline_method_compare_2023-08-05_14h_20260504_220955_maps.png`
- `field_baseline_method_compare_2023-08-05_14h_20260504_220955_scorecard.png`
- `field_baseline_method_compare_2023-08-05_14h_20260504_220955_shade_accuracy_spatial.png`

## 推奨配置

| ファイル | 論文での用途 | キャプション案 |
|----------|--------------|----------------|
| `paper_fig01_road_field_vs_sim.png` | 第3章 3.4 | 図1　2023年8月5日の実測歩行トレース（左）と，PLATEAU道路セル上の平坦地建物影シミュレーション（14時・15時）。評価は道路∩実測有効セル（*n*=468）。右2パネルは F1≈0.22，正解率61〜63%。建物影のみのため実測日陰（樹木等を含む）を過小評価する傾向が見られる。 |
| `paper_fig02_road_agreement.png` | 第3章 3.4 | 図2　道路セルにおける実測と予測の混同行列の空間分布。橙（FN）が多く，再現率が低い（約14%）ことを示す。 |
| `paper_fig03_method_scorecard.png` | 第3章 3.3–3.4 | 図3　実測有効セル（*n*=1165）に対する解析手法別の F1・適合率・再現率・正解率・IoU。③Rhino（Ladybug PDF ラスタ）が F1 最大（0.31）だが，②平坦地（Web同式）は 0.23。 |
| `paper_fig04_method_spatial_accuracy.png` | 第3章 3.4 | 図4　手法別の日陰正誤の空間分布（緑=TP，橙=FN 等）。建物影モデルは実測日陰の取りこぼし（FN）が広域に分布する。 |
| `paper_fig05_method_shade_maps.png` | 第3章 3.3 | 図5　実測と各手法の日陰分布比較（14:00）。 |
| `paper_fig06_evaluation_summary.png` | 第3章 冒頭または3.4 | 図6　道路限定評価（a）と全域手法比較（b）の概要。 |
| `paper_fig07_evaluation_full_stack.png` | 付録 | 評価図の一括版。 |
| `paper_fig08_method_spatial_and_overlay.png` | 第3章 3.3 | 図8　空間精度と手法重ね合わせ。 |

## 図中の略語

- **TP** 実測日陰かつ予測日陰　**TN** 実測日向かつ予測日向  
- **FP** 実測日向かつ予測日陰　**FN** 実測日陰かつ予測日向  
- **F1** 日陰クラスの調和平均 F 値　**正解率** (TP+TN)/n  

## 注意（本文に1文）

実測には樹木等の日陰を含むが，本モデルは建物影のみを対象とする（樹木なし前提）。
