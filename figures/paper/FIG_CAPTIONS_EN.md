# Figure captions (English) — Fig. 3-1 format

| Fig. | File | Caption |
|------|------|---------|
| **Fig. 3-1** | `word/fig3-1.png` | Overview of field comparison: (a) road cells (n = 468), measured trace vs. building-shadow simulation at 14:00 and 15:00 JST (F1 ≈ 0.22, accuracy 61–63%); (b) method-wise metrics (n = 1,165). Web flat model F1 = 0.23; Rhino reference F1 = 0.31. Building shade only; trees excluded. |
| **Fig. 3-2** | `word/fig3-2.png` | Field sun/shade on roads (blue = shade, red = sun) vs. simulation on PLATEAU road cells, 5 Aug. 2023. Evaluation: road ∩ valid field cells (n = 468). |
| **Fig. 3-3** | `word/fig3-3.png` | Spatial confusion on road cells: blue = TP, light green = TN, red = FP, orange = FN. Low shade recall (~14%) for the building-only model. |
| **Fig. 3-4** | `word/fig3-4.png` | Method comparison vs. field labels (n = 1,165, 14:00 JST): F1, precision, recall, accuracy, shade IoU. Ranked: ③ Rhino (0.31), ① geometry+DEM (0.27), ② flat/Web (0.23). |
| **Fig. 3-5** | `word/fig3-5.png` | Spatial shade accuracy by method (green = TP, orange = FN, etc.), ranked by F1. |
| **Fig. 3-6** | `word/fig3-6.png` | Measured shade/sun and predicted shade from three methods, 14:00 JST. |
| **Fig. 3-7** | `word/fig3-7.png` | Spatial accuracy (left) and overlay of field labels with method predictions (right). |

Word document: **`PlateauShade_Chapters2-3.docx`** (body in Japanese + embedded figures + English captions).

Regenerate:
```powershell
cd E:\plateaushade.04.26.27
py -3 assemble_paper_figures.py
py -3 build_paper_word.py
```
