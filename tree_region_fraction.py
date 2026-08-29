"""
航空写真の樹冠（影を落とす樹木）率を、ランドマーク矩形ごとに概算する。
近江町市場側（北の密な市街地）と金沢城公園側（中央の大きな緑塊）を比較。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = Path(__file__).resolve().parent
img = Image.open(root / "CCB20076-C10-16.jpg").convert("RGB")
arr = np.asarray(img).astype(np.float32)
H, W = arr.shape[:2]
R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
exg = 2 * G - R - B
bright = (R + G + B) / 3.0
content = bright > 25

# 樹冠: 緑が支配的でやや暗い（芝・明るい草地や灰色屋根を除外）
canopy = (exg > 22) & (G > R + 8) & (G >= B - 2) & (bright > 30) & (bright < 155) & content
# 緑地全般（芝も含む）: 参考
green_any = (exg > 14) & (G > R + 4) & content

print(f"image {W}x{H}  canopy={canopy.sum()/content.sum():.1%}  green_any={green_any.sum()/content.sum():.1%}")

# ランドマーク矩形 (x0,y0,x1,y1) 原画素
regions = {
    "金沢城公園・兼六園 (mesh72/73 相当・南)": (950, 1980, 1820, 2980),
    "近江町・下本多町方面 市街地 (mesh82 相当・北)": (1000, 1380, 2050, 1960),
    "[参照] 東部の森林丘陵 (高樹冠)": (3050, 1500, 3600, 2600),
    "[参照] 北西部の密集市街地 (低樹冠)": (300, 700, 1100, 1500),
}
rows = []
for name, (x0, y0, x1, y1) in regions.items():
    c = content[y0:y1, x0:x1]
    cn = canopy[y0:y1, x0:x1].sum() / max(1, c.sum())
    ga = green_any[y0:y1, x0:x1].sum() / max(1, c.sum())
    rows.append((name, cn, ga))
    print(f"  {name}: 樹冠率={cn:.1%}  緑地率={ga:.1%}")

plt.rcParams["font.sans-serif"] = ["MS Gothic", "Yu Gothic UI", "Meiryo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=120)
axes[0].imshow(img); axes[0].set_title("航空写真 CCB20076-C10-16（樹木分布の確認）", fontsize=12)
ov = np.asarray(img).copy()
ov[canopy] = [255, 0, 255]
axes[1].imshow(ov); axes[1].set_title("樹冠マスク（マゼンタ）", fontsize=12)
cols = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a"]
for (name, (x0, y0, x1, y1)), col in zip(regions.items(), cols):
    for ax in axes:
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=col, lw=2.2)
        ax.text(x0 + 8, y0 + 60, name.split(" ")[0], color=col, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))
fig.tight_layout()
out = root / "tree_region_fraction.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"saved: {out}")
