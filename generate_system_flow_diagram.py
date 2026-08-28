# -*- coding: utf-8 -*-
"""Generate high-quality system flow diagram with tree modeling details."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "system_flow_diagram_en.png"

# Setup figure
fig = plt.figure(figsize=(19, 10.5), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Colors
COLOR_BG = "#f8fafc"
COLOR_HEADER_BG = "#0f766e"  # Teal dark
COLOR_CARD_BORDER = "#cbd5e1"

# Box color styles
C_GREEN_BG = "#f0fdf4"
C_GREEN_BORDER = "#16a34a"
C_GREEN_HEADER = "#15803d"

C_BLUE_BG = "#eff6ff"
C_BLUE_BORDER = "#2563eb"
C_BLUE_HEADER = "#1d4ed8"

C_AMBER_BG = "#fffbeb"
C_AMBER_BORDER = "#d97706"
C_AMBER_HEADER = "#b45309"

C_PURPLE_BG = "#faf5ff"
C_PURPLE_BORDER = "#9333ea"
C_PURPLE_HEADER = "#7e22ce"

C_SUBBOX_BG = "#ffffff"
C_SUBBOX_BORDER = "#94a3b8"

# Background canvas
fig.patch.set_facecolor(COLOR_BG)

# Title Banner
banner = patches.FancyBboxPatch((3, 91.5), 94, 6.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                                facecolor=COLOR_HEADER_BG, edgecolor="none", zorder=2)
ax.add_patch(banner)
ax.text(50, 94.75, "System Flow of Proposed Shade Estimation, Validation & Navigation Framework",
        ha="center", va="center", color="white", fontsize=15.5, fontweight="bold", zorder=3)


def draw_card(x, y, w, h, title, bg_color, border_color, header_color, title_size=11):
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=1.8, zorder=2)
    ax.add_patch(card)
    # Header bar
    hbar = patches.FancyBboxPatch((x, y + h - 4.5), w, 4.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                  facecolor=header_color, edgecolor="none", zorder=3)
    ax.add_patch(hbar)
    ax.text(x + w / 2, y + h - 2.25, title, ha="center", va="center", color="white",
            fontsize=title_size, fontweight="bold", zorder=4)


def draw_subbox(x, y, w, h, title, items, title_color="#0f172a", text_size=8.5, title_size=9.2):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                                 facecolor=C_SUBBOX_BG, edgecolor=C_SUBBOX_BORDER, linewidth=1.0, zorder=3)
    ax.add_patch(box)
    ax.text(x + 1.2, y + h - 1.8, title, ha="left", va="center", color=title_color,
            fontsize=title_size, fontweight="bold", zorder=4)
    cy = y + h - 3.8
    for item in items:
        ax.text(x + 1.2, cy, item, ha="left", va="top", color="#334155",
                fontsize=text_size, linespacing=1.28, zorder=4)
        cy -= (item.count("\n") + 1) * 2.05


def draw_arrow(x1, y1, x2, y2, color="#475569", lw=2.0, style="-|>", rad=0.0):
    connection = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connection, shrinkA=3, shrinkB=3),
                zorder=5)


# ==========================================
# 1. LEFT COLUMN: 3D Geometric Modeling
# ==========================================
# (1) Building & Terrain Modeling (Top-Left)
draw_card(3, 50.5, 27.5, 38.5, "(1) Building & Terrain Modeling", C_GREEN_BG, C_GREEN_BORDER, C_GREEN_HEADER)
draw_subbox(4.2, 69.5, 25.1, 15.0, "[Input Data]", [
    "• PLATEAU 3D Urban Model (LOD1/LOD2)",
    "  - Building footprints & measured heights",
    "• PLATEAU DEM (Digital Elevation Model)",
    "• High-precision terrain mesh"
], title_color=C_GREEN_HEADER)
draw_subbox(4.2, 52.0, 25.1, 15.5, "[Modeling Process]", [
    "• Coordinate unification to EPSG:6670",
    "• 3D building wall geometry construction",
    "• Ground elevation matching with DEM",
    "• Automated preprocessing pipeline"
], title_color=C_GREEN_HEADER)
draw_arrow(16.75, 69.5, 16.75, 67.5, color=C_GREEN_BORDER, lw=1.5)

# (2) Tree Model Construction (Aerial Calibration) (Bottom-Left)
draw_card(3, 4.0, 27.5, 43.5, "(2) Tree Model Construction (Aerial)", C_GREEN_BG, C_GREEN_BORDER, C_GREEN_HEADER)
draw_subbox(4.2, 33.5, 25.1, 9.5, "[Input Data]", [
    "• High-resolution aerial orthoimage",
    "  (nendophoto2024, 0.2m/px RGB, Kanazawa)"
], title_color=C_GREEN_HEADER)
draw_subbox(4.2, 16.5, 25.1, 15.5, "[Canopy Extraction & Height]", [
    "• ExG Index: ExG = 2G - R - B",
    "• Adaptive thresholding (Canopy >= 20%)",
    "• Masking of building footprints (~bldg)",
    "• Shadow-length median calibration:",
    "  H ≈ 14.9 m (n = 165 samples, α ≈ 48.9°)"
], title_color=C_GREEN_HEADER)
draw_subbox(4.2, 5.5, 25.1, 9.5, "[Geometric Modeling]", [
    "• 3D prismatic tree column extrusion",
    "• High-precision tree shade generation"
], title_color=C_GREEN_HEADER)
draw_arrow(16.75, 33.5, 16.75, 32.0, color=C_GREEN_BORDER, lw=1.5)
draw_arrow(16.75, 16.5, 16.75, 15.0, color=C_GREEN_BORDER, lw=1.5)


# ==========================================
# 2. CENTER COLUMN: Shade Calculation Engine
# ==========================================
draw_card(34.5, 4.0, 31.0, 85.0, "(3) Hybrid Shade Calculation Engine", C_BLUE_BG, C_BLUE_BORDER, C_BLUE_HEADER, title_size=11.5)

draw_subbox(36.0, 75.5, 28.0, 8.5, "[User Inputs / Query]", [
    "• Date & time specified by user (6:00–18:00)",
    "• Target resolution (2m analysis / 12m routing)"
], title_color=C_BLUE_HEADER)

draw_subbox(36.0, 62.5, 28.0, 10.5, "[Solar Position (pvlib)]", [
    "• Precise solar position calculation",
    "• Solar Altitude Angle (θ) & Azimuth (φ)",
    "• Ephemeris computation for target time"
], title_color=C_BLUE_HEADER)

draw_subbox(36.0, 24.5, 28.0, 35.5, "[Shade Simulation & Synthesis]", [
    "• Ray-Tracing Shadow Casting Engine:",
    "  - Building shadow on DEM terrain",
    "  - Calibrated tree shadow (H ≈ 14.9 m)",
    "",
    "• High-Performance Hybrid Formulation:",
    "  1) Lightweight building shade:",
    "     S_light = Shade(Buildings, DEM)",
    "  2) High-precision tree residual shade:",
    "     S_residual = Shade(Trees) \\ S_light",
    "  3) Composite synthesized shade:",
    "     S_composite = S_light ∪ S_residual",
    "",
    "• Fast disk/memory caching (NPZ / NPY)"
], title_color=C_BLUE_HEADER)

draw_subbox(36.0, 6.0, 28.0, 16.0, "[System Output: Shade Raster]", [
    "• Binary Shade Raster Map (2m grid, EPSG:6670)",
    "• Pixel values: 0 = Sunlit, 1 = Shaded",
    "• Overlay layers for Web Map Visualization",
    "  - Blue: Building shade, Green: Tree residual"
], title_color=C_BLUE_HEADER)

draw_arrow(50.0, 75.5, 50.0, 73.0, color=C_BLUE_BORDER, lw=1.5)
draw_arrow(50.0, 62.5, 50.0, 60.0, color=C_BLUE_BORDER, lw=1.5)
draw_arrow(50.0, 24.5, 50.0, 22.0, color=C_BLUE_BORDER, lw=1.5)


# ==========================================
# 3. RIGHT COLUMN: Validation & Web System
# ==========================================
# (4) Accuracy Evaluation & Validation (Top-Right)
draw_card(69.5, 47.5, 27.5, 41.5, "(4) Accuracy Evaluation & Validation", C_AMBER_BG, C_AMBER_BORDER, C_AMBER_HEADER)
draw_subbox(70.7, 73.0, 25.1, 11.5, "[Ground-Truth Validation Data]", [
    "• 2023-08-05 Field survey walk map",
    "  - 10:30, 12:30, 14:30 JST (GeoTIFF)",
    "  - Blue = Shaded, Red = Sunlit"
], title_color=C_AMBER_HEADER)

draw_subbox(70.7, 49.0, 25.1, 22.0, "[Accuracy Evaluation & Findings]", [
    "• Confusion Matrix on Road Cells:",
    "  - Recall, Precision, F1-Score, IoU",
    "• Spatial evaluation across 3 areas:",
    "  - Omicho Market, Kanazawa Castle, Total",
    "• Quantitative Impact of Tree Shade:",
    "  - Building-only (14:30): F1 ≈ 0.29, Rec ≈ 20%",
    "  - Building + Tree: F1 ≈ 0.43, Rec ≈ 42%",
    "  - Massive reduction in False Negatives (FN)",
    "• Building model test: LOD0 wall superior to LOD2"
], title_color=C_AMBER_HEADER)
draw_arrow(83.25, 73.0, 83.25, 71.0, color=C_AMBER_BORDER, lw=1.5)

# (5) Heatstroke Risk Mitigation Web System (Bottom-Right)
draw_card(69.5, 4.0, 27.5, 40.5, "(5) Heatstroke Mitigation Web System", C_PURPLE_BG, C_PURPLE_BORDER, C_PURPLE_HEADER)
draw_subbox(70.7, 26.5, 25.1, 13.5, "[Integrated Infrastructure]", [
    "• OSM Walking Network (NetworkX Graph)",
    "  - Multi-factor cost: Distance, Time, DEM Slope,",
    "    and Real-time Path Shade Ratio",
    "• 57 Cooling Shelters (CS, GSI geocoded)"
], title_color=C_PURPLE_HEADER)

draw_subbox(70.7, 5.5, 25.1, 19.0, "[Navigation Web Services (FastAPI)]", [
    "• Multi-factor shortest path routing (Dijkstra)",
    "• Automated CS waypoint selection & dwell cost",
    "• 3-Tier Multi-System User Interface:",
    "  1) Legacy System (/route): Building shade only",
    "  2) Latest System (/route/new): Tree added",
    "  3) Route Comparison (/route/compare):",
    "     Side-by-side overlay & delta analysis",
    "• Tourist safety & thermal comfort enhancement"
], title_color=C_PURPLE_HEADER)
draw_arrow(83.25, 26.5, 83.25, 24.5, color=C_PURPLE_BORDER, lw=1.5)


# ==========================================
# 4. CROSS-COLUMN ARROWS
# ==========================================
# (1) -> (3)
draw_arrow(30.5, 69.5, 34.5, 69.5, color=C_GREEN_BORDER, lw=2.2)
# (2) -> (3)
draw_arrow(30.5, 26.0, 34.5, 35.0, color=C_GREEN_BORDER, lw=2.2)

# (3) -> (4)
draw_arrow(65.5, 68.0, 69.5, 68.0, color=C_BLUE_BORDER, lw=2.2)
# (3) -> (5)
draw_arrow(65.5, 24.0, 69.5, 24.0, color=C_BLUE_BORDER, lw=2.2)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Diagram successfully generated at: {OUT_PATH}")
