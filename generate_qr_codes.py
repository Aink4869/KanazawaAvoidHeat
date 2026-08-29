# -*- coding: utf-8 -*-
"""Generate high-resolution QR codes with clean labels."""
import qrcode
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path("E:/plateaushade.04.26.27")
OUT_DIR = ROOT / "figures" / "qr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITEMS = [
    {
        "filename": "qr_render_compare.png",
        "url": "https://kanazawaavoidheat.onrender.com/compare",
        "title1": "Kanazawa Route Comparison (Old vs New)",
        "title2": "https://kanazawaavoidheat.onrender.com/compare",
    },
    {
        "filename": "qr_render_main.png",
        "url": "https://kanazawaavoidheat.onrender.com/",
        "title1": "Kanazawa Tree & Shade Navigator",
        "title2": "https://kanazawaavoidheat.onrender.com/",
    },
    {
        "filename": "qr_github_repo.png",
        "url": "https://github.com/Aink4869/KanazawaAvoidHeat",
        "title1": "GitHub Source Code",
        "title2": "https://github.com/Aink4869/KanazawaAvoidHeat",
    },
]

for item in ITEMS:
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=3,
    )
    qr.add_data(item["url"])
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0f172a", back_color="#ffffff").convert("RGBA")

    w, h = img.size
    canvas = Image.new("RGBA", (w, h + 60), "#ffffff")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Text
    draw.text((w / 2, h + 8), item["title1"], fill="#0f766e", anchor="mt")
    draw.text((w / 2, h + 30), item["title2"], fill="#475569", anchor="mt")

    out_path = OUT_DIR / item["filename"]
    canvas.save(out_path)
    print(f"Generated: {out_path}")
