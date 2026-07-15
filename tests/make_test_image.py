"""Generate sample test images for OCR verification.

Creates Latin and (when a CJK font is available) Chinese/Japanese/Korean images
so the OCR pipeline can be exercised across all four languages.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Font search lists per script family, checked in order across
# Windows / Linux / macOS. Each language picks the first file that exists.
_FONT_BY_FAMILY: dict[str, list[str]] = {
    "latin": [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ],
    "cjk": [  # Chinese + Japanese
        "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ],
    "korean": [
        "C:/Windows/Fonts/malgun.ttf",    # Malgun Gothic (Hangul)
        "C:/Windows/Fonts/malgunbd.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansKR-Regular.otf",
        "/System/Library/Fonts/AppleGothic.ttf",
    ],
}


def _load_font(family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_BY_FAMILY.get(family, _FONT_BY_FAMILY["latin"]):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render(text: str, out: Path, family: str, size: int = 56) -> Path:
    img = Image.new("RGB", (760, 160), "white")
    draw = ImageDraw.Draw(img)
    draw.text((28, 50), text, fill="black", font=_load_font(family, size))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out}")
    return out


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "sample_data"
    render("PaddleOCR MCP Service 2024", out_dir / "en.png", "latin")
    render("Hello OCR \u4f60\u597d\u4e16\u754c 12345", out_dir / "zh.png", "cjk")
    render("\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", out_dir / "ja.png", "cjk")
    render("\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4", out_dir / "ko.png", "korean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
