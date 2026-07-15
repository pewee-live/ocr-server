"""Direct OCR engine smoke test (no MCP layer).

Usage: python tests/test_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from ocr_mcp.ocr_engine import OCREngine, load_image, run_ocr

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

# Map sample file -> expected substring (lowercased) for a quick sanity check.
EXPECT = {
    "en.png": "paddleocr",
    "zh.png": "\u4f60\u597d",
    "ja.png": "\u4e16\u754c",
    "ko.png": "\uc138\uacc4",
}


def main() -> int:
    engine = OCREngine()
    failures: list[str] = []

    for name, needle in EXPECT.items():
        path = SAMPLE_DIR / name
        if not path.exists():
            print(f"[skip] {name} not found")
            continue
        lang = {"en.png": "en", "zh.png": "ch", "ja.png": "japan", "ko.png": "korean"}[name]
        eng = engine.get_engine(lang)
        items = run_ocr(eng, load_image(str(path)))
        text = "\n".join(it["text"] for it in items).strip()
        print(f"\n=== {name} ({lang}) ===")
        for it in items:
            conf = f"{it['confidence']:.2f}" if it["confidence"] is not None else "?"
            print(f"  [{conf}] {it['text']}")
        ok = needle.lower() in text.lower()
        print("RESULT:", "OK" if ok else f"FAIL (missing '{needle}')")
        if not ok:
            failures.append(name)

    print("\n==== engine test summary ====")
    print("FAILURES:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
