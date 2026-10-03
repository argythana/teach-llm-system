#!/usr/bin/env python
"""Draw the annotated illustrations of the infrastructure guides.

    uv run python tools/draw_guide_illustrations.py              # all of them
    uv run python tools/draw_guide_illustrations.py --only 01d   # names containing "01d"
    uv run python tools/draw_guide_illustrations.py --list

The pictures in `lecture_*/infra_tools/screenshots/` are drawings, not captures: each
one is a small HTML page (tools/guide_illustrations/*.py) that imitates a Windows,
VS Code or Hugging Face screen, with green circles on what the student must click.
This script renders every page with headless Firefox at double resolution and saves
it as a palette PNG, about a third of the size of the raw capture.

The PNGs are committed, so students never run this. Run it only after editing a page,
and look at the result: the pages use the Noto Sans and DejaVu Sans Mono fonts, and a
computer without them draws slightly different pictures.

Needs Firefox on PATH. Pillow comes with matplotlib, a course dependency.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guide_illustrations import hf_token, vscode_kernel, windows  # noqa: E402

OUTPUT_DIR = REPO / "lecture_01_ollama_models_prompts_langchain/infra_tools/screenshots"
MODULES = (windows, vscode_kernel, hf_token)
PAGE_WIDTH = 1000  # CSS pixels; every page sets `zoom: 2`, so the PNG is twice as wide
SCALE = 2


def pages():
    """Yield (name, html document, height in CSS pixels) for every illustration."""
    for module in MODULES:
        for name, (body, height) in module.PAGES.items():
            html = (
                '<!doctype html><html><head><meta charset="utf-8">'
                f"<style>{module.CSS}</style></head>{body}</html>"
            )
            yield name, html, height


def render(firefox, html, height, png, workdir):
    """Capture one page with headless Firefox, in a throwaway profile."""
    page = workdir / "page.html"
    page.write_text(html, encoding="utf-8")
    subprocess.run(
        [
            firefox,
            "--headless",
            "--no-remote",
            "--profile",
            str(workdir / "profile"),
            "--window-size",
            f"{PAGE_WIDTH * SCALE},{height * SCALE}",
            "--screenshot",
            str(png),
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def compress(png):
    """Rewrite the PNG with a 256-colour palette: flat drawings lose nothing visible."""
    image = Image.open(png).convert("RGB")
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(png, optimize=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", help="draw only the names containing this text")
    parser.add_argument("--list", action="store_true", help="print the names and exit")
    args = parser.parse_args()

    selected = [p for p in pages() if not args.only or args.only in p[0]]
    if args.list:
        print("\n".join(name for name, _, _ in selected))
        return
    if not selected:
        sys.exit(f"No illustration name contains {args.only!r}.")

    firefox = shutil.which("firefox")
    if firefox is None:
        sys.exit("Firefox is not on PATH; it renders the pages.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        profile = workdir / "profile"
        profile.mkdir()
        # Text is laid out for a high-density screen, which the circles were placed on.
        (profile / "user.js").write_text(
            'user_pref("layout.css.devPixelsPerPx", "2.0");\n', encoding="utf-8"
        )
        for name, html, height in selected:
            png = OUTPUT_DIR / f"{name}.png"
            render(firefox, html, height, png, workdir)
            compress(png)
            print(f"{png.relative_to(REPO)}  {png.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
