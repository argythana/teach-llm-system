#!/usr/bin/env python
"""Draw the annotated illustrations of the infrastructure guides.

    uv run python tools/draw_guide_illustrations.py              # all of them
    uv run python tools/draw_guide_illustrations.py --only 01d   # names containing "01d"
    uv run python tools/draw_guide_illustrations.py --list

The pictures in `lecture_*/infra_tools/screenshots/` are drawings, not captures: each
one is a small HTML page (tools/guide_illustrations/*.py) that imitates a Windows,
PowerShell, VS Code, Hugging Face, MLflow or Docker Desktop screen, with green circles on
what the student must click. A picture goes to the lecture its name starts with:
`02a_...` to `lecture_02_*/infra_tools/screenshots/`.
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

from guide_illustrations import (  # noqa: E402
    docker_desktop,
    hf_token,
    mlflow_ui,
    powershell,
    vscode_kernel,
    vscode_terminal,
    windows,
)

MODULES = (
    windows,
    vscode_kernel,
    hf_token,
    powershell,
    vscode_terminal,
    mlflow_ui,
    docker_desktop,
)
PAGE_WIDTH = 1000  # CSS pixels; every page sets `zoom: 2`, so the PNG is twice as wide
SCALE = 2


def output_dir(name):
    """The screenshots folder of the lecture a picture's name starts with ("02a_..." -> lecture 2)."""
    (infra_tools,) = REPO.glob(f"lecture_{name[:2]}_*/infra_tools")
    return infra_tools / "screenshots"


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

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        profile = workdir / "profile"
        profile.mkdir()
        # Text is laid out for a high-density screen, which the circles were placed on.
        (profile / "user.js").write_text(
            'user_pref("layout.css.devPixelsPerPx", "2.0");\n', encoding="utf-8"
        )
        for name, html, height in selected:
            png = output_dir(name) / f"{name}.png"
            png.parent.mkdir(exist_ok=True)
            render(firefox, html, height, png, workdir)
            compress(png)
            print(f"{png.relative_to(REPO)}  {png.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
