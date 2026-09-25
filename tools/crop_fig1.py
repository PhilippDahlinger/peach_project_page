"""Render Figure 1 (overview) and cut out the parts used in the "How PEACH works" cards.

Steps:
  1. render the full figure to docs/static/images/fig1_overview.png
  2. remove the three section frames (each is a single vector path; a tiny redaction that touches
     the frame's top edge removes the whole path) and white out the section titles
  3. crop each part, trim surrounding white space and add a uniform margin

Usage:  python tools/crop_fig1.py <peach_fig1.pdf> [--out docs/static/images]
Requires pymupdf and pillow. Crop boxes are in PDF points of the 755 x 467 pt figure.
"""
import argparse
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

# tiny rectangles on the top edges of the three section frames
FRAME_EDGES = [(300, 11.5, 302, 14.5), (460, 162.5, 462, 165.5), (120, 316.5, 122, 319.5)]
# white-outs: section titles "Mesh Simulation" / "Auxiliary Losses" and arrow stubs at crop borders
WHITEOUTS = [(215, 170, 356, 196), (215, 322, 356, 348), (385, 62, 399, 110)]
CROPS = {
    "observe": (18, 38, 200.5, 156),
    "encode": (215.8, 38, 390, 132),
    "aggregate": (399, 58, 535, 114),
    "simulate": (24, 174, 428.3, 304),
    "aux_material": (28, 350, 300, 428),
    "aux_sdf": (306, 326, 728, 448),
}


def main(pdf, out):
    out = Path(out)
    (out / "steps").mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    page = doc[0]
    page.get_pixmap(dpi=220).save(out / "fig1_overview.png")

    for r in FRAME_EDGES:
        page.add_redact_annot(pymupdf.Rect(*r), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                          text=pymupdf.PDF_REDACT_TEXT_NONE)
    for r in WHITEOUTS:
        page.draw_rect(pymupdf.Rect(*r), color=None, fill=(1, 1, 1), overlay=True)

    for name, r in CROPS.items():
        pix = page.get_pixmap(dpi=300, clip=pymupdf.Rect(*r))
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        im = im.crop(ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox())
        canvas = Image.new("RGB", (im.width + 24, im.height + 24), "white")
        canvas.paste(im, (12, 12))
        canvas.save(out / "steps" / f"{name}.png", optimize=True)
        print(f"{name}: {canvas.size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="docs/static/images")
    a = ap.parse_args()
    main(a.pdf, a.out)
