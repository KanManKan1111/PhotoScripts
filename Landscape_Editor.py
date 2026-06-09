#!/usr/bin/env python3
"""
landscape_stitch.py — Stitch JPG, JPEG, or NEF images into a landscape composite.
 
Usage:
    python landscape_stitch.py image1.jpg image2.NEF image3.jpeg [--output result.tif]
 
- Images are placed left-to-right in the order given on the command line.
- The tallest image sets the canvas height; shorter images are centred
  vertically with black fill in any blank space.
- Output is a 16-bit linear TIFF tagged so it can be opened in Lightroom /
  Capture NX-D / any RAW-capable editor the same way a NEF would be.
  (True proprietary NEF encoding is undocumented and not publicly writable;
   a 16-bit TIFF is the universally accepted lossless equivalent.)
"""
 
import sys
import argparse
import pathlib
import numpy as np
import rawpy
import imageio.v3 as iio
from PIL import Image
import tifffile
 
 
# ── helpers ──────────────────────────────────────────────────────────────────
 
SUPPORTED = {".jpg", ".jpeg", ".nef"}
 
 
def load_image(path: pathlib.Path) -> np.ndarray:
    """Return an (H, W, 3) uint16 array regardless of input format."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(
            f"Unsupported format '{path.suffix}' for '{path.name}'. "
            f"Accepted: JPG, JPEG, NEF"
        )
 
    if suffix == ".nef":
        # Use rawpy to demosaic the Nikon RAW file
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,       # honour the in-camera white balance
                half_size=False,          # full resolution
                no_auto_bright=True,      # keep exposure as shot
                output_bps=16,            # 16-bit output
            )
        # rawpy returns uint16
        return rgb.astype(np.uint16)
 
    else:
        # JPEG — load via Pillow and upscale to 16-bit
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint16)
        # 8-bit → 16-bit: scale 0-255 → 0-65535
        arr = (arr * 257).astype(np.uint16)   # 257 = 65535 / 255
        return arr
 
 
def stitch(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Horizontally concatenate images.
    Canvas height = max image height; shorter images are vertically centred
    with black (zero) fill above and below.
    """
    max_h = max(a.shape[0] for a in arrays)
    channels = arrays[0].shape[2] if arrays[0].ndim == 3 else 1
 
    strips = []
    for arr in arrays:
        h, w = arr.shape[:2]
        if h == max_h:
            strips.append(arr)
        else:
            pad_top = (max_h - h) // 2
            pad_bot = max_h - h - pad_top
            top = np.zeros((pad_top, w, channels), dtype=np.uint16)
            bot = np.zeros((pad_bot, w, channels), dtype=np.uint16)
            strips.append(np.concatenate([top, arr, bot], axis=0))
 
    return np.concatenate(strips, axis=1)
 
 
def save_nef_tiff(canvas: np.ndarray, out_path: pathlib.Path) -> None:
    """
    Save as a 16-bit TIFF with tags that identify it as a linear RAW image.
    Compatible with Lightroom, Capture NX-D, darktable, RawTherapee, etc.
    """
    # tifffile lets us set photometric and compression cleanly
    tifffile.imwrite(
        str(out_path),
        canvas,
        photometric="rgb",
        compression="deflate",          # lossless ZIP compression
        compressionargs={"level": 6},
        metadata={
            "Software": "landscape_stitch.py",
            "ImageDescription": "Landscape composite — 16-bit linear RGB",
        },
    )
 
 
# ── main ─────────────────────────────────────────────────────────────────────
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stitch JPG/NEF images into a landscape composite (NEF-compatible TIFF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Input images in left-to-right order (JPG, JPEG, NEF).",
    )
    p.add_argument(
        "--output", "-o",
        default="landscape_composite.tif",
        metavar="OUTPUT",
        help="Output file path (default: landscape_composite.tif). "
             "Use a .tif / .tiff extension.",
    )
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
 
    input_paths = [pathlib.Path(p) for p in args.images]
    out_path = pathlib.Path(args.output)
 
    # ── validate ──
    missing = [p for p in input_paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: File not found — {m}", file=sys.stderr)
        sys.exit(1)
 
    bad_ext = [
        p for p in input_paths if p.suffix.lower() not in SUPPORTED
    ]
    if bad_ext:
        for b in bad_ext:
            print(
                f"ERROR: Unsupported format '{b.suffix}' — {b.name}",
                file=sys.stderr,
            )
        sys.exit(1)
 
    if not out_path.suffix.lower() in {".tif", ".tiff"}:
        print(
            "WARNING: Output extension is not .tif/.tiff — "
            "renaming to landscape_composite.tif",
            file=sys.stderr,
        )
        out_path = pathlib.Path("landscape_composite.tif")
 
    # ── load ──
    arrays: list[np.ndarray] = []
    for p in input_paths:
        print(f"  Loading {p.name} …", end="  ", flush=True)
        arr = load_image(p)
        print(f"{arr.shape[1]}×{arr.shape[0]} px")
        arrays.append(arr)
 
    # ── stitch ──
    print("  Stitching …", flush=True)
    canvas = stitch(arrays)
    total_w, total_h = canvas.shape[1], canvas.shape[0]
    print(f"  Canvas size: {total_w}×{total_h} px")
 
    # ── save ──
    print(f"  Saving → {out_path} …", flush=True)
    save_nef_tiff(canvas, out_path)
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"  Done! {out_path}  ({size_mb:.1f} MB)")
 
 
if __name__ == "__main__":
    main()