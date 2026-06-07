#!/usr/bin/env python3
"""
merge_images.py
---------------
Layers and merges multiple JPG or RAW images (NEF, CR2, ARW, DNG, etc.)
of the same subject in the current directory.

Strategy:
  1. Auto-detect all .jpg / .jpeg / .nef / .cr2 / .arw / .dng / .raf / .orf
     files in the script's directory.
  2. Align every image to the first one using ORB feature matching + homography
     (handles slight camera shifts, rotation, zoom differences).
  3. Merge aligned images using one of three modes:
       --mode mean    : average pixel values (reduces noise, sharpens detail)
       --mode median  : robust average, suppresses outliers / moving objects
       --mode focus   : focus-stacking — keeps the sharpest pixel per position

RAW handling:
  - NEF and other RAW files are decoded via rawpy using a standard camera-space
    conversion (no auto-brightness, no auto-white-balance — values are kept
    linear so blending math stays consistent across all input images).
  - RAW files are exported as 16-bit internally and down-converted to 8-bit for
    JPEG output, or kept as 16-bit if --out is a .tif / .tiff / .png file.

Usage:
  python merge_images.py                          # mean blend, auto output name
  python merge_images.py --mode median            # median blend
  python merge_images.py --mode focus             # focus stacking
  python merge_images.py --mode mean --out result.jpg
  python merge_images.py --mode mean --out result.tif   # 16-bit TIFF output
  python merge_images.py --no-align               # skip alignment
  python merge_images.py --raw-dir /path/to/nefs  # explicit directory
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# rawpy is required only if RAW files are present; import is deferred.
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False


# ──────────────────────────────────────────────────────────────
# Supported extensions
# ──────────────────────────────────────────────────────────────

JPEG_EXTS = {".jpg", ".jpeg"}
RAW_EXTS  = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}
ALL_EXTS  = JPEG_EXTS | RAW_EXTS


# ──────────────────────────────────────────────────────────────
# Image discovery
# ──────────────────────────────────────────────────────────────

def find_images(directory: Path) -> list[Path]:
    """Return sorted list of supported image files in *directory*."""
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ALL_EXTS
    )
    return files


# ──────────────────────────────────────────────────────────────
# Image loading  (JPEG or RAW)
# ──────────────────────────────────────────────────────────────

def load_image(path: Path, use_16bit: bool = False) -> np.ndarray:
    """
    Load a JPEG or RAW file and return a BGR numpy array.

    Parameters
    ----------
    path       : file to load
    use_16bit  : if True, RAW files are kept as uint16; otherwise clipped to uint8.
                 JPEG files are always uint8 regardless of this flag.
    """
    ext = path.suffix.lower()

    if ext in RAW_EXTS:
        if not HAS_RAWPY:
            sys.exit(
                "❌  rawpy is required to read RAW files.\n"
                "    Install it with:  pip install rawpy"
            )
        with rawpy.imread(str(path)) as raw:
            # postprocess with no auto-adjustments so images are comparable
            rgb = raw.postprocess(
                use_camera_wb=True,      # honour the in-camera white balance
                no_auto_bright=True,     # keep linear exposure
                output_bps=16,           # always decode to 16-bit
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )
        # rawpy returns RGB; convert to BGR for OpenCV consistency
        bgr16 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if use_16bit:
            return bgr16
        # Scale 16-bit → 8-bit
        return (bgr16 >> 8).astype(np.uint8)

    # JPEG / standard format
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


# ──────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────

def align_to_reference(ref: np.ndarray, src: np.ndarray) -> np.ndarray:
    """
    Return *src* warped to match *ref* using ORB + homography.
    Falls back to returning *src* unchanged if alignment fails.
    """
    # For 16-bit images, convert to 8-bit just for feature detection
    def to_gray8(img):
        if img.dtype == np.uint16:
            img8 = (img >> 8).astype(np.uint8)
        else:
            img8 = img
        return cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    ref_gray = to_gray8(ref)
    src_gray = to_gray8(src)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(src_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("    ⚠  Not enough features — skipping alignment for this image.")
        return src

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    n_good = max(8, int(len(matches) * 0.70))
    good = matches[:n_good]

    if len(good) < 4:
        print("    ⚠  Too few good matches — skipping alignment.")
        return src

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        print("    ⚠  Homography failed — skipping alignment.")
        return src

    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(
        src, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


# ──────────────────────────────────────────────────────────────
# Merge modes  (work on uint8 or uint16 arrays)
# ──────────────────────────────────────────────────────────────

def merge_mean(stack: list[np.ndarray]) -> np.ndarray:
    """Average all images pixel-wise."""
    acc = np.zeros_like(stack[0], dtype=np.float64)
    for img in stack:
        acc += img.astype(np.float64)
    result = acc / len(stack)
    return np.clip(result, 0, np.iinfo(stack[0].dtype).max).astype(stack[0].dtype)


def merge_median(stack: list[np.ndarray]) -> np.ndarray:
    """Median of all images pixel-wise (removes transient objects / noise)."""
    arr = np.stack([img.astype(np.float32) for img in stack], axis=0)
    return np.median(arr, axis=0).astype(stack[0].dtype)


def merge_focus(stack: list[np.ndarray]) -> np.ndarray:
    """
    Focus stacking: per-pixel, keep the value from the image with the
    highest local Laplacian variance (= sharpest detail).
    """
    def sharpness_map(img):
        if img.dtype == np.uint16:
            gray = cv2.cvtColor((img >> 8).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.boxFilter(lap ** 2, cv2.CV_64F, (15, 15))

    sharpness_arr = np.stack([sharpness_map(img) for img in stack], axis=0)
    best_idx = np.argmax(sharpness_arr, axis=0)   # (H, W)

    h, w, c = stack[0].shape
    result = np.zeros((h, w, c), dtype=stack[0].dtype)
    for ch in range(c):
        channel_stack = np.stack([img[:, :, ch] for img in stack], axis=0)
        result[:, :, ch] = channel_stack[best_idx, np.arange(h)[:, None], np.arange(w)]
    return result


MERGE_MODES = {
    "mean":   merge_mean,
    "median": merge_median,
    "focus":  merge_focus,
}


# ──────────────────────────────────────────────────────────────
# Output saving
# ──────────────────────────────────────────────────────────────

def save_result(result: np.ndarray, out_path: Path, quality: int) -> None:
    """Save *result* to *out_path*, choosing the right encoder automatically."""
    ext = out_path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        # TIFF supports 16-bit natively
        cv2.imwrite(str(out_path), result)
    elif ext == ".png":
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        # JPEG: ensure 8-bit
        if result.dtype == np.uint16:
            result = (result >> 8).astype(np.uint8)
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer and merge JPG/RAW images (NEF, CR2, ARW, DNG …) of the same subject."
    )
    parser.add_argument(
        "--mode", choices=list(MERGE_MODES), default="mean",
        help="Merge mode: mean (default), median, or focus (focus stacking).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output filename (default: merged_<mode>.jpg). Use .tif for 16-bit output.",
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="Skip alignment step (use when images are already aligned).",
    )
    parser.add_argument(
        "--dir", default=None,
        help="Directory containing images (default: directory of this script).",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality 1-100 (default: 95).",
    )
    parser.add_argument(
        "--16bit", dest="use_16bit", action="store_true",
        help="Process RAW files at full 16-bit depth (default: convert to 8-bit). "
             "Automatically enabled when --out is a .tif or .png file.",
    )
    args = parser.parse_args()

    # Resolve directory
    base_dir = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent
    out_name = args.out or f"merged_{args.mode}.jpg"
    out_path = base_dir / out_name

    # Auto-enable 16-bit for lossless output formats
    use_16bit = args.use_16bit or out_path.suffix.lower() in {".tif", ".tiff", ".png"}

    print(f"📂  Scanning: {base_dir}")
    files = find_images(base_dir)

    # Exclude the output file itself
    files = [f for f in files if f.resolve() != out_path.resolve()]

    if len(files) < 2:
        sys.exit(f"❌  Need at least 2 images, found {len(files)}.")

    raw_count  = sum(1 for f in files if f.suffix.lower() in RAW_EXTS)
    jpeg_count = sum(1 for f in files if f.suffix.lower() in JPEG_EXTS)
    print(f"🖼   Found {len(files)} images  ({raw_count} RAW, {jpeg_count} JPEG)")
    print(f"⚙️   Mode: {args.mode}  |  Align: {not args.no_align}  |  Bit depth: {'16-bit' if use_16bit else '8-bit'}")

    if raw_count and not HAS_RAWPY:
        sys.exit(
            "❌  rawpy is required to read RAW files.\n"
            "    Install it with:  pip install rawpy"
        )

    # Load reference image
    print(f"\n  [1/{len(files)}] Reference: {files[0].name}")
    ref = load_image(files[0], use_16bit=use_16bit)
    print(f"    {ref.shape[1]}×{ref.shape[0]}  dtype={ref.dtype}")

    stack = [ref]

    for i, fpath in enumerate(files[1:], start=2):
        print(f"  [{i}/{len(files)}] {'Aligning' if not args.no_align else 'Loading'}: {fpath.name} … ", end="", flush=True)
        try:
            img = load_image(fpath, use_16bit=use_16bit)
        except IOError as e:
            print(f"\n    ⚠  {e} — skipping.")
            continue

        # Resize to reference dimensions if needed
        if img.shape[:2] != ref.shape[:2]:
            print(f"\n    ↕  Resizing from {img.shape[1]}×{img.shape[0]} → {ref.shape[1]}×{ref.shape[0]}", end=" … ")
            img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        if not args.no_align:
            img = align_to_reference(ref, img)
            print("done")
        else:
            print("done")

        stack.append(img)

    print(f"\n🔀  Merging {len(stack)} images using '{args.mode}' …")
    result = MERGE_MODES[args.mode](stack)

    save_result(result, out_path, args.quality)
    print(f"✅  Saved → {out_path}")


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        sys.exit("❌  OpenCV not found. Install it with:\n    pip install opencv-python numpy")
    main()