#!/usr/bin/env python3
"""
merge_images.py
---------------
Layers and merges multiple JPG images of the same subject in the current directory.
 
Strategy:
  1. Auto-detect all .jpg / .jpeg files in the script's directory.
  2. Align every image to the first one using ORB feature matching + homography
     (handles slight camera shifts, rotation, zoom differences).
  3. Merge aligned images using one of three modes:
       --mode mean    : average pixel values (reduces noise, sharpens detail)
       --mode median  : robust average, suppresses outliers / moving objects
       --mode focus   : focus-stacking — keeps the sharpest pixel per position
 
Usage:
  python merge_images.py                          # mean blend, auto output name
  python merge_images.py --mode median            # median blend
  python merge_images.py --mode focus             # focus stacking
  python merge_images.py --mode mean --out result.jpg
  python merge_images.py --no-align               # skip alignment (images already aligned)
"""
 
import argparse
import sys
from pathlib import Path
 
import cv2
import numpy as np
 
 
# ──────────────────────────────────────────────────────────────
# Image discovery
# ──────────────────────────────────────────────────────────────
 
def find_jpegs(directory: Path) -> list[Path]:
    """Return sorted list of JPEG files in *directory*."""
    exts = {".jpg", ".jpeg"}
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    return files
 
 
# ──────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────
 
def align_to_reference(ref: np.ndarray, src: np.ndarray) -> np.ndarray:
    """
    Return *src* warped to match *ref* using ORB + homography.
    Falls back to returning *src* unchanged if alignment fails.
    """
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
 
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(src_gray, None)
 
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("    ⚠  Not enough features — skipping alignment for this image.")
        return src
 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
 
    # Keep best 70 % of matches, minimum 8
    n_good = max(8, int(len(matches) * 0.70))
    good = matches[:n_good]
 
    if len(good) < 4:
        print("    ⚠  Too few good matches — skipping alignment.")
        return src
 
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
 
    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        print("    ⚠  Homography failed — skipping alignment.")
        return src
 
    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(src, H, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
    return aligned
 
 
# ──────────────────────────────────────────────────────────────
# Merge modes
# ──────────────────────────────────────────────────────────────
 
def merge_mean(stack: list[np.ndarray]) -> np.ndarray:
    """Average all images pixel-wise."""
    acc = np.zeros_like(stack[0], dtype=np.float64)
    for img in stack:
        acc += img.astype(np.float64)
    return np.clip(acc / len(stack), 0, 255).astype(np.uint8)
 
 
def merge_median(stack: list[np.ndarray]) -> np.ndarray:
    """Median of all images pixel-wise (removes transient objects / noise)."""
    arr = np.stack([img.astype(np.float32) for img in stack], axis=0)
    return np.median(arr, axis=0).astype(np.uint8)
 
 
def merge_focus(stack: list[np.ndarray]) -> np.ndarray:
    """
    Focus stacking: per-pixel, keep the value from the image with the
    highest local Laplacian variance (= sharpest detail).
    """
    # Compute sharpness maps
    sharpness = []
    for img in stack:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        # Local variance via box filter
        lap_sq = lap ** 2
        local_var = cv2.boxFilter(lap_sq, cv2.CV_64F, (15, 15))
        sharpness.append(local_var)
 
    sharpness_arr = np.stack(sharpness, axis=0)          # (N, H, W)
    best_idx = np.argmax(sharpness_arr, axis=0)          # (H, W)
 
    h, w, c = stack[0].shape
    result = np.zeros((h, w, c), dtype=np.uint8)
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
# Main
# ──────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Layer and merge JPG images of the same subject."
    )
    parser.add_argument(
        "--mode", choices=list(MERGE_MODES), default="mean",
        help="Merge mode: mean (default), median, or focus (focus stacking)."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output filename (default: merged_<mode>.jpg)."
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="Skip alignment step (use when images are already aligned)."
    )
    parser.add_argument(
        "--dir", default=None,
        help="Directory containing images (default: directory of this script)."
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality 1-100 (default: 95)."
    )
    args = parser.parse_args()
 
    # Resolve directory
    base_dir = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent
    out_name = args.out or f"merged_{args.mode}.jpg"
    out_path = base_dir / out_name
 
    print(f"📂  Scanning: {base_dir}")
    files = find_jpegs(base_dir)
 
    # Exclude the output file itself from input list
    files = [f for f in files if f.name != out_path.name]
 
    if len(files) < 2:
        sys.exit(f"❌  Need at least 2 JPEG images, found {len(files)}.")
 
    print(f"🖼   Found {len(files)} images: {[f.name for f in files]}")
    print(f"⚙️   Mode: {args.mode}  |  Align: {not args.no_align}")
 
    # Load reference
    ref = cv2.imread(str(files[0]))
    if ref is None:
        sys.exit(f"❌  Could not read {files[0]}")
    print(f"\n  [1/{len(files)}] Reference: {files[0].name}  {ref.shape[1]}×{ref.shape[0]}")
 
    stack = [ref]
 
    for i, fpath in enumerate(files[1:], start=2):
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"    ⚠  Could not read {fpath.name} — skipping.")
            continue
 
        # Resize to reference size if different
        if img.shape[:2] != ref.shape[:2]:
            print(f"    ↕  Resizing {fpath.name} from {img.shape[1]}×{img.shape[0]} "
                  f"→ {ref.shape[1]}×{ref.shape[0]}")
            img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)
 
        if not args.no_align:
            print(f"  [{i}/{len(files)}] Aligning: {fpath.name} … ", end="", flush=True)
            img = align_to_reference(ref, img)
            print("done")
        else:
            print(f"  [{i}/{len(files)}] Loading:  {fpath.name}")
 
        stack.append(img)
 
    print(f"\n🔀  Merging {len(stack)} images using '{args.mode}' …")
    merge_fn = MERGE_MODES[args.mode]
    result = merge_fn(stack)
 
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    print(f"✅  Saved → {out_path}")
 
 
if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        sys.exit("❌  OpenCV not found. Install it with:\n    pip install opencv-python numpy")
    main()