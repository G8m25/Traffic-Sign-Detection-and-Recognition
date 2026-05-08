"""
main.py — Stage 6: Integration & End-to-End Pipeline
=====================================================
Traffic Sign Detection & Recognition — GTSRB Project

Connects all modules in order:
  1. Preprocessing  (Preprocessing.py)
  2. Harris Corners (harris.py)
  3. Pyramid        (pyramid.py)
  4. SIFT Matching  (sift_matching.py)
  5. Segmentation   (segmentation.py)
  6. Classification (classification.py)

Usage
-----
    python main.py --demo                          # synthetic test, no dataset needed
    python main.py --image path/to/sign.jpg        # single image
    python main.py --image_dir data/test/          # whole folder
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Import all team members' modules ──────────────────────────────────────
from Preprocessing  import full_pipeline as preprocess_np
from harris         import detect_harris_corners, draw_corners
from pyramid        import build_gaussian_pyramid
from sift_matching  import sift_matching
from segmentation   import segment_traffic_signs
from classification import TrafficSignClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  CORE PIPELINE  — runs all 6 stages on ONE image
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_bgr:    np.ndarray,
    classifier:   Optional[TrafficSignClassifier] = None,
    class_names:  Optional[list] = None,
    reference_img: Optional[np.ndarray] = None,
    output_dir:   Optional[str] = None,
    image_name:   str = "image",
    verbose:      bool = True,
) -> dict:
    """
    Run the complete pipeline on a single BGR image.
    Returns a dict of all results so evaluation.py can collect metrics.
    """
    results = {"image_name": image_name, "timing": {}}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _save(tag, img):
        """Helper: save an image to output_dir if set."""
        if output_dir:
            p = os.path.join(output_dir, f"{image_name}_{tag}.jpg")
            cv2.imwrite(p, img)
            log.info("  Saved → %s", p)

    log.info("══ [%s] Pipeline starting ══", image_name)
    _save("0_original", image_bgr)

    # ── STAGE 1 : Preprocessing ───────────────────────────────────────
    t0 = time.perf_counter()
    image_rgb          = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preprocessed_float = preprocess_np(image_rgb, size=(128, 128),
                                       normalize_method="minmax",
                                       equalize=True, denoise=True)
    preprocessed_uint8 = (preprocessed_float * 255).astype(np.uint8)
    preprocessed_bgr   = cv2.cvtColor(preprocessed_uint8, cv2.COLOR_RGB2BGR)
    results["timing"]["preprocessing"] = time.perf_counter() - t0
    results["preprocessed"] = preprocessed_bgr
    _save("1_preprocessed", preprocessed_bgr)
    if verbose:
        log.info("  [1] Preprocessing done  (%.3fs)", results["timing"]["preprocessing"])

    # ── STAGE 2 : Harris Corner Detection ────────────────────────────
    t0 = time.perf_counter()
    corners, R_nms = detect_harris_corners(preprocessed_bgr)
    harris_vis     = draw_corners(preprocessed_bgr, corners)
    results["timing"]["harris"] = time.perf_counter() - t0
    results["corners"] = corners
    results["R_nms"]   = R_nms
    _save("2_harris", harris_vis)
    if verbose:
        log.info("  [2] Harris: %d corners  (%.3fs)", len(corners), results["timing"]["harris"])

    # ── STAGE 3 : Gaussian Pyramid ───────────────────────────────────
    t0 = time.perf_counter()
    pyramid_levels = build_gaussian_pyramid(preprocessed_bgr, scale=1.5, min_size=32)
    # Tile the first 4 levels side-by-side for visualisation
    tiles = [cv2.resize(lvl.image, (128, 128)) for lvl in pyramid_levels[:4]]
    pyramid_vis = np.hstack(tiles) if tiles else preprocessed_bgr
    results["timing"]["pyramid"]      = time.perf_counter() - t0
    results["pyramid_levels"]         = len(pyramid_levels)
    _save("3_pyramid", pyramid_vis)
    if verbose:
        log.info("  [3] Pyramid: %d levels  (%.3fs)", len(pyramid_levels), results["timing"]["pyramid"])

    # ── STAGE 4 : SIFT Matching ───────────────────────────────────────
    t0 = time.perf_counter()
    gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
    if reference_img is not None:
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        sift_vis, n_good, n_total = sift_matching(gray, ref_gray)
    else:
        # Self-match when no reference is provided
        sift_vis, n_good, n_total = sift_matching(gray, gray)
    results["timing"]["sift"]        = time.perf_counter() - t0
    results["sift_good"]             = n_good
    results["sift_total"]            = n_total
    _save("4_sift", sift_vis)
    if verbose:
        log.info("  [4] SIFT: %d/%d good matches  (%.3fs)", n_good, n_total, results["timing"]["sift"])

    # ── STAGE 5 : Segmentation ────────────────────────────────────────
    t0 = time.perf_counter()
    seg_vis, n_boxes, mask = segment_traffic_signs(image_bgr)
    results["timing"]["segmentation"] = time.perf_counter() - t0
    results["seg_boxes"] = n_boxes
    results["seg_mask"]  = mask
    _save("5_segmentation", seg_vis)
    _save("5_mask", mask)
    if verbose:
        log.info("  [5] Segmentation: %d boxes  (%.3fs)", n_boxes, results["timing"]["segmentation"])

    # ── STAGE 6 : Classification ──────────────────────────────────────
    t0 = time.perf_counter()
    pred_label, pred_conf = "N/A", 0.0
    if classifier is not None and classifier.is_trained:
        try:
            preds, confs = classifier.predict([preprocessed_uint8])
            idx = int(preds[0])
            pred_conf = float(confs[0])
            pred_label = class_names[idx] if (class_names and idx < len(class_names)) else str(idx)
        except Exception as e:
            log.warning("  [6] Classification error: %s", e)
    else:
        log.info("  [6] No trained classifier — skipping.")
    results["timing"]["classification"] = time.perf_counter() - t0
    results["pred_label"] = pred_label
    results["pred_conf"]  = pred_conf
    if verbose:
        log.info("  [6] Class: %s  conf=%.3f  (%.3fs)",
                 pred_label, pred_conf, results["timing"]["classification"])

    # ── Summary overlay image ─────────────────────────────────────────
    summary = seg_vis.copy()
    for i, line in enumerate([
        f"Corners : {len(corners)}",
        f"Pyr lvls: {len(pyramid_levels)}",
        f"SIFT    : {n_good} good matches",
        f"Boxes   : {n_boxes}",
        f"Class   : {pred_label} ({pred_conf:.2f})",
    ]):
        cv2.putText(summary, line, (8, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    _save("6_summary", summary)

    results["timing"]["total"] = sum(results["timing"].values())
    log.info("  Total: %.3fs", results["timing"]["total"])
    return results


# ══════════════════════════════════════════════════════════════════════
#  DEMO IMAGE  (synthetic — no dataset needed)
# ══════════════════════════════════════════════════════════════════════

def make_demo_image() -> np.ndarray:
    """Create a simple red-circle stop-sign-like image for testing."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 180
    cv2.circle(img, (100, 100), 70, (0, 0, 200), -1)
    cv2.circle(img, (100, 100), 70, (255, 255, 255), 5)
    cv2.putText(img, "STOP", (55, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return img


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Traffic Sign Detection & Recognition — Integrated Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",      default=None, help="Path to a single input image.")
    p.add_argument("--image_dir",  default=None, help="Path to a folder of images.")
    p.add_argument("--reference",  default=None, help="Reference image for SIFT matching.")
    p.add_argument("--output_dir", default="pipeline_output", help="Folder to save result images.")
    p.add_argument("--demo",       action="store_true", help="Run on a synthetic demo image.")
    p.add_argument("--no_clf",     action="store_true", help="Skip classification.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Choose image source
    if args.demo:
        log.info("Demo mode: using synthetic image.")
        images = [("demo", make_demo_image())]
    elif args.image:
        path = Path(args.image)
        if not path.exists():
            sys.exit(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            sys.exit(f"Could not read image: {path}")
        images = [(path.stem, img)]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
        paths = sorted(p for p in Path(args.image_dir).iterdir()
                       if p.suffix.lower() in exts)
        images = [(p.stem, cv2.imread(str(p))) for p in paths]
        images = [(n, i) for n, i in images if i is not None]
        if not images:
            sys.exit(f"No images found in {args.image_dir}")
    else:
        sys.exit("Provide --image, --image_dir, or --demo.")

    reference_img = cv2.imread(args.reference) if args.reference else None

    # Run pipeline on each image
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    for name, img in images:
        r = run_pipeline(img,
                         reference_img=reference_img,
                         output_dir=args.output_dir,
                         image_name=name,
                         verbose=True)
        all_results.append(r)

    # Print timing summary
    print("\n" + "═" * 50)
    print("  TIMING SUMMARY")
    print("═" * 50)
    for stage in ["preprocessing", "harris", "pyramid", "sift", "segmentation", "classification"]:
        times = [r["timing"].get(stage, 0) for r in all_results]
        print(f"  {stage:<22}  {sum(times)/len(times)*1000:7.1f} ms avg")
    totals = [r["timing"].get("total", 0) for r in all_results]
    print(f"  {'TOTAL':<22}  {sum(totals)/len(totals)*1000:7.1f} ms avg")
    print("═" * 50)
    print(f"\n  Output saved to: {os.path.abspath(args.output_dir)}/")
    print("  Run  python evaluation.py --demo  for full metrics.\n")


if __name__ == "__main__":
    main()
