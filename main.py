"""
main.py — Stage 6: Integration & End-to-End Pipeline
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

from Preprocessing import full_pipeline as preprocess_np
from harris import detect_harris_corners, draw_corners
from pyramid import build_gaussian_pyramid
from sift_matching import sift_matching
from segmentation import segment_traffic_signs
from classification import TrafficSignClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(
    image_bgr: np.ndarray,
    classifier: Optional[TrafficSignClassifier] = None,
    class_names: Optional[list] = None,
    reference_img: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    image_name: str = "image",
    verbose: bool = True,
) -> dict:
    """Run the complete pipeline on a single BGR image."""
    results = {"image_name": image_name, "timing": {}}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _save(tag, img):
        if output_dir:
            p = os.path.join(output_dir, f"{image_name}_{tag}.jpg")
            cv2.imwrite(p, img)

    if verbose:
        log.info("══ [%s] Pipeline starting ══", image_name)
    _save("0_original", image_bgr)

    # STAGE 1 : Preprocessing
    t0 = time.perf_counter()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preprocessed_float = preprocess_np(image_rgb, size=(128, 128),
                                       normalize_method="minmax",
                                       equalize=True, denoise=True)
    preprocessed_uint8 = (preprocessed_float * 255).astype(np.uint8)
    preprocessed_bgr = cv2.cvtColor(preprocessed_uint8, cv2.COLOR_RGB2BGR)
    results["timing"]["preprocessing"] = time.perf_counter() - t0
    results["preprocessed"] = preprocessed_bgr
    _save("1_preprocessed", preprocessed_bgr)
    if verbose:
        log.info("  [1] Preprocessing done (%.3fs)", results["timing"]["preprocessing"])

    # STAGE 2 : Harris Corner Detection
    t0 = time.perf_counter()
    corners, R_nms = detect_harris_corners(preprocessed_bgr)
    harris_vis = draw_corners(preprocessed_bgr, corners)
    results["timing"]["harris"] = time.perf_counter() - t0
    results["corners"] = corners
    results["R_nms"] = R_nms
    _save("2_harris", harris_vis)
    if verbose:
        log.info("  [2] Harris: %d corners (%.3fs)", len(corners), results["timing"]["harris"])

    # STAGE 3 : Gaussian Pyramid
    t0 = time.perf_counter()
    pyramid_levels = build_gaussian_pyramid(preprocessed_bgr, scale=1.5, min_size=32)
    tiles = [cv2.resize(lvl.image, (128, 128)) for lvl in pyramid_levels[:4]]
    pyramid_vis = np.hstack(tiles) if tiles else preprocessed_bgr
    results["timing"]["pyramid"] = time.perf_counter() - t0
    results["pyramid_levels"] = len(pyramid_levels)
    _save("3_pyramid", pyramid_vis)
    if verbose:
        log.info("  [3] Pyramid: %d levels (%.3fs)", len(pyramid_levels), results["timing"]["pyramid"])

    # STAGE 4 : SIFT Matching
    t0 = time.perf_counter()
    gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
    if reference_img is not None:
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        sift_vis, n_good, n_total = sift_matching(gray, ref_gray)
    else:
        sift_vis, n_good, n_total = sift_matching(gray, gray)
    results["timing"]["sift"] = time.perf_counter() - t0
    results["sift_good"] = n_good
    results["sift_total"] = n_total
    _save("4_sift", sift_vis)
    if verbose:
        log.info("  [4] SIFT: %d/%d good matches (%.3fs)", n_good, n_total, results["timing"]["sift"])

    # STAGE 5 : Segmentation
    t0 = time.perf_counter()
    seg_vis, n_boxes, mask = segment_traffic_signs(image_bgr)
    results["timing"]["segmentation"] = time.perf_counter() - t0
    results["seg_boxes"] = n_boxes
    results["seg_mask"] = mask
    _save("5_segmentation", seg_vis)
    _save("5_mask", mask)
    if verbose:
        log.info("  [5] Segmentation: %d boxes (%.3fs)", n_boxes, results["timing"]["segmentation"])

    # STAGE 6 : Classification
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
    results["timing"]["classification"] = time.perf_counter() - t0
    results["pred_label"] = pred_label
    results["pred_conf"] = pred_conf
    if verbose and classifier is not None:
        log.info("  [6] Class: %s conf=%.3f (%.3fs)", pred_label, pred_conf, results["timing"]["classification"])

    # FIXED: Summary overlay with VERY SMALL text
    summary = seg_vis.copy()
    
    # Create semi-transparent dark panel for text (top-left corner)
    panel_h = 130   # Even smaller panel
    panel_w = 280   # Narrower panel
    overlay = summary.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (0, 0, 0), -1)
    summary = cv2.addWeighted(overlay, 0.6, summary, 0.4, 0)
    
    # VERY SMALL FONT
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35  # Much smaller (was 0.45, now 0.35)
    thickness = 1
    line_height = 16   # Tighter spacing
    start_y = 18       # Start higher up
    
    # Truncate long class names aggressively (keep first 20 chars)
    if len(pred_label) > 20:
        display_class = pred_label[:17] + "..."
    else:
        display_class = pred_label
    
    # Short labels for better fit
    sift_text = f"SIFT: {n_good}/{n_total}"
    conf_text = f"Conf: {pred_conf:.2f}"
    
    text_lines = [
        f"CORN: {len(corners)}",
        f"PYR: {len(pyramid_levels)}",
        sift_text,
        f"BOX: {n_boxes}",
        display_class,
        conf_text
    ]
    
    for i, line in enumerate(text_lines):
        y = start_y + i * line_height
        # Black outline for readability
        cv2.putText(summary, line, (11, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Yellow text
        cv2.putText(summary, line, (10, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    
    _save("6_summary", summary)

    results["timing"]["total"] = sum(results["timing"].values())
    if verbose:
        log.info("  Total: %.3fs", results["timing"]["total"])
    return results


def make_demo_image() -> np.ndarray:
    """Create a synthetic stop-sign-like image for testing."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 180
    cv2.circle(img, (100, 100), 70, (0, 0, 200), -1)
    cv2.circle(img, (100, 100), 70, (255, 255, 255), 5)
    cv2.putText(img, "STOP", (55, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return img


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection Pipeline")
    parser.add_argument("--image", default=None, help="Single image file")
    parser.add_argument("--image_dir", default=None, help="Directory of images")
    parser.add_argument("--reference", default=None, help="Reference image for SIFT")
    parser.add_argument("--output_dir", default="pipeline_output", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic demo")
    args = parser.parse_args()

    if args.demo:
        log.info("Demo mode: using synthetic image")
        images = [("demo", make_demo_image())]
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Cannot read: {args.image}")
        images = [(Path(args.image).stem, img)]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
        images = []
        data_path = Path(args.image_dir)
        
        # Check for subdirectories (class folders)
        has_subdirs = False
        for item in data_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                has_subdirs = True
                for img_file in item.iterdir():
                    if img_file.suffix.lower() in exts:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            images.append((f"{item.name}_{img_file.stem}", img))
        
        if not has_subdirs:
            # Flat structure
            for img_file in data_path.iterdir():
                if img_file.suffix.lower() in exts:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        images.append((img_file.stem, img))
        
        if not images:
            sys.exit(f"No images found in {args.image_dir}")
    else:
        sys.exit("Please provide --image, --image_dir, or --demo")

    reference_img = cv2.imread(args.reference) if args.reference else None
    os.makedirs(args.output_dir, exist_ok=True)

    for name, img in images:
        run_pipeline(img, reference_img=reference_img, output_dir=args.output_dir, image_name=name)


if __name__ == "__main__":
    main()
