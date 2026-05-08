"""
pyramid.py — Multi-Scale Traffic Sign Detection via Gaussian Image Pyramids
============================================================================

SCALE-SPACE THEORY
------------------
Natural images contain structures at many different resolutions. A stop sign
50 m away occupies only ~8×8 pixels; the same sign at 5 m may span 80×80
pixels.  A single fixed-resolution detector tuned for the near sign misses
the far one entirely.

Witkin (1983) and Koenderink (1984) formalised *scale-space*: repeatedly
convolve an image with a Gaussian kernel (σ ↑) so that fine detail is
progressively removed, leaving only structures that survive at coarser
resolutions.  Subsampling the blurred image by a factor s produces a
*Gaussian pyramid* — each level is a band-limited approximation of the
original at 1/s the linear size.

Detection at every pyramid level is therefore equivalent to sweeping a
detector window over many effective sizes in the original image, giving
robust multi-scale coverage at a fraction of the cost of literally resizing
the template for every possible scale.

COORDINATE TRANSFORMATION
--------------------------
When a bounding box (x, y, w, h) is found at pyramid level L (scale factor s^L):

    x_orig = x_L · s^L
    y_orig = y_L · s^L
    w_orig = w_L · s^L
    h_orig = h_L · s^L

Because each level is 1/s of the previous in each dimension, a pixel at
(x_L, y_L) in level L maps back to (x_L · s^L, y_L · s^L) in the
original image.

ACCURACY vs SPEED TRADE-OFF  (see `image_pyramid` docstring)
-------------------------------------------------------------
More pyramid levels → finer scale coverage → higher recall
Fewer levels        → faster inference   → lower recall but faster pipeline
`min_size` and `scale` together control this curve without changing the
detector itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases (mirrors harris.py conventions)
# ---------------------------------------------------------------------------
ColorImage = np.ndarray   # HxWx3 uint8, BGR
GrayImage  = np.ndarray   # HxW   uint8 or float32
BBoxArray  = np.ndarray   # Nx4   float32  [x, y, w, h]  (origin = top-left)
ScoreArray = np.ndarray   # N,    float32

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single bounding-box detection with a confidence score and scale info."""
    x:      float   # left edge in the *original* image
    y:      float   # top  edge in the *original* image
    w:      float   # width  in the *original* image
    h:      float   # height in the *original* image
    score:  float   # detector confidence (higher = better)
    level:  int     # pyramid level this detection came from
    scale:  float   # cumulative scale factor  (original → level)

    # Convenience ─────────────────────────────────────────────────────────
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_int_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y),
                int(self.x + self.w), int(self.y + self.h))


@dataclass
class PyramidLevel:
    """One level of the Gaussian pyramid together with its metadata."""
    image:        np.ndarray    # the downsampled image (BGR or gray)
    level:        int           # 0 = original
    scale_factor: float         # cumulative: original_size * scale_factor = this size
    # The *inverse* of scale_factor maps coordinates back to the original image.


# ---------------------------------------------------------------------------
# Pyramid construction
# ---------------------------------------------------------------------------

def build_gaussian_pyramid(
    image:      ColorImage | GrayImage,
    scale:      float = 1.5,
    min_size:   int   = 32,
    max_levels: Optional[int] = None,
    blur_sigma: float = 0.0,
) -> List[PyramidLevel]:
    """
    Build a Gaussian image pyramid.

    Each level is produced by:
      1. Optionally applying a Gaussian blur (anti-aliasing before decimation).
      2. Downsampling by ``scale`` using INTER_AREA interpolation (best for
         shrinking) so that no aliasing aliases are introduced.

    Parameters
    ----------
    image       : Input image (colour or greyscale).
    scale       : Reduction factor per level (must be > 1).  Typical: 1.2–2.0.
                  - Smaller values (e.g. 1.2) give denser scale coverage but
                    more levels → slower.
                  - Larger values (e.g. 2.0) give coarser coverage → faster.
    min_size    : Stop adding levels when either dimension falls below this.
    max_levels  : Hard cap on the number of levels (None = no cap).
    blur_sigma  : If > 0, a Gaussian with this σ is applied before each
                  downsampling step.  0.0 lets OpenCV's INTER_AREA handle
                  anti-aliasing, which is sufficient for most cases.

    Returns
    -------
    List of PyramidLevel objects, index 0 being the *original* image.

    Accuracy vs Speed
    -----------------
    The ``min_size`` / ``scale`` pair is the primary knob:
      - Decrease ``scale`` + decrease ``min_size``: finer coverage, more
        compute — use for high-accuracy offline analysis.
      - Increase ``scale`` + increase ``min_size``: fewer levels, faster
        inference — use for real-time or embedded scenarios.
    The caller never needs to change the detector; only pyramid parameters
    change.
    """
    if scale <= 1.0:
        raise ValueError(f"scale must be > 1.0, got {scale}")

    levels: List[PyramidLevel] = []
    current = image.copy()
    cumulative_scale = 1.0   # starts at 1.0 (= original size)

    for lvl in range(max_levels if max_levels is not None else 999):
        h, w = current.shape[:2]

        levels.append(PyramidLevel(
            image=current.copy(),
            level=lvl,
            scale_factor=cumulative_scale,
        ))
        logger.debug("Pyramid level %d  size=(%d×%d)  cum_scale=%.4f",
                     lvl, w, h, cumulative_scale)

        # Compute next dimensions
        new_w = int(round(w / scale))
        new_h = int(round(h / scale))

        if new_w < min_size or new_h < min_size:
            logger.debug("Stopping: next size (%d×%d) < min_size %d",
                         new_w, new_h, min_size)
            break

        # Optional explicit blur before downsampling
        blurred = current
        if blur_sigma > 0:
            ksize = _sigma_to_ksize(blur_sigma)
            blurred = cv2.GaussianBlur(current, (ksize, ksize), blur_sigma)

        current = cv2.resize(blurred, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        cumulative_scale /= scale   # pixel → original coordinate multiplier = 1/cumulative

    return levels


def image_pyramid(
    image: ColorImage | GrayImage,
    scale: float = 1.5,
    min_size: int = 32,
) -> Iterator[Tuple[np.ndarray, float]]:
    """
    Generator that yields ``(scaled_image, cumulative_scale_factor)`` for
    each pyramid level, starting from the original.

    This is the lightweight iterator interface; use ``build_gaussian_pyramid``
    when you need richer metadata.

    Accuracy vs Speed trade-off
    ---------------------------
    Every additional pyramid level doubles the total pixels processed
    (for scale=√2).  Halving ``min_size`` roughly doubles the number of
    levels and thus the total work.  For a 640×480 input:

      scale=1.5, min_size=64  →  ~5 levels, ~1.5× slower than single-scale
      scale=2.0, min_size=64  →  ~4 levels, ~1.3× slower
      scale=1.2, min_size=32  →  ~12 levels, ~3× slower

    Use coarser scale / larger min_size for real-time pipelines, finer
    values for offline precision.
    """
    levels = build_gaussian_pyramid(image, scale=scale, min_size=min_size)
    for lvl in levels:
        yield lvl.image, lvl.scale_factor


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def map_boxes_to_original(
    boxes:        BBoxArray,
    scale_factor: float,
) -> BBoxArray:
    """
    Map bounding boxes from a pyramid level back to original-image coordinates.

    At pyramid level L with cumulative scale factor ``s`` (< 1.0 for levels
    below the original, because each step *divides* by the per-level scale),
    a pixel (x_L, y_L) corresponds to:

        x_orig = x_L / s      (equivalently: x_L * (1/s))
        y_orig = y_L / s

    Width and height scale identically.

    Parameters
    ----------
    boxes        : Array of shape (N, 4) with columns [x, y, w, h].
    scale_factor : The cumulative scale factor stored in ``PyramidLevel``
                   (1.0 for level-0, 1/scale for level-1, etc.).

    Returns
    -------
    Array of shape (N, 4) in original-image coordinates.

    Maths note
    ----------
    ``scale_factor`` = product of (1/per_level_scale) applied so far.
    Dividing by ``scale_factor`` reverses all the downsampling steps.
    """
    if len(boxes) == 0:
        return boxes
    return (boxes.astype(np.float32) / scale_factor)


# ---------------------------------------------------------------------------
# Multi-scale detection
# ---------------------------------------------------------------------------

DetectorFn = Callable[[np.ndarray], Tuple[BBoxArray, ScoreArray]]
"""
A detector function signature expected by ``detect_multiscale``.

Parameters
----------
image  : numpy array (same colour space as the input to ``detect_multiscale``)

Returns
-------
boxes  : np.ndarray of shape (N, 4)  [x, y, w, h]
scores : np.ndarray of shape (N,)    confidence for each box
"""


def _placeholder_detector(
    image: np.ndarray,
) -> Tuple[BBoxArray, ScoreArray]:
    """
    Placeholder detector — replace with your trained model.

    Current behaviour: returns zero detections so the pipeline can be
    exercised end-to-end without a real model.
    """
    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)


def detect_multiscale(
    image:      ColorImage | GrayImage,
    detector:   DetectorFn = _placeholder_detector,
    scale:      float = 1.5,
    min_size:   int   = 32,
    nms_iou_threshold: float = 0.4,
    score_threshold:   float = 0.3,
) -> List[Detection]:
    """
    Run ``detector`` at every scale of the Gaussian pyramid and return
    de-duplicated detections in *original-image* coordinates.

    Pipeline
    --------
    1. Build the Gaussian pyramid from ``image``.
    2. For each level, call ``detector(level_image)`` → raw boxes + scores.
    3. Filter detections below ``score_threshold``.
    4. Map surviving boxes back to original coordinates using the level's
       cumulative scale factor.
    5. Pool all detections across levels.
    6. Apply NMS to suppress duplicates that arise when the same physical
       sign is detected at multiple scales.

    Parameters
    ----------
    image              : Input image (BGR or greyscale).
    detector           : Any callable matching ``DetectorFn``.  Plug in your
                         YOLO / SSD / template-matching function here.
    scale              : Per-level downsampling factor (see ``build_gaussian_pyramid``).
    min_size           : Minimum image dimension before stopping (see above).
    nms_iou_threshold  : IoU threshold for Non-Maximum Suppression.
                         Lower → more aggressive suppression.
    score_threshold    : Detections with score < this are discarded before NMS.

    Returns
    -------
    List of ``Detection`` objects sorted by descending score.
    """
    pyramid = build_gaussian_pyramid(image, scale=scale, min_size=min_size)
    all_detections: List[Detection] = []

    for lvl in pyramid:
        raw_boxes, raw_scores = detector(lvl.image)

        if len(raw_boxes) == 0:
            continue

        # Filter by score threshold
        keep = raw_scores >= score_threshold
        raw_boxes  = raw_boxes[keep]
        raw_scores = raw_scores[keep]

        if len(raw_boxes) == 0:
            continue

        # Map to original image coordinates
        orig_boxes = map_boxes_to_original(raw_boxes, lvl.scale_factor)

        for box, score in zip(orig_boxes, raw_scores):
            all_detections.append(Detection(
                x=float(box[0]),
                y=float(box[1]),
                w=float(box[2]),
                h=float(box[3]),
                score=float(score),
                level=lvl.level,
                scale=lvl.scale_factor,
            ))

        logger.debug("Level %d: %d raw → %d kept detections",
                     lvl.level, int(keep.size), int(keep.sum()))

    if not all_detections:
        return []

    # Non-Maximum Suppression across all scales
    all_detections = apply_nms(all_detections, iou_threshold=nms_iou_threshold)

    # Sort by descending score
    all_detections.sort(key=lambda d: d.score, reverse=True)
    return all_detections


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def compute_iou(a: Detection, b: Detection) -> float:
    """Compute Intersection-over-Union for two Detection objects."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a.w * a.h
    area_b = b.w * b.h
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def apply_nms(
    detections:    List[Detection],
    iou_threshold: float = 0.4,
) -> List[Detection]:
    """
    Greedy Non-Maximum Suppression.

    Algorithm
    ---------
    1. Sort detections by descending score.
    2. Greedily pick the highest-scoring box.
    3. Suppress all remaining boxes whose IoU with the picked box exceeds
       ``iou_threshold``.
    4. Repeat until no boxes remain.

    This removes duplicates that arise naturally when the same sign is
    detected at multiple pyramid scales (it appears at different sizes
    but maps to overlapping regions in the original image).

    Parameters
    ----------
    detections    : List of Detection objects (any order).
    iou_threshold : Boxes with IoU ≥ this value are suppressed.

    Returns
    -------
    Subset of ``detections`` after suppression.
    """
    if not detections:
        return []

    # Sort by score descending
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    suppressed = [False] * len(sorted_dets)

    for i, det in enumerate(sorted_dets):
        if suppressed[i]:
            continue
        kept.append(det)
        for j in range(i + 1, len(sorted_dets)):
            if suppressed[j]:
                continue
            if compute_iou(det, sorted_dets[j]) >= iou_threshold:
                suppressed[j] = True

    return kept


# ---------------------------------------------------------------------------
# OpenCV-accelerated NMS (faster for large detection sets)
# ---------------------------------------------------------------------------

def apply_nms_cv(
    detections:    List[Detection],
    iou_threshold: float = 0.4,
) -> List[Detection]:
    """
    NMS using ``cv2.dnn.NMSBoxes`` — faster than the pure-Python version
    for large detection sets (> ~200 boxes).

    Returns
    -------
    Subset of ``detections`` after suppression, sorted by descending score.
    """
    if not detections:
        return []

    boxes  = [[d.x, d.y, d.w, d.h] for d in detections]
    scores = [d.score for d in detections]

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.0,      # pre-filtered upstream
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    # cv2 returns shape (N,1) in older versions, (N,) in newer
    indices = np.array(indices).flatten().tolist()
    kept = [detections[i] for i in indices]
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_detections(
    image:      ColorImage,
    detections: List[Detection],
    color:      Tuple[int, int, int] = (0, 200, 0),
    thickness:  int = 2,
    show_level: bool = True,
) -> ColorImage:
    """
    Draw bounding boxes and optional scale-level labels on *image*.

    Parameters
    ----------
    image      : Original (full-resolution) BGR image.  Not modified.
    detections : List of Detection objects in original-image coordinates.
    color      : BGR colour for the box outline.
    thickness  : Line thickness in pixels.
    show_level : If True, annotate each box with its pyramid level and score.

    Returns
    -------
    A copy of ``image`` with detections drawn.
    """
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.to_int_xyxy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if show_level:
            label = f"L{det.level} s={det.score:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _sigma_to_ksize(sigma: float) -> int:
    """Convert a Gaussian σ to the nearest valid (odd) kernel size."""
    k = int(2.0 * 3.0 * sigma + 1.0)
    return k if k % 2 == 1 else k + 1


# ---------------------------------------------------------------------------
# Public API summary
# ---------------------------------------------------------------------------
__all__ = [
    # Data types
    "Detection",
    "PyramidLevel",
    # Pyramid construction
    "build_gaussian_pyramid",
    "image_pyramid",
    # Coordinate mapping
    "map_boxes_to_original",
    # Detection pipeline
    "detect_multiscale",
    # NMS
    "apply_nms",
    "apply_nms_cv",
    "compute_iou",
    # Visualisation
    "draw_detections",
    # Placeholder (swap with real model)
    "_placeholder_detector",
]


# ---------------------------------------------------------------------------
# Quick smoke-test  (python pyramid.py --image path/to/sign.jpg)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(description="Pyramid smoke-test")
    p.add_argument("--image", required=True)
    p.add_argument("--scale", type=float, default=1.5)
    p.add_argument("--min-size", type=int, default=32)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"File not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"cv2.imread failed: {img_path}")

    print(f"Input: {img_path.name}  {img.shape[1]}×{img.shape[0]}")
    pyramid = build_gaussian_pyramid(img, scale=args.scale,
                                     min_size=args.min_size)
    for lvl in pyramid:
        h, w = lvl.image.shape[:2]
        print(f"  Level {lvl.level:2d}  {w:4d}×{h:4d}  "
              f"scale_factor={lvl.scale_factor:.5f}  "
              f"(≈ {1/lvl.scale_factor:.2f}× original)")

    # Run pipeline with placeholder detector (zero detections expected)
    dets = detect_multiscale(img, scale=args.scale, min_size=args.min_size)
    print(f"\ndetect_multiscale returned {len(dets)} detection(s).")
    print("Swap _placeholder_detector with your model to see real results.")

    if args.output:
        vis = draw_detections(img, dets)
        cv2.imwrite(args.output, vis)
        print(f"Result saved → {args.output}")
