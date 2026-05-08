"""
evaluation.py — Stage 6: Quantitative Evaluation
=================================================
Traffic Sign Detection & Recognition — GTSRB Project

Computes ALL metrics required by the project rubric:
  • Classification : Accuracy, Precision, Recall, F1, Confusion Matrix
  • Segmentation   : IoU (Intersection over Union)
  • SIFT Matching  : Match accuracy (good / total ratio)
  • Harris         : Corner count statistics
  • Timing         : Per-stage latency

Usage
-----
    python evaluation.py --demo                         # no dataset needed
    python evaluation.py --image_dir data/test/ --labels data/labels.csv
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # works without a display / monitor
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)

from main           import run_pipeline, make_demo_image
from classification import TrafficSignClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════

def pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Pixel-level Intersection over Union between two binary masks.
    Both masks should be uint8 (0/255) or bool arrays.
    IoU = intersection / union   →   range [0, 1]
    """
    pred = (pred_mask > 0)
    gt   = (gt_mask   > 0)
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or (pred, gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def sift_accuracy(n_good: int, n_total: int) -> float:
    """Ratio of Lowe-ratio-passing matches to total matches."""
    return n_good / n_total if n_total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def evaluate(
    images:       list,           # [(name, bgr_ndarray), ...]
    true_labels:  list,           # integer class ID per image
    class_names:  Optional[list] = None,
    output_dir:   Optional[str]  = None,
    gt_masks:     Optional[list] = None,  # ground-truth binary masks (optional)
    train_images: Optional[list] = None,  # images to train classifier on
    train_labels: Optional[list] = None,
) -> dict:
    """
    Run the full pipeline on every image and collect all metrics.
    Returns a dict of metric keys → values.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Train classifier ──────────────────────────────────────────────
    classifier = None
    if train_images and train_labels:
        log.info("Training classifier on %d images …", len(train_images))
        classifier = TrafficSignClassifier()
        train_metrics = classifier.train(train_images, train_labels)
        log.info("  Training split metrics: %s", train_metrics)
    else:
        log.warning("No training data provided — skipping classification metrics.")

    # ── Run pipeline on every test image ─────────────────────────────
    all_results = []
    for name, img in images:
        r = run_pipeline(
            img,
            classifier=classifier,
            class_names=class_names,
            output_dir=output_dir,
            image_name=name,
            verbose=False,
        )
        all_results.append(r)

    # ── 1. Classification metrics ─────────────────────────────────────
    clf_metrics = {}
    if classifier is not None:
        pred_labels = []
        for r in all_results:
            lbl = r["pred_label"]
            if class_names and lbl in class_names:
                pred_labels.append(class_names.index(lbl))
            else:
                try:
                    pred_labels.append(int(lbl))
                except (ValueError, TypeError):
                    pred_labels.append(-1)

        valid_pairs = [(t, p) for t, p in zip(true_labels, pred_labels) if p != -1]
        if valid_pairs:
            y_true, y_pred = zip(*valid_pairs)
            clf_metrics = {
                "accuracy" : accuracy_score (y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall"   : recall_score   (y_true, y_pred, average="weighted", zero_division=0),
                "f1"       : f1_score       (y_true, y_pred, average="weighted", zero_division=0),
            }

            # Confusion matrix
            cm  = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(max(6, len(set(y_true))),
                                            max(5, len(set(y_true)))))
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
                ax=ax, colorbar=True, xticks_rotation="vertical")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            if output_dir:
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path, dpi=120, bbox_inches="tight")
                log.info("  Confusion matrix → %s", cm_path)
            plt.close(fig)
        else:
            clf_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    else:
        clf_metrics = {"accuracy": "N/A", "precision": "N/A",
                       "recall": "N/A", "f1": "N/A"}

    # ── 2. Segmentation IoU ───────────────────────────────────────────
    iou_scores = []
    if gt_masks:
        for r, gt in zip(all_results, gt_masks):
            if gt is not None and r.get("seg_mask") is not None:
                iou_scores.append(pixel_iou(r["seg_mask"], gt))
    seg_iou = float(np.mean(iou_scores)) if iou_scores else "N/A (no GT masks provided)"

    # ── 3. SIFT matching accuracy ─────────────────────────────────────
    match_accs = [sift_accuracy(r["sift_good"], r["sift_total"]) for r in all_results]
    sift_avg   = float(np.mean(match_accs)) if match_accs else 0.0

    # ── 4. Harris corner statistics ───────────────────────────────────
    counts = [len(r["corners"]) for r in all_results]
    harris_stats = {
        "mean": float(np.mean(counts)),
        "std" : float(np.std (counts)),
        "min" : int  (np.min (counts)),
        "max" : int  (np.max (counts)),
    }

    # ── 5. Timing statistics ──────────────────────────────────────────
    stages = ["preprocessing", "harris", "pyramid",
              "sift", "segmentation", "classification", "total"]
    timing = {}
    for s in stages:
        vals = [r["timing"].get(s, 0) for r in all_results]
        timing[s] = {"mean_ms": float(np.mean(vals)) * 1000,
                     "std_ms" : float(np.std (vals)) * 1000}

    # ── 6. Segmentation box stats ─────────────────────────────────────
    boxes = [r["seg_boxes"] for r in all_results]
    seg_stats = {
        "mean_boxes" : float(np.mean(boxes)),
        "total_boxes": int  (sum  (boxes)),
    }

    metrics = {
        "n_images"        : len(images),
        "classification"  : clf_metrics,
        "segmentation_iou": seg_iou,
        "seg_stats"       : seg_stats,
        "sift_match_acc"  : sift_avg,
        "harris_stats"    : harris_stats,
        "timing_ms"       : timing,
    }

    # Save report
    if output_dir:
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        _write_report(metrics, report_path)

    return metrics


# ══════════════════════════════════════════════════════════════════════
#  REPORT WRITER
# ══════════════════════════════════════════════════════════════════════

def _write_report(metrics: dict, save_path: str = None):
    sep  = "═" * 58
    dash = "─" * 58
    lines = [
        sep,
        "  TRAFFIC SIGN PIPELINE — EVALUATION REPORT",
        sep,
        f"  Images evaluated : {metrics['n_images']}",
        "",
        "  1. CLASSIFICATION METRICS",
        dash,
    ]
    for k, v in metrics["classification"].items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"    {k:<14} {val}")

    lines += [
        "",
        "  2. SEGMENTATION",
        dash,
    ]
    iou = metrics["segmentation_iou"]
    lines.append(f"    IoU            : {f'{iou:.4f}' if isinstance(iou, float) else iou}")
    lines.append(f"    Avg boxes/img  : {metrics['seg_stats']['mean_boxes']:.2f}")
    lines.append(f"    Total boxes    : {metrics['seg_stats']['total_boxes']}")

    lines += [
        "",
        "  3. SIFT MATCHING",
        dash,
        f"    Avg match acc  : {metrics['sift_match_acc']:.4f}",
        "",
        "  4. HARRIS CORNERS",
        dash,
    ]
    h = metrics["harris_stats"]
    lines.append(f"    Mean ± std     : {h['mean']:.1f} ± {h['std']:.1f}")
    lines.append(f"    Min / Max      : {h['min']} / {h['max']}")

    lines += [
        "",
        "  5. PIPELINE TIMING (milliseconds)",
        dash,
    ]
    for stage, t in metrics["timing_ms"].items():
        lines.append(f"    {stage:<22} {t['mean_ms']:8.2f} ± {t['std_ms']:.2f} ms")

    lines += ["", sep]
    report = "\n".join(lines)
    print("\n" + report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        log.info("  Report saved → %s", save_path)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="evaluation.py",
        description="Evaluate Traffic Sign pipeline — quantitative metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image_dir",  default=None)
    p.add_argument("--labels",     default=None,
                   help="CSV with ClassId, Name columns.")
    p.add_argument("--output_dir", default="eval_output")
    p.add_argument("--demo",       action="store_true",
                   help="Run demo with 30 synthetic images — no dataset needed.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.demo:
        log.info("Demo mode: generating synthetic traffic sign images.")
        class_names   = ["stop_sign", "speed_limit"]
        train_images, train_labels = [], []
        test_images,  test_labels  = [], []

        # 20 training images (10 per class)
        for i in range(20):
            img   = make_demo_image()
            label = i % 2
            if label == 1:
                img = cv2.rectangle(img.copy(), (30, 30), (170, 170), (0, 0, 200), -1)
            train_images.append(img)
            train_labels.append(label)

        # 10 test images
        for i in range(10):
            img   = make_demo_image()
            label = i % 2
            if label == 1:
                img = cv2.rectangle(img.copy(), (30, 30), (170, 170), (0, 0, 200), -1)
            test_images.append((f"test_{i:02d}", img))
            test_labels.append(label)

        metrics = evaluate(
            images=test_images,
            true_labels=test_labels,
            class_names=class_names,
            output_dir=args.output_dir,
            train_images=train_images,
            train_labels=train_labels,
        )

    elif args.image_dir and args.labels:
        import pandas as pd
        df          = pd.read_csv(args.labels)
        class_names = sorted(df["Name"].unique().tolist())
        exts        = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
        paths       = sorted(p for p in Path(args.image_dir).iterdir()
                              if p.suffix.lower() in exts)
        all_images  = [(p.stem, cv2.imread(str(p))) for p in paths]
        all_images  = [(n, i) for n, i in all_images if i is not None]
        all_labels  = [0] * len(all_images)   # replace with real labels from CSV

        n_train = max(1, int(len(all_images) * 0.7))
        metrics = evaluate(
            images=all_images[n_train:],
            true_labels=all_labels[n_train:],
            class_names=class_names,
            output_dir=args.output_dir,
            train_images=[i for _, i in all_images[:n_train]],
            train_labels=all_labels[:n_train],
        )
    else:
        sys.exit("Use --demo  OR  --image_dir + --labels")

    log.info("Done. All results in: %s/", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
