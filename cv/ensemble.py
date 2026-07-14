"""
ensemble.py

Combines CNN probability outputs with HSV confidence scores to produce
the best possible single color prediction for each sticker region.

Fusion strategy
---------------
Both classifiers return a score for each of the 6 colors.
We combine them with a weighted sum, then apply two post-processing rules:

  1. Agreement bonus: if CNN and HSV independently agree on the top class,
     boost that class's combined score by 20% before final argmax.

  2. Orange-red disambiguation: these two are the hardest pair — orange
     reads redder than its nominal hue under most non-daylight lighting,
     so a fixed universal hue threshold can't separate them reliably across
     different rooms/cameras. When the combined scores for orange and red
     are within 0.15 of each other, we break the tie using hue relative to
     a boundary that is *calibrated per session* from the two center
     stickers we already scan (R's center is always red, L's is always
     orange — see calibrate()). Falls back to a fixed default before both
     centers have been seen.

Weights are configurable. Defaults:
  CNN weight:  0.65  (generalises better across lighting conditions)
  HSV weight:  0.35  (fast, interpretable, good in ideal conditions)
"""

import numpy as np
from color_classifier import (
    classify_region_hsv,
    hsv_confidence,
    CLASSES,
    COLOR_BGR,
    COLOR_TO_FACE,
)
from cnn_classifier import classify_region_cnn
import cv2

# How much weight to give each classifier in the soft fusion
CNN_WEIGHT = 0.65
HSV_WEIGHT = 0.35

# If the top-2 combined scores are within this margin, apply tiebreak rules
TIEBREAK_MARGIN = 0.15

# ---------------------------------------------------------------------------
# Per-session color calibration
# ---------------------------------------------------------------------------
# The caller (state_finder.py) knows each face's true center color in
# advance from the fixed WCA scheme, before it ever runs classification on
# it. We use that free ground truth to adapt the orange/red boundary to
# this session's actual camera + lighting, rather than trusting one fixed
# hue threshold that can't account for lighting-dependent hue drift — the
# main reason orange gets misread as red.
_calibrated_hue = {}   # color_name -> measured hue from a known-color sample
_DEFAULT_ORANGE_RED_BOUNDARY = 12  # matches the original fixed threshold


def calibrate(color_name, h, s, v):
    """
    Record a known-good HSV sample for a color (e.g. a face's center
    sticker, whose color is known in advance, not predicted).
    """
    _calibrated_hue[color_name] = h


def reset_calibration():
    """Clear calibration state — call at the start of a new scan session."""
    _calibrated_hue.clear()


def calibration_status():
    """
    {"red": hue or None, "orange": hue or None, "active": bool} — for
    debug/UI display, so it's visible whether the session is using
    calibrated or fallback thresholds.
    """
    red_h    = _calibrated_hue.get("red")
    orange_h = _calibrated_hue.get("orange")
    return {
        "red":    red_h,
        "orange": orange_h,
        "active": red_h is not None and orange_h is not None,
    }


def _orange_red_boundary():
    """
    Hue that separates red from orange for this session.

    Uses the midpoint between this session's actual calibrated red and
    orange hues once both are known; falls back to the original fixed
    threshold otherwise.
    """
    red_h    = _calibrated_hue.get("red")
    orange_h = _calibrated_hue.get("orange")
    if red_h is not None and orange_h is not None and orange_h > red_h:
        return (red_h + orange_h) / 2
    return _DEFAULT_ORANGE_RED_BOUNDARY


def ensemble_classify(bgr_region):
    """
    Classify a single BGR sticker patch using the full ensemble.

    Parameters
    ----------
    bgr_region : np.ndarray
        Small BGR image crop of one sticker (any size; both classifiers resize internally).

    Returns
    -------
    color   : str
        Predicted color name ("white", "yellow", "red", "orange", "blue", "green",
        or "unknown" if all scores are very low).
    confidence : float
        Overall confidence 0.0-1.0. Below 0.4 means the prediction is uncertain.
    detail  : dict
        Full breakdown for UI display:
        {
          "cnn_probs":     {color: prob, ...},   # raw CNN output (None if no model)
          "hsv_scores":    {color: score, ...},  # raw HSV confidence
          "combined":      {color: score, ...},  # weighted fusion
          "method":        "ensemble" | "hsv_only",
          "agreement":     bool,
        }
    """
    # ---- Step 1: Get HSV scores -----------------------------------------------
    if bgr_region is None or bgr_region.size == 0:
        return "unknown", 0.0, {}

    hsv_region = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    h = float(np.median(hsv_region[:, :, 0]))
    s = float(np.median(hsv_region[:, :, 1]))
    v = float(np.median(hsv_region[:, :, 2]))

    hsv_scores = hsv_confidence(h, s, v)
    hsv_total  = sum(hsv_scores.values())
    # Normalise HSV scores so they sum to 1 (like probabilities)
    if hsv_total > 0:
        hsv_norm = {c: hsv_scores[c] / hsv_total for c in CLASSES}
    else:
        hsv_norm = {c: 1.0 / len(CLASSES) for c in CLASSES}

    hsv_top = max(hsv_norm, key=hsv_norm.get)

    # ---- Step 2: Get CNN probabilities -----------------------------------------
    cnn_probs = classify_region_cnn(bgr_region)
    cnn_available = cnn_probs is not None

    if cnn_available:
        cnn_top = max(cnn_probs, key=cnn_probs.get)
        # Weighted fusion
        combined = {
            c: CNN_WEIGHT * cnn_probs[c] + HSV_WEIGHT * hsv_norm[c]
            for c in CLASSES
        }
        method = "ensemble"
        agreement = (cnn_top == hsv_top)
    else:
        # No CNN model available — fall back to HSV only
        combined = dict(hsv_norm)
        method = "hsv_only"
        agreement = False
        cnn_probs = None

    # ---- Step 3: Agreement bonus -----------------------------------------------
    if agreement:
        bonus_color = hsv_top
        combined[bonus_color] = min(combined[bonus_color] * 1.20, 1.0)

    # ---- Step 4: Orange-red tiebreak -------------------------------------------
    orange_score = combined.get("orange", 0)
    red_score    = combined.get("red",    0)

    if abs(orange_score - red_score) < TIEBREAK_MARGIN:
        # Use hue relative to this session's calibrated orange/red boundary
        # to break the tie (falls back to a fixed default before both
        # center stickers have been calibrated — see calibrate() above).
        if s > 80:    # only meaningful if saturation is high enough
            boundary = _orange_red_boundary()
            if h <= boundary - 1:
                combined["red"]    += 0.12
                combined["orange"] -= 0.06
            elif h >= boundary + 1:
                combined["orange"] += 0.12
                combined["red"]    -= 0.06

    # ---- Step 5: Final decision -------------------------------------------------
    best_color = max(combined, key=combined.get)
    best_score = combined[best_color]

    # Reject if everything is too uncertain
    if best_score < 0.20:
        best_color = "unknown"

    detail = {
        "cnn_probs":  cnn_probs,
        "hsv_scores": hsv_scores,
        "combined":   combined,
        "method":     method,
        "agreement":  agreement,
        "h": round(h), "s": round(s), "v": round(v),
    }

    return best_color, round(best_score, 3), detail


def classify_face(sticker_regions):
    """
    Classify all 9 sticker regions from one cube face.

    Parameters
    ----------
    sticker_regions : list of np.ndarray
        9 BGR image patches, in row-major order (top-left to bottom-right).

    Returns
    -------
    colors      : list of str          — 9 color names
    confidences : list of float        — 9 confidence values
    details     : list of dict         — 9 detail dicts from ensemble_classify
    """
    colors, confidences, details = [], [], []
    for region in sticker_regions:
        color, conf, detail = ensemble_classify(region)
        colors.append(color)
        confidences.append(conf)
        details.append(detail)
    return colors, confidences, details