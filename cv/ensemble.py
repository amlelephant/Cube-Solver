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

  2. Orange-red disambiguation: these two are the hardest pair. When the
     combined scores for orange and red are within 0.15 of each other,
     we apply an extra HSV rule: if saturation is high AND hue < 15,
     prefer red; if hue > 10, prefer orange. This hard rule breaks ties
     that the soft scores cannot resolve.

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
        # Use HSV hue to break the tie
        if s > 80:    # only meaningful if saturation is high enough
            if h <= 11:
                combined["red"]    += 0.12
                combined["orange"] -= 0.06
            elif h >= 13:
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