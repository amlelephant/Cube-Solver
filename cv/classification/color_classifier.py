"""
color_classifier.py

HSV-based color classifier with confidence scoring.
Used both standalone and as one arm of the ensemble.

Color space notes:
  OpenCV HSV: H = 0-179, S = 0-255, V = 0-255
  Red wraps around at H=0 and H=179, so we check two ranges.
"""

import numpy as np

# Canonical ordered list of the 6 cube colors.
# Both cnn_classifier and ensemble import this from here.
CLASSES = ["white", "yellow", "red", "orange", "blue", "green"]

# ---------------------------------------------------------------------------
# HSV range definitions
# Each entry: list of (h_min, h_max, s_min, s_max, v_min, v_max) tuples.
# Multiple tuples = OR logic (used for red's hue wrap-around).
# ---------------------------------------------------------------------------
HSV_RANGES = {
    "white": [
        (0, 179, 0, 60, 170, 255),
    ],
    "yellow": [
        (18, 38, 80, 255, 80, 255),
    ],
    "red": [
        (0, 12, 90, 255, 60, 255),    # lower red
        (165, 179, 90, 255, 60, 255), # upper red (wrap-around)
    ],
    "orange": [
        (8, 22, 100, 255, 60, 255),
    ],
    "blue": [
        (90, 135, 60, 255, 30, 255),
    ],
    "green": [
        (40, 90, 50, 255, 30, 255),
    ],
}

# Display BGR for each color (used by the UI overlay)
COLOR_BGR = {
    "white":   (255, 255, 255),
    "yellow":  (0,   230, 255),
    "red":     (0,   0,   210),
    "orange":  (0,   140, 255),
    "blue":    (210, 60,  0  ),
    "green":   (0,   180, 50 ),
    "unknown": (90,  90,  90 ),
}

# Standard cube face letters
COLOR_TO_FACE = {
    "white":  "U",
    "red":    "R",
    "green":  "F",
    "yellow": "D",
    "orange": "L",
    "blue":   "B",
}

FACE_TO_COLOR = {v: k for k, v in COLOR_TO_FACE.items()}


def hsv_confidence(h, s, v):
    """
    Return a dict of {color_name: confidence_score} for a single HSV pixel.

    Confidence is computed as how deeply the value falls inside each color's
    range, normalized 0.0-1.0. Values outside all ranges return 0.0 for that
    color. The returned dict always has all 6 colors as keys.
    """
    scores = {name: 0.0 for name in HSV_RANGES}

    for color_name, range_list in HSV_RANGES.items():
        best = 0.0
        for (h_min, h_max, s_min, s_max, v_min, v_max) in range_list:
            # Check membership
            h_ok = h_min <= h <= h_max
            s_ok = s_min <= s <= s_max
            v_ok = v_min <= v <= v_max

            if h_ok and s_ok and v_ok:
                # How centered is each channel within its range?
                h_span = max(h_max - h_min, 1)
                s_span = max(s_max - s_min, 1)
                v_span = max(v_max - v_min, 1)

                h_center = 1.0 - abs((h - (h_min + h_max) / 2) / (h_span / 2))
                s_center = 1.0 - abs((s - (s_min + s_max) / 2) / (s_span / 2))
                v_center = 1.0 - abs((v - (v_min + v_max) / 2) / (v_span / 2))

                # Weighted: hue matters most for color identity
                score = 0.5 * h_center + 0.3 * s_center + 0.2 * v_center
                best = max(best, score)

        scores[color_name] = round(best, 4)

    return scores


def classify_hsv(h, s, v):
    """
    Return (color_name, confidence) for a single HSV triplet.
    Confidence is the winning score from hsv_confidence.
    """
    scores = hsv_confidence(h, s, v)
    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]

    if best_score == 0.0:
        return "unknown", 0.0

    return best_color, best_score


def classify_region_hsv(bgr_region):
    """
    Given a small BGR image patch (the sticker region), return
    (color_name, confidence) using the median HSV value of the patch.

    Using median rather than mean makes it robust to specular highlights
    and dark edge shadows that appear at sticker borders.
    """
    import cv2
    if bgr_region is None or bgr_region.size == 0:
        return "unknown", 0.0

    hsv_region = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)

    # Median is more robust than mean for small patches with edge noise
    h = float(np.median(hsv_region[:, :, 0]))
    s = float(np.median(hsv_region[:, :, 1]))
    v = float(np.median(hsv_region[:, :, 2]))

    return classify_hsv(h, s, v)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        (5,   220, 200, "red"),
        (170, 200, 180, "red"),     # wrap-around red
        (15,  210, 180, "orange"),
        (28,  200, 200, "yellow"),
        (0,   10,  240, "white"),
        (110, 180, 200, "blue"),
        (60,  180, 180, "green"),
    ]

    print("HSV Classifier Self-Test")
    print("-" * 50)
    all_pass = True
    for h, s, v, expected in test_cases:
        result, conf = classify_hsv(h, s, v)
        ok = result == expected
        all_pass = all_pass and ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] H={h:3d} S={s:3d} V={v:3d} → {result:<8s} (conf={conf:.2f})  expected={expected}")

    print("-" * 50)
    print("All tests passed." if all_pass else "Some tests FAILED — tune HSV_RANGES.")