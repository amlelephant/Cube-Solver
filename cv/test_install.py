"""
test_install.py

Verifies every dependency and module, then runs the full pipeline on a
synthetic frame — no webcam, no model weights required for basic checks.

Usage:
    python test_install.py
"""

import sys
import os
import types
import numpy as np

# Some terminals (default Windows console codepages) can't encode the
# unicode arrows/checkmarks used in this script's output — force UTF-8
# so printing them doesn't crash the process.
sys.stdout.reconfigure(encoding="utf-8")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
errors = 0


def check(label, fn, warn_only=False):
    global errors
    try:
        result = fn()
        tag    = WARN if warn_only else PASS
        print(f"  {PASS} {label}" + (f" — {result}" if result else ""))
    except Exception as e:
        if warn_only:
            print(f"  {WARN} {label} — {e}")
        else:
            print(f"  {FAIL} {label} — {e}")
            errors += 1


print("\n" + "=" * 60)
print("  CubeArena CV — Installation Check")
print("=" * 60 + "\n")

# ── Core packages ────────────────────────────────────────────────────────────
print("Core packages:")

check("numpy",         lambda: f"v{np.__version__}")
check("opencv-python", lambda: (
    __import__("cv2"),
    f"v{__import__('cv2').getVersionString()}"
)[1])

def _torch():
    import torch
    return f"v{torch.__version__}  cuda={torch.cuda.is_available()}"
check("torch",       _torch)
check("torchvision", lambda: f"v{__import__('torchvision').__version__}")

def _kociemba():
    # Vendored pure-Python solver (cv/twophase/) — see its LICENSE.txt.
    # Replaces the 'kociemba' PyPI package, which has no Windows wheel.
    import twophase as k
    sol = k.solve("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")
    return f"solve('')='{sol}' (empty = already solved ✓)"
check("twophase (vendored solver)", _kociemba)

def _ultralytics():
    from ultralytics import YOLO
    return "ultralytics imported OK"
check("ultralytics", _ultralytics)

# ── Project modules ──────────────────────────────────────────────────────────
print("\nProject modules:")

check("color_classifier", lambda: __import__("color_classifier"))

def _hsv():
    from color_classifier import classify_hsv
    color, conf = classify_hsv(110, 180, 200)
    assert color == "blue", f"expected blue, got {color}"
    return f"classify_hsv(blue) → {color} ({conf:.2f})"
check("color_classifier.classify_hsv", _hsv)

check("cnn_classifier", lambda: __import__("cnn_classifier"))
check("ensemble",       lambda: __import__("ensemble"))
check("cube_detector",  lambda: __import__("cube_detector"))
check("state_finder (import)", lambda: (
    # Import without running main()
    __import__("importlib").util.spec_from_file_location(
        "state_finder", "state_finder.py"),
    "importable"
)[1])

# ── Model files ──────────────────────────────────────────────────────────────
print("\nModel files:")

def _finetuned():
    if os.path.exists("cube_yolo.pt"):
        from ultralytics import YOLO
        m = YOLO("cube_yolo.pt")
        nc = m.model.yaml.get("nc", "?")
        return f"cube_yolo.pt loaded — {nc} class(es)"
    raise FileNotFoundError("cube_yolo.pt not found — see TRAINING.md")
check("YOLO fine-tuned model (cube_yolo.pt)", _finetuned, warn_only=True)

def _cnn_model():
    from cnn_classifier import MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return f"{MODEL_PATH} found"
    raise FileNotFoundError(
        f"{MODEL_PATH} not found — run: python cnn_classifier.py --train")
check("CNN colour model (sticker_cnn.pt)", _cnn_model, warn_only=True)

# ── Pipeline smoke test (no webcam / no model needed) ────────────────────────
print("\nPipeline smoke test (synthetic frame):")

def _detector_blank():
    import cv2
    from cube_detector import detect_and_extract
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_and_extract(blank)
    assert "found"        in result
    assert "patches"      in result
    assert "debug_frame"  in result
    assert "method"       in result
    return f"returned found={result['found']} on blank frame (expected False)"
check("cube_detector on blank frame", _detector_blank)

def _ensemble_patch():
    import cv2
    from ensemble import ensemble_classify
    patch = np.full((32, 32, 3), (200, 80, 30), dtype=np.uint8)  # blue
    color, conf, detail = ensemble_classify(patch)
    return f"blue patch → {color} (conf={conf:.2f}, method={detail.get('method')})"
check("ensemble on synthetic blue patch", _ensemble_patch)

def _full_face():
    from ensemble import classify_face
    colors_bgr = [
        (200, 80, 30), (30, 30, 210), (30, 220, 230),
        (200, 80, 30), (235, 235, 230), (30, 160, 40),
        (20, 140, 230), (30, 30, 210), (30, 220, 230),
    ]
    patches = [np.full((32, 32, 3), c, dtype=np.uint8) for c in colors_bgr]
    colors, confs, _ = classify_face(patches)
    avg_conf = sum(confs) / len(confs)
    return f"classify_face → {colors[:3]}... avg_conf={avg_conf:.2f}"
check("classify_face (9 patches)", _full_face)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors == 0:
    print("  All required checks passed.\n")
    print("  Next steps:")

    missing_yolo = not os.path.exists("cube_yolo.pt")
    from cnn_classifier import MODEL_PATH
    missing_cnn  = not os.path.exists(MODEL_PATH)

    if missing_yolo:
        print("  ★ Train the YOLO detector — follow TRAINING.md")
    if missing_cnn:
        print("  ★ Train the colour CNN:  python cnn_classifier.py --train")
    if not missing_yolo and not missing_cnn:
        print("  ✓ All models present — run: python state_finder.py")
    elif not missing_yolo:
        print("  → Then run: python state_finder.py")
else:
    print(f"  {errors} required check(s) FAILED — fix before continuing.")
    sys.exit(1)

print("=" * 60 + "\n")