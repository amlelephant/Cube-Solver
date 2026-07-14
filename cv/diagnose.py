"""
diagnose.py

Inspects your trained cube_yolo.pt and tells you exactly what went wrong.
Run this before anything else when detection seems wrong.

Usage:
    python diagnose.py
    python diagnose.py --image path/to/test.jpg
    python diagnose.py --webcam
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--image",  type=str, help="Path to a test image")
parser.add_argument("--webcam", action="store_true", help="Test live webcam")
parser.add_argument("--model",  type=str, default="cube_yolo.pt")
args = parser.parse_args()

# ── 1. Check model file exists ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CubeArena — Model Diagnostics")
print("=" * 60)

if not os.path.exists(args.model):
    print(f"\n  ERROR: {args.model} not found.")
    print("  Run: python train.py")
    sys.exit(1)

size_mb = os.path.getsize(args.model) / 1_000_000
print(f"\n  Model file : {args.model}")
print(f"  File size  : {size_mb:.1f} MB")

# A valid YOLOv8n fine-tuned model should be 5-7 MB.
# If it's much smaller something went wrong saving.
if size_mb < 3:
    print("  WARNING: File is very small — may be corrupted or empty.")
elif size_mb > 50:
    print("  NOTE: File is large — you may have saved a non-nano model.")
else:
    print("  File size looks correct for YOLOv8n.")

# ── 2. Load and inspect the model ────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("\n  ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

import torch
print(f"\n  PyTorch    : {torch.__version__}")
print(f"  CUDA       : {torch.cuda.is_available()}")

print("\n  Loading model...")
model = YOLO(args.model)

# ── 3. Check what classes it was trained on ───────────────────────────────
names = model.names
nc    = len(names)

print(f"\n  Classes trained ({nc} total):")
for idx, name in names.items():
    print(f"    [{idx}] {name}")

# The critical check
if nc == 80:
    print("\n  !! PROBLEM FOUND !!")
    print("  Your model has 80 classes — these are the standard COCO classes")
    print("  (person, car, cat, etc.), NOT your cube class.")
    print("  This means your data.yaml was misconfigured during training.")
    print("  The model never actually learned to detect a cube.")
    print("\n  FIX: See 'Fix A' printed at the bottom of this script.")
elif nc == 1 and list(names.values()) == ["cube"]:
    print("\n  Class config looks correct: 1 class named 'cube'.")
elif nc == 1:
    print(f"\n  1 class found: '{names[0]}'")
    if names[0].lower() not in ["cube", "rubik", "rubiks", "rubix"]:
        print("  WARNING: Class name is not 'cube' — check your data.yaml labels.")
else:
    print(f"\n  {nc} classes found: {list(names.values())}")
    if "cube" not in names.values():
        print("  WARNING: No class named 'cube' found.")
        print("  Your labels in Roboflow may not have been unified correctly.")

# ── 4. Check training results if available ────────────────────────────────
results_csv = os.path.join("runs", "detect", "cube_yolo", "results.csv")
if os.path.exists(results_csv):
    try:
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            # Strip whitespace from keys
            last = {k.strip(): v.strip() for k, v in last.items()}
            print("\n  Training results (final epoch):")
            for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                        "metrics/precision(B)", "metrics/recall(B)"]:
                val = last.get(key, last.get(key.replace("(B)", ""), "N/A"))
                print(f"    {key:<30} {val}")

            map50 = float(last.get("metrics/mAP50(B)",
                          last.get("metrics/mAP50", 0)))
            print()
            if map50 < 0.3:
                print("  !! PROBLEM: mAP50 is very low — the model barely learned anything.")
                print("     Most likely cause: wrong class names in data.yaml.")
            elif map50 < 0.6:
                print("  NOTE: mAP50 is mediocre. More data or epochs needed.")
            else:
                print(f"  mAP50 = {map50:.3f} — training looks healthy.")
    except Exception as e:
        print(f"  Could not parse results.csv: {e}")
else:
    print("\n  No results.csv found at runs/detect/cube_yolo/results.csv")
    print("  Cannot check training metrics.")

# ── 5. Run inference on a test image or webcam ────────────────────────────
import cv2
import numpy as np

def run_inference_on_frame(frame, label="test"):
    results = model(frame, verbose=False, conf=0.10)  # very low conf to see anything
    boxes_found = 0
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names.get(cls, str(cls))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"    Detected: '{name}' (class {cls})  conf={conf:.3f}  "
                  f"box=[{x1},{y1},{x2},{y2}]")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 60), 2)
            cv2.putText(frame, f"{name} {conf:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 230, 60), 1)
            boxes_found += 1
    if boxes_found == 0:
        print("    Nothing detected at conf >= 0.10")
    return frame, boxes_found

if args.image:
    print(f"\n  Running inference on: {args.image}")
    frame = cv2.imread(args.image)
    if frame is None:
        print("  ERROR: Could not read image.")
    else:
        frame_out, n = run_inference_on_frame(frame, "image")
        out_path = "diagnose_output.jpg"
        cv2.imwrite(out_path, frame_out)
        print(f"  Saved annotated image to: {out_path}")

elif args.webcam:
    print("\n  Opening webcam — press Q to quit, S to save a frame...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        out, _ = run_inference_on_frame(frame.copy())
        cv2.imshow("Diagnose — conf >= 0.10", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite("diagnose_frame.jpg", out)
            print("  Saved diagnose_frame.jpg")
    cap.release()
    cv2.destroyAllWindows()

else:
    # Run on a solid white test image just to confirm model responds at all
    print("\n  Running on synthetic blank frame (sanity check)...")
    blank = np.full((640, 640, 3), 128, dtype=np.uint8)
    _, n = run_inference_on_frame(blank)
    print("  (Expected: nothing detected on a grey frame)")

# ── 6. Print fix instructions ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Diagnosis complete.")
print("=" * 60)

print("""
Common problems and fixes:

FIX A — Model has 80 COCO classes instead of 1 cube class
  Your data.yaml nc value was wrong or Roboflow exported with wrong
  class config. Check cube-detector-final/data.yaml and confirm:
      nc: 1
      names: ['cube']
  If it says nc: 2 or lists 'Scrambled'/'Solved' as separate classes,
  merge them in Roboflow and re-export. Then re-run train.py.

FIX B — Model has 1 class but detects you instead of the cube
  The training images had you in them without the cube labeled,
  so the model learned YOUR appearance as the background and is
  firing on high-contrast regions.
  Solution: open Roboflow, inspect ~20 random training images.
  If any show a person holding the cube but the CUBE is not labeled
  (only empty space around it), the model learned person=cube.
  Add negative examples (images with NO cube) labeled as background,
  re-augment, re-export, re-train.

FIX C — mAP50 was below 0.5 after training
  Not enough images or too many epochs caused overfitting.
  Check runs/detect/cube_yolo/val_batch0_pred.jpg — does it even
  detect the cube on the validation set images?
  If not: add 50+ photos of YOUR cube in YOUR room, label in
  Roboflow, re-export, re-train.

FIX D — Detects cube sometimes but very low confidence
  Lower the threshold in cube_detector.py:
      YOLO_CONF = 0.20   (was 0.35)
  If still inconsistent, add more images of your specific cube.
""")