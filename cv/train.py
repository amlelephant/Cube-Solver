"""
train.py

Fine-tunes YOLOv8n on the cube dataset exported from Roboflow.
Run this from inside cube-cv/ after completing Steps 1-9 in TRAINING.md.

Usage:
    python train.py
    python train.py --epochs 100
    python train.py --batch 8      # if you get out-of-memory errors
    python train.py --cpu          # force CPU even if GPU is available
"""

import argparse
import os
import shutil
import sys

# ── Argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train cube_yolo.pt")
parser.add_argument("--epochs", type=int,   default=60,
                    help="Number of training epochs (default: 60)")
parser.add_argument("--batch",  type=int,   default=16,
                    help="Batch size (default: 16; use 8 or 4 for low VRAM)")
parser.add_argument("--imgsz",  type=int,   default=640,
                    help="Input image size (default: 640; use 416 on CPU)")
parser.add_argument("--data",   type=str,   default="face_dataset/data.yaml",
                    help="Path to dataset data.yaml")
parser.add_argument("--cpu",    action="store_true",
                    help="Force CPU training even if a GPU is available")
args = parser.parse_args()

# ── Imports ────────────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Follow Step 7 in TRAINING.md to install the CUDA-enabled version.")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.")
    print("Run: pip install ultralytics")
    sys.exit(1)

if(__name__ == '__main__'):


    # ── Environment report ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CubeArena — YOLO Training")
    print("=" * 55)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.cuda.is_available()}")

    if torch.cuda.is_available() and not args.cpu:
        device = 0
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM     : {vram_gb:.1f} GB")
        if vram_gb < 4.0 and args.batch > 8:
            print(f"\n  WARNING: Low VRAM ({vram_gb:.1f} GB). Consider --batch 8 or --batch 4")
    else:
        device = "cpu"
        if args.cpu:
            print("  Device   : CPU (forced via --cpu flag)")
        else:
            print("  Device   : CPU (no NVIDIA GPU detected)")
        print("  NOTE: CPU training works but is slow (~90-120 min for 60 epochs)")
        print("        Follow Step 7 in TRAINING.md to enable GPU training")
        if args.imgsz == 640:
            print("  TIP: Add --imgsz 416 to speed up CPU training")

    print(f"\n  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Img size : {args.imgsz}")
    print("=" * 55 + "\n")

    # ── Dataset check ──────────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at '{args.data}'")
        print()
        print("Make sure you have:")
        print("  1. Downloaded the dataset from Roboflow (Steps 1-5 in TRAINING.md)")
        print("  2. Unzipped it into cube-cv/face_dataset/")
        print("  3. Updated the paths in data.yaml (Step 9 in TRAINING.md)")
        print("  4. Are running this script from inside cube-cv/")
        sys.exit(1)

    # ── Load base model ────────────────────────────────────────────────────────
    print("Loading YOLOv8n pretrained weights...")
    print("(Downloads ~6MB on first run)\n")
    model = YOLO("yolov8n.pt")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Starting training...\n")
    results = model.train(
        data      = args.data,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = device,
        name      = "cube_yolo",
        patience  = 15,          # early stop if val mAP doesn't improve for 15 epochs
        amp       = (device != "cpu"),  # mixed precision: GPU only
        plots     = True,        # saves results.png and val prediction images
        verbose   = True,
    )

    # ── Copy best weights to project root ─────────────────────────────────────
    src = os.path.join("runs", "detect", "cube_yolo", "weights", "best.pt")
    dst = "cube_yolo.pt"

    print("\n" + "=" * 55)

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  Model saved to: {dst}")

        # Print final metrics
        metrics = results  # ultralytics returns Results object
        try:
            map50 = results.results_dict.get("metrics/mAP50(B)", 0)
            map95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
            prec  = results.results_dict.get("metrics/precision(B)", 0)
            rec   = results.results_dict.get("metrics/recall(B)", 0)
            print(f"\n  Final metrics:")
            print(f"    mAP50    : {map50:.3f}  (target > 0.80)")
            print(f"    mAP50-95 : {map95:.3f}  (target > 0.55)")
            print(f"    Precision: {prec:.3f}  (target > 0.80)")
            print(f"    Recall   : {rec:.3f}  (target > 0.75)")

            if map50 >= 0.80:
                print("\n  RESULT: Good — model should detect your cube reliably")
            elif map50 >= 0.65:
                print("\n  RESULT: Decent — may miss cube in some conditions")
                print("          Add more images and retrain (see TRAINING.md)")
            else:
                print("\n  RESULT: Low accuracy — check data.yaml and class names")
                print("          See Troubleshooting section in TRAINING.md")
        except Exception:
            pass  # metrics parsing is best-effort

        print(f"\n  Training charts saved to: runs/detect/cube_yolo/")
        print(f"\n  Next step: python state_finder.py")

    else:
        print(f"  WARNING: best.pt not found at {src}")
        print(f"  Check runs/detect/cube_yolo/weights/ manually")

    print("=" * 55 + "\n")