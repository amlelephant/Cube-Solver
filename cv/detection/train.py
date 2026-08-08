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
parser.add_argument("--model",  type=str,   default="yolov8n.pt",
                    help="Base weights to fine-tune (default: yolov8n.pt; "
                         "yolov8s.pt for higher accuracy on a decent GPU)")
parser.add_argument("--name",   type=str,   default="cube_yolo",
                    help="Run name under runs/detect/ (default: cube_yolo)")
parser.add_argument("--patience", type=int, default=15,
                    help="Early-stop patience in epochs (default: 15)")
parser.add_argument("--no-copy", action="store_true",
                    help="Don't copy best.pt over cube_yolo.pt afterwards "
                         "(for experiments that shouldn't touch the deployed "
                         "weights)")
parser.add_argument("--cpu",    action="store_true",
                    help="Force CPU training even if a GPU is available")
parser.add_argument("--lr0",    type=float, default=None,
                    help="Fixed initial learning rate (default: ultralytics' "
                         "auto-selected value). Set this explicitly and low "
                         "(e.g. 0.0005) for a small, conservative fine-tune "
                         "from an already-good checkpoint — the auto-picked "
                         "rate is tuned for training from scratch and can "
                         "overshoot in just a few epochs on a small dataset "
                         "(2026-08-03: auto lr0=0.002 on a 1884-image "
                         "fine-tune from detect_full_cube.pt regressed "
                         "recall badly by epoch 1).")
parser.add_argument("--freeze", type=int,   default=None,
                    help="Freeze the first N backbone layers (default: none "
                         "frozen). Keeps early general-purpose features "
                         "fixed during a small fine-tune, reducing the risk "
                         "of catastrophic forgetting.")
parser.add_argument("--optimizer", type=str, default="auto",
                    help="ultralytics optimizer name, e.g. SGD, AdamW, or "
                         "auto (default: auto-selected, matches prior "
                         "behavior)")
parser.add_argument("--workers", type=int,   default=None,
                    help="DataLoader worker processes (default: ultralytics' "
                         "default, 8). Set to 0 to disable multiprocessing "
                         "if training hits FileNotFoundError from a worker "
                         "process (2026-08-03: hit this reproducibly on a "
                         "hardlinked dataset — looked like AV/indexer lock "
                         "contention on freshly-touched files racing 8 "
                         "parallel workers, not present with workers=0).")
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
    print(f"Loading pretrained weights: {args.model}")
    print("(Downloads on first run)\n")
    model = YOLO(args.model)

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Starting training...\n")
    train_kwargs = dict(
        data      = args.data,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = device,
        name      = args.name,
        patience  = args.patience,  # early stop if val mAP doesn't improve
        amp       = (device != "cpu"),  # mixed precision: GPU only
        plots     = True,        # saves results.png and val prediction images
        verbose   = True,
        optimizer = args.optimizer,
        hsv_h     = 0.06,        # wider hue aug — indoor lighting shifts color
                                 # far beyond the 0.015 default (see 2026-07-15
                                 # night-lighting diagnosis)
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    results = model.train(**train_kwargs)

    # ── Copy best weights to project root ─────────────────────────────────────
    # results.save_dir, not a hardcoded path: when runs/detect/cube_yolo already
    # exists ultralytics writes to cube_yolo2/3/..., and the hardcoded path would
    # silently copy a previous run's weights.
    src = os.path.join(str(results.save_dir), "weights", "best.pt")
    dst = "cube_yolo.pt"

    print("\n" + "=" * 55)

    if os.path.exists(src):
        if args.no_copy:
            print(f"  --no-copy: weights left at {src}")
        else:
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