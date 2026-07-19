"""
augment_motion_blur.py — build a motion-blur-augmented copy of a YOLO
detection dataset. Train split gets one extra copy of every image
convolved with a random linear motion kernel; val is copied untouched so
metrics stay comparable across runs.

Why: detection recall drops when the cube is waved quickly (beta
2026-07-16; the 20260715 run measured P=0.90 / R=0.82 — the model misses
blurred cubes rather than hallucinating). Ultralytics has no built-in
motion-blur augmentation, so we bake it into the dataset instead. Labels
are reused unchanged: a linear blur smears the cube only a few px past
its box, well inside YOLO's localization tolerance.

Run from inside cv/detection:
    python augment_motion_blur.py                # merged_dataset -> merged_dataset_mblur
    python augment_motion_blur.py --src S --dst D [--seed 0]
"""

import argparse
import math
import os
import random
import shutil

import cv2
import numpy as np

# kernel length as a fraction of image width: covers a gentle smear up to a
# hard sideways wave (at 1280px wide: ~13-45px of smear)
KLEN_FRAC = (0.010, 0.035)


def motion_kernel(length, angle_deg):
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for i in range(length):
        t = i - c
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    return k / k.sum()


def main():
    ap = argparse.ArgumentParser(description="Motion-blur-augment a YOLO dataset")
    ap.add_argument("--src", default="merged_dataset")
    ap.add_argument("--dst", default="merged_dataset_mblur")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} already exists — delete it first if you "
                         f"mean to regenerate")

    copied = blurred = 0
    for split in ("train", "val"):
        img_src = os.path.join(args.src, "images", split)
        lbl_src = os.path.join(args.src, "labels", split)
        img_dst = os.path.join(args.dst, "images", split)
        lbl_dst = os.path.join(args.dst, "labels", split)
        os.makedirs(img_dst)
        os.makedirs(lbl_dst)

        for name in sorted(os.listdir(img_src)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            shutil.copy2(os.path.join(img_src, name), os.path.join(img_dst, name))
            lbl = os.path.join(lbl_src, stem + ".txt")
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(lbl_dst, stem + ".txt"))
            copied += 1

            if split != "train":
                continue
            img = cv2.imread(os.path.join(img_src, name))
            if img is None:
                continue
            klen = max(5, int(random.uniform(*KLEN_FRAC) * img.shape[1]))
            klen += 1 - klen % 2  # odd
            out = cv2.filter2D(img, -1, motion_kernel(klen, random.uniform(0, 180)))
            cv2.imwrite(os.path.join(img_dst, stem + "_mblur" + ext), out)
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(lbl_dst, stem + "_mblur.txt"))
            blurred += 1

    with open(os.path.join(args.dst, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.dst)}\n"
                "train: images/train\n"
                "val: images/val\n"
                "names:\n  0: cube\n")

    print(f"done: {copied} originals copied, {blurred} motion-blurred "
          f"variants added -> {args.dst}")


if __name__ == "__main__":
    main()
