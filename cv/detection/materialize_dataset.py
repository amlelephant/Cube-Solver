"""
materialize_dataset.py — build a YOLO train/val dataset from cube_dataset/
plus cv/labeling/mine_out/hard_negatives.json, without duplicating any image
data.

Every file placed under the output dataset is a hardlink (os.link — same
NTFS volume, zero extra disk, new directory entry pointing at the same
bytes), never a copy: cube_dataset/'s existing 2050 Roboflow-exported images
stay untouched and un-duplicated, and the ~100k-frame ble/training_data/
corpus that hard_negatives.json's 34 entries point into is never touched
either. Hard negatives get an empty YOLO label file (zero objects) so the
detector learns those specific boxes are not a cube, split across train/val
in the same proportion as the existing positives.

Run from inside cv/detection/ (bare model filenames — see CLAUDE.md):
    python materialize_dataset.py
    python train.py --data cube_dataset_hardneg/data.yaml --model detect_full_cube.pt --name detect_hardneg
"""

import json
import os
import random

BASE = "cube_dataset"
HARD_NEG_MANIFEST = "../labeling/mine_out/hard_negatives.json"
OUT = "cube_dataset_hardneg"
VAL_FRAC = 0.15  # matches cube_dataset's own existing ~9.5% val split closely
                 # enough, and autolabel.py's convention elsewhere


def _hardlink(src, dst, retries=5, delay=0.1):
    # This project's Documents folder is OneDrive-synced (see the stale
    # absolute path in merged_dataset/data.yaml) — a burst of hardlink
    # creation can outrun the cloud-filter driver, same race autolabel.py's
    # _write_retrying works around. A dataloader worker reading moments
    # later than materialize_dataset.py wrote it hit exactly this once
    # (FileNotFoundError mid-training on a file that existed by the time it
    # was checked afterward) — verify each link is actually stat-able before
    # moving on, not just that os.link() didn't raise.
    import time
    if os.path.exists(dst):
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(retries):
        try:
            os.link(src, dst)
            os.stat(dst)
            return
        except FileNotFoundError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _link_split(pairs, split):
    for img_src, lbl_src, stem in pairs:
        img_dst = os.path.join(OUT, "images", split, stem + os.path.splitext(img_src)[1])
        lbl_dst = os.path.join(OUT, "labels", split, stem + ".txt")
        _hardlink(img_src, img_dst)
        if lbl_src is None:
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
            open(lbl_dst, "w").close()  # empty label = zero objects (negative)
        else:
            _hardlink(lbl_src, lbl_dst)


def main():
    random.seed(0)

    # 1. existing positives: hardlink cube_dataset's own train/valid split
    #    straight across (it's already split; we don't reshuffle it).
    n_pos_train = n_pos_val = 0
    for split, sub in (("train", "train"), ("val", "valid")):
        img_dir = os.path.join(BASE, sub, "images")
        lbl_dir = os.path.join(BASE, sub, "labels")
        pairs = []
        for n in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(n)[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            pairs.append((os.path.join(img_dir, n),
                          lbl if os.path.isfile(lbl) else None,
                          "pos_" + stem))
        _link_split(pairs, split)
        if split == "train":
            n_pos_train = len(pairs)
        else:
            n_pos_val = len(pairs)

    # 2. hard negatives: fresh split (we control this set), same VAL_FRAC
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(HARD_NEG_MANIFEST) as f:
        negs = json.load(f)
    neg_pairs = []
    for i, n in enumerate(negs):
        img_src = os.path.join(repo_root, n["frame"])
        stem = f"neg_{n['session']}_{i:04d}"
        neg_pairs.append((img_src, None, stem))
    random.shuffle(neg_pairs)
    n_val = max(1, round(len(neg_pairs) * VAL_FRAC)) if neg_pairs else 0
    _link_split(neg_pairs[:n_val], "val")
    _link_split(neg_pairs[n_val:], "train")

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\n"
                "train: images/train\nval: images/val\n"
                "names:\n  0: cube\n")

    print(f"{OUT}/  (all hardlinks — no image bytes duplicated)")
    print(f"  train: {n_pos_train} positive + {len(neg_pairs) - n_val} hard-negative "
          f"= {n_pos_train + len(neg_pairs) - n_val}")
    print(f"  val  : {n_pos_val} positive + {n_val} hard-negative "
          f"= {n_pos_val + n_val}")
    print(f"\nNext: python train.py --data {OUT}/data.yaml "
          "--model detect_full_cube.pt --name detect_hardneg")


if __name__ == "__main__":
    main()
