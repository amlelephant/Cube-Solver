"""
viz_encodings.py

Renders what each move-window encoding (encodings_move.py) actually looks
like on real recorded moves.

Two jobs, deliberately in one script so they cannot disagree:

  * Debugging. An encoding that looks like noise, or that lights up the
    wrong half of the cube, is visible here in seconds and would take a
    training run to notice otherwise. Every image below is produced by the
    SAME encoder function the trainer calls, on the SAME cropped window —
    if the picture looks right, the tensor is right.

  * Presentation. The chroma encodings turn a turn into a legible
    heat-map of motion — bright where the cube moved, hue telling you WHEN
    within the window it moved — and those make good figures.

Outputs (--out, default viz_out/):

  encoding_grid.png     one row per sampled move, one column per encoding,
                        plus the raw first frame for reference.
  chroma_gallery.png    one chroma render per move class (U, U', D, ...),
                        with the hue-to-time legend.
  hero_<class>.png      single large chroma renders.

Usage:
  cd ble
  python viz_encodings.py --sessions training_data/solve_*/
  python viz_encodings.py --sessions training_data/solve_*/ \
      --encodings chroma chroma8 --per-class 1 --size 320
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

import encodings_move as enc_mod
from encodings_move import ENCODINGS, hue_sweep, preview, time_legend
from train_move_classifier import (FRAME_ORDER, MoveDiffDataset,
                                   WCA_CLASS_NAMES)

# Dark-on-dark so the chroma renders (whose background is black) sit on a
# panel rather than a white page.
BG    = (18, 16, 22)
FG    = (232, 230, 238)
MUTED = (140, 138, 150)

FONT  = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, org, scale=0.5, colour=FG, thick=1):
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


def load_window(sample, size_hint=None) -> list[np.ndarray] | None:
    """The move's frames, cropped exactly as the trainer crops them."""
    paths, _, box = sample
    imgs = [cv2.imread(str(p)) for p in paths]
    if any(i is None for i in imgs):
        return None
    if box is not None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            imgs = [i[y1:y2, x1:x2] for i in imgs]
    return imgs


def pick_samples(ds: MoveDiffDataset, per_class: int, seed: int
                 ) -> list[tuple[int, int]]:
    """(sample_index, label) pairs, up to per_class of each class."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, (_, lab, box) in enumerate(ds.samples):
        # Only cropped moves — an uncropped one would be shown at a scale
        # the model never sees, which is the whole 2026-07-24 trap.
        if box is None:
            continue
        by_class.setdefault(lab, []).append(i)
    out = []
    for lab in sorted(by_class):
        idxs = by_class[lab]
        rng.shuffle(idxs)
        out.extend((i, lab) for i in idxs[:per_class])
    return out


def encoding_grid(ds, samples, names, size, out_path):
    """One row per move, one column per encoding, raw frame first."""
    pad, head, lgut = 10, 46, 74
    cols = ["raw"] + names
    w = lgut + len(cols) * (size + pad) + pad
    h = head + len(samples) * (size + pad) + pad
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)

    for c, name in enumerate(cols):
        x = lgut + c * (size + pad)
        label(canvas, name, (x, head - 16), 0.52, FG)
        if name in ENCODINGS:
            label(canvas, f"{ENCODINGS[name].channels}ch",
                  (x + size - 34, head - 16), 0.42, MUTED)

    for r, (idx, lab) in enumerate(samples):
        frames = load_window(ds.samples[idx])
        if frames is None:
            continue
        y = head + r * (size + pad)
        label(canvas, WCA_CLASS_NAMES[lab], (pad, y + size // 2), 0.85, FG, 2)
        for c, name in enumerate(cols):
            x = lgut + c * (size + pad)
            if name == "raw":
                tile = cv2.resize(frames[0], (size, size))
            else:
                tile = cv2.resize(preview(name, frames, enc_mod.IMG_SIZE),
                                  (size, size))
            canvas[y:y + size, x:x + size] = tile

    cv2.imwrite(str(out_path), canvas)
    return out_path


def chroma_gallery(ds, samples, size, out_path, name="chroma"):
    """A wall of motion heat-maps, one per class, with the time legend."""
    pad, head, foot = 12, 54, 78
    cols = 6
    rows = int(np.ceil(len(samples) / cols))
    w = cols * (size + pad) + pad
    h = head + rows * (size + pad) + foot
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)

    label(canvas, "Move-window motion, hue-coded by time",
          (pad + 2, 34), 0.74, FG, 2)

    for i, (idx, lab) in enumerate(samples):
        frames = load_window(ds.samples[idx])
        if frames is None:
            continue
        r, c = divmod(i, cols)
        x = pad + c * (size + pad)
        y = head + r * (size + pad)
        canvas[y:y + size, x:x + size] = cv2.resize(
            preview(name, frames, enc_mod.IMG_SIZE), (size, size))
        label(canvas, WCA_CLASS_NAMES[lab], (x + 6, y + size - 8), 0.62,
              FG, 2)

    # Legend: the hue sweep, start of window -> end of window.
    ly = h - foot + 22
    lw = min(w - 2 * pad - 190, 420)
    strip = time_legend(64, lw, 20)
    canvas[ly:ly + 20, pad + 92:pad + 92 + lw] = strip
    label(canvas, "start", (pad, ly + 15), 0.5, MUTED)
    label(canvas, "end of turn", (pad + 100 + lw, ly + 15), 0.5, MUTED)
    # chroma8 splits each instant into a brighten/darken pair, so hue still
    # sweeps with time but two adjacent hues share one instant. Saying just
    # "hue = time" would be a quarter-turn wrong on that encoding.
    extra = ("  (adjacent hue pairs = the same instant, brightening vs "
             "darkening)" if name == "chroma8" else "")
    label(canvas, "brightness = how much that pixel changed;  "
                  "white = changed throughout" + extra,
          (pad, ly + 44), 0.46, MUTED)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def hero(ds, sample, size, out_path, name="chroma"):
    """One large render — the figure-quality version of a single move."""
    idx, lab = sample
    frames = load_window(ds.samples[idx])
    if frames is None:
        return None
    img = cv2.resize(preview(name, frames, enc_mod.IMG_SIZE), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    pad = 26
    canvas = np.full((size + 2 * pad + 30, size + 2 * pad, 3), BG,
                     dtype=np.uint8)
    canvas[pad:pad + size, pad:pad + size] = img
    label(canvas, f"{WCA_CLASS_NAMES[lab]}   ({name})",
          (pad, size + pad + 24), 0.66, FG, 2)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def story(ds, sample, size, out_path, name="chroma"):
    """
    The explainer figure: the five raw window frames on top, the encoding
    they collapse into underneath, at the size the network sees.

    Worth having as its own output because the gallery images are pretty
    but unreadable to anyone who has not been told what they are — this
    one shows the input and the output in the same frame, so the claim
    "the colours are time" can be checked by eye against the raw strip.
    """
    idx, lab = sample
    frames = load_window(ds.samples[idx])
    if frames is None:
        return None

    n = len(frames)
    small = size // 2
    pad, head = 18, 56
    strip_w = n * small + (n - 1) * pad
    big = min(strip_w, size * 2)
    w = max(strip_w, big) + 2 * pad
    h = head + small + 46 + big + 64
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)

    label(canvas, f"{WCA_CLASS_NAMES[lab]}   one move window, 5 frames",
          (pad, 36), 0.72, FG, 2)

    x0 = (w - strip_w) // 2
    for i, f in enumerate(frames):
        x = x0 + i * (small + pad)
        canvas[head:head + small, x:x + small] = cv2.resize(f, (small, small))
        label(canvas, FRAME_ORDER[i], (x + 2, head + small + 16), 0.44, MUTED)

    y = head + small + 46
    x = (w - big) // 2
    canvas[y:y + big, x:x + big] = cv2.resize(
        preview(name, frames, enc_mod.IMG_SIZE), (big, big),
        interpolation=cv2.INTER_CUBIC)

    ly = y + big + 26
    lw = min(w - 2 * pad - 200, 380)
    canvas[ly:ly + 18, pad + 92:pad + 92 + lw] = time_legend(64, lw, 18)
    label(canvas, "start", (pad, ly + 14), 0.46, MUTED)
    label(canvas, "end", (pad + 100 + lw, ly + 14), 0.46, MUTED)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Render move-window encodings for inspection and figures")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--encodings", nargs="+",
                    default=["diffstack", "rgbtime", "rgbtime0",
                             "chroma", "chroma8"],
                    help=f"Any of: {sorted(ENCODINGS)}")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--hero-size", type=int, default=640)
    ap.add_argument("--heroes", type=int, default=4,
                    help="How many large single-move renders to write")
    ap.add_argument("--gallery-encoding", default="chroma")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="viz_out")
    args = ap.parse_args()

    unknown = [n for n in args.encodings if n not in ENCODINGS]
    if unknown:
        sys.exit(f"Unknown encoding(s) {unknown}; have {sorted(ENCODINGS)}")

    session_dirs = [Path(p) for pat in args.sessions
                    for p in (Path(".").glob(pat) if "*" in pat else [Path(pat)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found.")

    # diffstack loads the full 5-frame window, which every encoding here
    # needs; the dataset is only used as a loader, not for its tensors.
    ds = MoveDiffDataset(session_dirs, augment=False, label_mode="wca",
                         encoding="diffstack")
    if not len(ds):
        sys.exit("No samples found. Run postprocess_session.py first.")

    samples = pick_samples(ds, args.per_class, args.seed)
    if not samples:
        sys.exit("No CROPPED samples found — run cache_crops.py first.")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    written = []

    written.append(encoding_grid(ds, samples[:12], args.encodings, args.size,
                                 out / "encoding_grid.png"))
    written.append(chroma_gallery(ds, samples, args.size,
                                  out / f"gallery_{args.gallery_encoding}.png",
                                  args.gallery_encoding))
    rng = random.Random(args.seed)
    for s in rng.sample(samples, min(args.heroes, len(samples))):
        tag = WCA_CLASS_NAMES[s[1]].replace(chr(39), "p")
        for fn, stem in ((hero, "hero"), (story, "story")):
            p = fn(ds, s, args.hero_size,
                   out / f"{stem}_{tag}_{args.gallery_encoding}.png",
                   args.gallery_encoding)
            if p:
                written.append(p)

    print(f"\n  {len(samples)} move(s) sampled from {len(session_dirs)} "
          f"session(s)")
    for p in written:
        print(f"    {p}")


if __name__ == "__main__":
    main()
