"""
solved_check.py — is the cube on camera SOLVED, right now, with no facelet
registration and no knowledge of which cube it is.

Why this exists
---------------
The one physical attack the move-count gate cannot see (ANTICHEAT.md §3) is:
make a full solve's worth of plausible moves without solving, stop the timer,
then substitute a solved cube before the verification scan. The count is
genuinely real, so counting cannot help; and appearance-magnitude swap
detection is measured-dead (`swap_check.py`, 2026-08-04) because the largest
persistent appearance change in an HONEST session is the solve completing —
the same transformation, at the same moment.

This file attacks it from the other end. Instead of asking "is this the same
cube" — unanswerable against a second cube of the same brand — it asks the
question that has a deterministic answer:

    at the moment the timer stopped, was the cube on camera solved?

A cheat that flails and stops has a scrambled cube on camera at t_stop and is
caught BEFORE any substitution occurs. Pair it with custody continuity over
the (short, deliberate) post-stop scan window and the swap has nowhere left
to happen: solved at the stop, same continuously-tracked object through a
scan that validates solved.

The statistic: colour FRAGMENTATION, not colour identity
--------------------------------------------------------
`cube_detector.detect_and_extract` boxes the whole cube, so its 3x3 slice
only lands on real facelets when a face is held flat-on. At the timer stop
the cube is at whatever angle the solver's hands left it, so anything
needing a facelet grid is out. Nothing here registers stickers.

A solved cube shows 1-3 SOLID faces: each visible colour is one large
connected blob. A scrambled cube is a fine mosaic of the same colours. So:

    fragmentation = 1 - (sum over classes of largest connected component)
                        / (total classified area)

0.0 = every colour present forms a single blob (solved). Higher = mosaic.
Rotation-invariant, scale-invariant, needs no grid, and says nothing at all
about WHICH cube it is — which is the point, since the two cubes in this
attack are chosen to be identical.

Merging opposite faces is FREE, and it is what makes this robust
----------------------------------------------------------------
White/yellow, red/orange and blue/green are OPPOSITE face pairs, so on a
solved cube they can never be visible simultaneously. Collapsing each pair
into one "axis" class therefore costs exactly nothing on the solved side, and
it deletes every hard colour discrimination in this project at once — the
orange/red tiebreak (`ensemble.calibrate`, the hardest pair and the one that
needs per-session calibration), white/yellow under warm light, and blue/green.
What survives is warm-vs-yellow and yellow-vs-cool: two boundaries tens of
degrees of hue apart.

The cost lands only on the scrambled side, where same-axis stickers merge and
the mosaic reads slightly less fragmented than it truly is. That is the right
direction to lose accuracy in: it makes the test CONSERVATIVE (a scrambled
cube looks more solved than it is), so the error it commits is a miss, never
a false DQ. `--palette full` scores the unmerged 6-class version for
comparison.

Which statistic ships, and why it is not the best-looking one
-------------------------------------------------------------
Three candidates are computed: `fragmentation`, `max_blob` (largest single
solid region as a share of the cube) and `n_regions` (count of distinct solid
regions). Swept over saturation floor x close-kernel and scored at the
zero-false-DQ operating point, `fragmentation` peaks highest in sample — and
that peak is an artifact. The threshold is pinned by the WORST legitimate
solve, so it is an `extreme`-kind statistic: one session IS the operating
point, and its neighbours in the sweep collapse from 86% to 61% and 43%.
Held out by date, `fragmentation` false-DQs in 10 of 15 settings.

`n_regions` holds the no-false-DQ line across almost the whole parameter
surface, at all three cut dates tried, which is the property a gate that must
never false-DQ actually needs. It ships. Measured at S_MIN=100,
CLOSE_KSIZE=11, threshold `n_regions > 4`:

  | date cut | held-out catch | held-out false DQ |
  |---|---|---|
  | before 2026-07-28 | 71%  (10/14) | 0%  (0/14) |
  | before 2026-07-31 | 90%  (9/10)  | 0%  (0/10) |
  | before 2026-08-02 | 100% (8/8)   | 0%  (0/8)  |

Read the catch column as "roughly three quarters to all", not as three
distinct numbers: each holdout has 8-14 attack sessions, so one session is
7-12 points and the spread between those cells is within the noise. The
false-DQ column is the one making a claim.

The residual defect, and why it is not a constant to tune
---------------------------------------------------------
The 8 in-sample misses were diagnosed from label maps rather than guessed at,
and they share one cause: **in warm evening light the BACKGROUND classifies as
a single enormous solid region**, so the wall becomes the "largest blob" and a
scrambled cube scores as perfectly solved (fragmentation as low as 0.013 —
lower than any genuine solve). A beige wall is low-saturation and bright,
which is exactly the white-sticker test; colour alone cannot separate cube
from scene.

Three geometric fixes were tried on the theory that the background enters via
the axis-aligned box's corners. **All three are measured worse** and none is
the answer:

  | variant | best held-out catch at 0 false-DQ |
  |---|---|
  | no mask (`box`, shipped) | **90%** |
  | inscribed hexagon | 80% |
  | inscribed ellipse | 60% |
  | centre crop 0.65 / 0.55 / 0.45 | no setting held the line at all |

The reason is the same in every case: the cube fills most of its own box, so
any mask that removes background removes cube in similar proportion,
compressing both sides of the statistic and adding noise. The real fix is a
cube SEGMENTATION rather than a box — the detector emitting a mask or a true
face quad — which is actual work and not a threshold. Until then this is the
same evening-lighting cliff the rest of the pipeline has, arriving through a
different door, and the honest response is the abstain path in `solved_at`.

Two consequences worth stating plainly. The parameter surface is noisy: 57
sessions with ~10 per holdout means one session is 7-12 points, so the
difference between the 71%/90%/100% cells is not real and neither is the gap
between the top few sweep cells. And the test is EVADABLE by presenting the
cube badly at the stop — which is why an unreadable window abstains to review
rather than passing.

Measuring it
------------
`ble/training_data` is already a labelled corpus for exactly this, at no
cost: `*_solve` sessions end SOLVED and `*_scramble` sessions end SCRAMBLED,
and a scramble session's end state IS the attack state — a cube that has had
a full sequence of real moves performed on it and is not solved. Same
structural proxy that the count gate's 21/21 catch rests on.

    cd cv/detection
    python dump_trajectories.py --root ../../ble/training_data   # once
    python solved_check.py score --root ../../ble/training_data

Run from inside cv/detection (see CLAUDE.md).
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Pixel -> axis class.
#
# Class ids: 0 unassigned, 1 warm (red|orange), 2 light (white|yellow),
# 3 cool (blue|green). Hue cuts sit in the wide gaps BETWEEN the merged
# pairs, never inside one, which is the whole reason for merging.
# --------------------------------------------------------------------------
V_MIN        = 40    # below this it is black plastic, a sticker gap or shadow
S_MIN        = 100   # a cube sticker is vivid. MEASURED (see `sweep`): at 70
                     #   the desk and the solver's hands classify as warm,
                     #   merge with the cube's red/orange stickers into one
                     #   enormous blob, and a SCRAMBLED cube reads as solved.
S_WHITE      = 70    # white stickers are the low-saturation case
V_WHITE      = 120   # a white sticker is bright; dim grey is the cube body
                     #
                     # S_WHITE < S_MIN leaves a deliberate DEAD BAND at
                     # 70 <= S < 100: too washed out to be a coloured sticker,
                     # too saturated to be white. Skin and wood live there.
                     # The band is load-bearing, not an oversight — closing it
                     # is what the s_min=70 row of the sweep measures, and it
                     # costs the whole result.
H_WARM_MAX   = 21    # warm | yellow  (orange centres ~15, yellow ~28)
H_YELLOW_MAX = 46    # yellow | cool  (green centres ~65)
H_WRAP       = 145   # cool | warm    (red wraps past 165)

N_CLASSES = 3

# Six-class palette, for --palette full. Same cuts plus the three hard
# in-pair splits this file exists to avoid needing.
H_RED_MAX    = 11    # red | orange, inside the warm class
H_GREEN_MAX  = 88    # green | blue, inside the cool class

WORK_LONG    = 120   # crop is resized so its longer side is this many px.
                     #   Bounds cost and low-passes sticker-print texture.
CENTRE_FRAC  = 0.80  # the detector box is axis-aligned and the cube projects
                     #   to a hexagon, so the corners are background. Shrink.
                     #   MEASURED: tightening this HURTS (see the docstring's
                     #   negative results) — it removes cube and background in
                     #   equal proportion. 0.80 is the measured best of
                     #   {0.45, 0.55, 0.65, 0.80}.
MASK         = "box" # region mask inside the crop. Also measured, also a
                     #   negative result: `hex` and `ellipse` are worse than
                     #   no mask at all. Kept selectable so the finding stays
                     #   reproducible rather than becoming folklore.
CLOSE_KSIZE  = 11    # morphological CLOSE: bridges the dark sticker borders
                     #   so a solid face is one component. See frame_stats.
MIN_REGION   = 0.02  # a connected component below this share of classified
                     #   area is speckle, not a face
MIN_CLASS    = 0.04  # ditto for a whole class's share
MIN_COVERAGE = 0.35  # classified area below this share of the crop means the
                     #   box is mostly not cube -> the frame is unreadable
QUANTILE     = 0.10  # window aggregation: see window_stats() for why a LOW
                     #   quantile is the correct statistic here and a median
                     #   is not

# The shipped operating point. `n_regions` is the statistic, NOT the
# lower-variance-looking `fragmentation` — see the module docstring's
# "Which statistic ships" note. A cube showing more than this many distinct
# solid regions is not solved.
SOLVED_MAX_REGIONS = 4


def axis_map(bgr, palette="axis", s_min=None):
    """BGR crop -> (label image, coverage). Label 0 = unassigned.

    `s_min` is the saturation floor that separates a cube STICKER from the
    scene it is held in. It is the single most load-bearing constant here:
    set too low, the desk and the solver's hands classify as warm, merge with
    the cube's red/orange stickers into one enormous blob, and a scrambled
    cube reads as perfectly solved. Measured, not guessed — see the
    `sweep` subcommand.
    """
    s_min = S_MIN if s_min is None else s_min
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)

    bright = v >= V_MIN
    chroma = bright & (s >= s_min)
    white  = bright & (s < S_WHITE) & (v >= V_WHITE)

    warm   = chroma & ((h <= H_WARM_MAX) | (h >= H_WRAP))
    yellow = chroma & (h > H_WARM_MAX) & (h <= H_YELLOW_MAX)
    cool   = chroma & (h > H_YELLOW_MAX) & (h < H_WRAP)

    lab = np.zeros(h.shape, np.uint8)
    if palette == "axis":
        lab[warm] = 1
        lab[yellow] = 2
        lab[white] = 2
        lab[cool] = 3
    else:
        # red 1 | orange 2 | yellow 3 | white 4 | green 5 | blue 6
        lab[warm & ((h <= H_RED_MAX) | (h >= H_WRAP))] = 1
        lab[warm & (h > H_RED_MAX) & (h < H_WRAP)] = 2
        lab[yellow] = 3
        lab[white] = 4
        lab[cool & (h <= H_GREEN_MAX)] = 5
        lab[cool & (h > H_GREEN_MAX)] = 6

    coverage = float((lab > 0).sum()) / lab.size
    return lab, coverage


_SIL_CACHE = {}


def _silhouette(shape, kind="hex"):
    """Binary mask keeping the cube and dropping the box CORNERS.

    A cube viewed corner-on projects to a hexagon inscribed in its own
    bounding box: full width at the middle, empty at the four corners. That is
    precisely where the background lives, and background is the failure mode
    that matters — a warm-lit wall classifies as one enormous solid region and
    a scrambled cube reads as solved.

    Shrinking a centred SQUARE (the obvious fix) was measured and is worse: it
    removes cube and background in equal proportion, compressing both sides of
    the statistic and adding noise. Cutting the corners specifically keeps the
    whole cube.
    """
    key = (shape, kind)
    m = _SIL_CACHE.get(key)
    if m is not None:
        return m
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx - (w - 1) / 2) / max((w - 1) / 2, 1)
    v = (yy - (h - 1) / 2) / max((h - 1) / 2, 1)
    if kind == "ellipse":
        m = (u * u + v * v) <= 1.0
    else:  # hexagon: a diamond relaxed toward the box
        m = (np.abs(u) + np.abs(v)) <= 1.5
    m = m.astype(np.uint8)
    _SIL_CACHE[key] = m
    return m


def frame_stats(frame, box, palette="axis", s_min=None, close=None,
                centre=None, mask=MASK):
    """Fragmentation statistics for one frame's cube box.

    Returns None when the crop is unusable (off-frame, degenerate, or mostly
    not cube) — an unreadable frame must not contribute a number.
    """
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 12:
        return None
    # Shrink to the centre. This is not cosmetic trimming: the detector box
    # is AXIS-ALIGNED around a cube that projects to a hexagon, so a large
    # fraction of it is always background, and background cannot be rejected
    # by colour — a beige wall under warm light passes the white-sticker test
    # (low saturation, high value) and classifies as a single enormous solid
    # region, which reads as a perfectly solved cube. Geometry is the only
    # thing that separates cube from scene here.
    #
    # Viewed corner-on the three visible faces MEET at the cube's centre, so
    # the central region still spans all of them and the statistic loses
    # nothing by being computed there.
    centre = CENTRE_FRAC if centre is None else centre
    dx, dy = int(w * (1 - centre) / 2), int(h * (1 - centre) / 2)
    x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    crop = frame[y1:y2, x1:x2]
    scale = WORK_LONG / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    lab, coverage = axis_map(crop, palette, s_min)
    if mask != "box":
        lab = lab * _silhouette(lab.shape, mask)
        coverage = float((lab > 0).sum()) / lab.size
    if coverage < MIN_COVERAGE:
        return None

    close = CLOSE_KSIZE if close is None else close
    n_cls = N_CLASSES if palette == "axis" else 6
    total = 0
    largest_sum = 0
    largest_any = 0
    regions = 0
    classes = 0
    per_class = []

    # CLOSE, not open. A cube's stickers are separated by wide dark borders
    # that classify as unassigned, so on a SOLID face every sticker is its own
    # connected component and a solved face reads as nine fragments. Closing
    # bridges the borders so a face becomes one blob, which is the entire
    # premise of the statistic. Opening — the obvious speckle filter — widens
    # exactly the gaps that must be bridged and is precisely wrong here.
    kern = np.ones((close, close), np.uint8)
    masks = []
    for c in range(1, n_cls + 1):
        m = cv2.morphologyEx((lab == c).astype(np.uint8), cv2.MORPH_CLOSE, kern)
        masks.append(m)
        total += int(m.sum())
    if total == 0:
        return None

    for m in masks:
        area = int(m.sum())
        per_class.append(area / total)
        if area == 0:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        # label 0 is background
        areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)),
                       reverse=True)
        if not areas:
            continue
        largest_sum += areas[0]
        largest_any = max(largest_any, areas[0])
        regions += sum(1 for a in areas if a / total >= MIN_REGION)
        if area / total >= MIN_CLASS:
            classes += 1

    return {
        "fragmentation": 1.0 - largest_sum / total,
        # `max_blob` is the complementary read: the single largest solid
        # region as a share of the cube. A solved cube seen corner-on is
        # three faces at ~1/3 each; a scrambled one's biggest solid patch is
        # a sticker or two. Kept alongside fragmentation because the two fail
        # differently — fragmentation is fooled by a class that happens to
        # land in one clump, max_blob by a face-on view of one solved face.
        "max_blob": largest_any / total,
        "n_regions": regions,
        "n_classes": classes,
        "coverage": coverage,
        "class_share": per_class,
    }


def window_stats(frames_and_boxes, palette="axis", min_frames=5,
                 quantile=QUANTILE, s_min=None, close=None, centre=None,
                 mask=MASK):
    """Low-quantile statistics over a window of (frame, box) pairs.

    A LOW QUANTILE, not a median, and the reason is an asymmetry rather than
    a preference. Every error mode available to this measurement pushes
    fragmentation in the SAME direction:

      * a hand across a face splits one blob into two (and skin is warm-hued,
        so it also adds area to the warm class);
      * motion blur smears sticker borders into unassigned pixels;
      * a misclassified pixel run carves a hole in an otherwise solid face.

    Nothing pushes it down except the cube actually being solved. So an
    occluded or blurred frame is UNREADABLE, not incriminating, and a solved
    cube needs only one clean look inside the window to prove it — whereas a
    scrambled cube has no clean look to offer, because its fragmentation is
    structural rather than incidental.

    Taking a quantile rather than the outright minimum keeps one freak frame
    (a detection box that clipped to a single solid face, say) from deciding
    the answer on its own.
    """
    rows = []
    for frame, box in frames_and_boxes:
        st = frame_stats(frame, box, palette, s_min, close, centre, mask)
        if st is not None:
            rows.append(st)
    if len(rows) < min_frames:
        return None

    def q(key, hi=False):
        # `max_blob` runs the other way — occlusion SHRINKS the largest solid
        # region — so its clean-look quantile is the upper one.
        return float(np.quantile([r[key] for r in rows],
                                 1.0 - quantile if hi else quantile))

    return {
        "fragmentation": q("fragmentation"),
        "max_blob": q("max_blob", hi=True),
        "n_regions": q("n_regions"),
        "n_classes": float(np.median([r["n_classes"] for r in rows])),
        "coverage": float(np.median([r["coverage"] for r in rows])),
        "frag_median": float(np.median([r["fragmentation"] for r in rows])),
        "n_frames": len(rows),
    }


def solved_at(frames_and_boxes, palette="axis"):
    """The shipped call: was the cube in this window of frames SOLVED?

    Feed it the frames straddling the timer stop. Returns a plain dict so
    `anticheat_gate.adjudicate` stays a pure function over plain data and the
    server can re-run it on a stored bundle.

    Three outcomes, and the third is not a formality:

      solved=True    every clean look shows <= SOLVED_MAX_REGIONS solid
                     regions
      solved=False   no clean look does -> the cube on camera was not solved
                     when the timer stopped
      solved=None    UNREADABLE: too few usable frames (the cube was
                     occluded, out of frame, or the box was mostly not cube).
                     Must abstain, never reject. Occlusion is the single most
                     common thing at a timer stop — hands are still on the
                     cube — and it pushes the statistic toward "scrambled",
                     so treating unreadable as incriminating would false-DQ
                     exactly the honest solves that end in a tight grip.
    """
    st = window_stats(frames_and_boxes, palette)
    if st is None:
        return {"solved": None, "reason": "unreadable", "n_regions": None,
                "n_frames": 0}
    return {
        "solved": st["n_regions"] <= SOLVED_MAX_REGIONS,
        "reason": None,
        "n_regions": st["n_regions"],
        "threshold": SOLVED_MAX_REGIONS,
        "fragmentation": st["fragmentation"],
        "max_blob": st["max_blob"],
        "coverage": st["coverage"],
        "n_frames": st["n_frames"],
    }


# --------------------------------------------------------------------------
# Corpus measurement
# --------------------------------------------------------------------------

def _load_session(session_dir):
    p = os.path.join(session_dir, "trajectory.npz")
    if not os.path.isfile(p):
        return None
    d = np.load(p, allow_pickle=True)
    by_frame = {}
    for fi, t, b in zip(d["frame_idx"], d["t"], d["boxes"]):
        fi = int(fi)
        # keep the highest-confidence box per frame
        prev = by_frame.get(fi)
        if prev is None or b[4] > prev[1][4]:
            by_frame[fi] = (float(t), tuple(float(x) for x in b))
    entries = []
    with open(os.path.join(session_dir, "frames.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda r: r["ts"])
    return by_frame, entries


def tail_window(session_dir, seconds, skip_last, palette="axis",
                s_min=None, close=None, centre=None, mask=MASK,
                cache=None):
    """Statistics over a window ending `skip_last` seconds before the last
    detected frame.

    `skip_last` matters: the final moments of a recording are the hand
    reaching for the keyboard to stop it, which occludes the cube. The window
    is defined in TIME, so a variable capture rate cannot distort it.
    """
    loaded = _load_session(session_dir)
    if loaded is None:
        return None
    by_frame, entries = loaded
    if not by_frame:
        return None
    t_last = max(t for t, _ in by_frame.values())
    hi = t_last - skip_last
    lo = hi - seconds

    if cache is not None and session_dir in cache:
        pairs = cache[session_dir]
    else:
        pairs = []
        for idx, (t, box) in sorted(by_frame.items()):
            if lo <= t <= hi and idx < len(entries):
                path = os.path.join(session_dir, "frames", entries[idx]["file"])
                frame = cv2.imread(path)
                if frame is not None:
                    pairs.append((frame, box))
        if cache is not None:
            cache[session_dir] = pairs
    return window_stats(pairs, palette, s_min=s_min, close=close,
                        centre=centre, mask=mask)


def discover(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "trajectory.npz")))


def cmd_score(args):
    sessions = discover(args.root)
    if not sessions:
        raise SystemExit(f"no sessions with trajectory.npz under {args.root} — "
                         "run dump_trajectories.py first")
    if args.limit:
        sessions = sessions[:args.limit]

    t0 = time.time()
    rows = []
    for i, name in enumerate(sessions):
        st = tail_window(os.path.join(args.root, name),
                         args.window, args.skip_last, args.palette,
                         centre=args.centre, mask=args.mask)
        # A `_scramble` session ends scrambled; a `_solve` session ends
        # solved. Anything unlabelled is scored but excluded from the split.
        if name.endswith("_scramble"):
            truth = "scrambled"
        elif name.endswith("_solve"):
            truth = "solved"
        else:
            truth = None
        rows.append((name, truth, st))
        if st is None:
            print(f"[{i+1}/{len(sessions)}] {name}: unscoreable")
        else:
            print(f"[{i+1}/{len(sessions)}] {name}: frag={st['fragmentation']:.3f} "
                  f"regions={st['n_regions']:.0f} classes={st['n_classes']:.0f} "
                  f"cov={st['coverage']:.2f} n={st['n_frames']}"
                  + (f"  [{truth}]" if truth else ""))

    print(f"\n{time.time() - t0:.0f}s\n")
    _report(rows, args)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "palette": args.palette,
                "window_s": args.window,
                "skip_last_s": args.skip_last,
                "rows": [{"session": n, "truth": t, **(s or {})}
                         for n, t, s in rows],
            }, f, indent=2)
        print(f"\nwrote {args.out}")


def _report(rows, args):
    solved = [s for _, t, s in rows if t == "solved" and s]
    scram = [s for _, t, s in rows if t == "scrambled" and s]
    if not solved or not scram:
        print("no labelled split available")
        return

    print(f"{len(solved)} solved-ending vs {len(scram)} scrambled-ending "
          f"sessions   (palette={args.palette}, window={args.window}s, "
          f"skip_last={args.skip_last}s)\n")
    print(f"  {'statistic':<16} {'solved med':>11} {'solved max':>11} "
          f"{'scram med':>10} {'scram min':>10} {'gap':>7}")
    for key in ("fragmentation", "n_regions", "n_classes"):
        a = np.array([s[key] for s in solved])
        b = np.array([s[key] for s in scram])
        gap = b.min() - a.max()
        print(f"  {key:<16} {np.median(a):>11.3f} {a.max():>11.3f} "
              f"{np.median(b):>10.3f} {b.min():>10.3f} {gap:>7.3f}")

    # The operating point that matters: a threshold that never false-DQs a
    # legit solve, and what it then catches. Reported in that order, because
    # the false-DQ side is the one with no acceptable failure.
    print("\n  zero-false-DQ operating points (threshold strictly above the "
          "worst legit solve):")
    for key in ("fragmentation", "n_regions"):
        a = np.array([s[key] for s in solved])
        b = np.array([s[key] for s in scram])
        thr = float(a.max())
        caught = int((b > thr).sum())
        print(f"    {key:<16} thr>{thr:.3f}  catches {caught}/{len(b)} "
              f"({100*caught/len(b):.0f}%)")
        worst = sorted(zip(a, [n for n, t, s in rows if t == 'solved' and s]),
                       reverse=True)[:3]
        print(f"      worst legit: "
              + ", ".join(f"{v:.3f} {n}" for v, n in worst))
        missed = sorted(zip(b, [n for n, t, s in rows if t == 'scrambled' and s]))[:3]
        print(f"      least-scrambled attack: "
              + ", ".join(f"{v:.3f} {n}" for v, n in missed))


def _separation(rows, key, higher_is_scrambled=True):
    """(threshold, catch rate) at the zero-false-DQ operating point.

    Reported this way round on purpose: the threshold is pinned by the WORST
    legitimate solve, and the catch rate is whatever falls out. A gate that
    must never false-DQ has no freedom to trade the other way.
    """
    solved = [s[key] for _, t, s in rows if t == "solved" and s]
    scram = [s[key] for _, t, s in rows if t == "scrambled" and s]
    if not solved or not scram:
        return None
    if higher_is_scrambled:
        thr = max(solved)
        caught = sum(1 for v in scram if v > thr)
    else:
        thr = min(solved)
        caught = sum(1 for v in scram if v < thr)
    return thr, caught / len(scram), len(scram)


def _holdout(rows, key, cut, higher_is_scrambled=True):
    """Calibrate the threshold on sessions BEFORE `cut`, score those after.

    Split by DATE, never at random. Session names are date-prefixed so
    sorting is chronological; a random split would put the same sitting's
    solves on both sides and report a same-day number as if it generalised
    (the lesson from the classifier's holdout ladder — 98% on a same-day
    split, 77% across environments).

    Returns (threshold, catch rate, n_attacks, false-DQ rate, n_legit) on the
    TEST half, or None when either half is empty.
    """
    tr_s = [s[key] for n, t, s in rows if s and t == "solved" and n < cut]
    te_s = [s[key] for n, t, s in rows if s and t == "solved" and n >= cut]
    te_a = [s[key] for n, t, s in rows if s and t == "scrambled" and n >= cut]
    if not tr_s or not te_s or not te_a:
        return None
    if higher_is_scrambled:
        thr = max(tr_s)
        caught = sum(1 for v in te_a if v > thr)
        false_dq = sum(1 for v in te_s if v > thr)
    else:
        thr = min(tr_s)
        caught = sum(1 for v in te_a if v < thr)
        false_dq = sum(1 for v in te_s if v < thr)
    return thr, caught / len(te_a), len(te_a), false_dq / len(te_s), len(te_s)


def cmd_sweep(args):
    """Set S_MIN and CLOSE_KSIZE by measurement rather than by eye.

    Both were found by looking at label maps (the desk classifying as warm;
    solid faces splitting at the sticker borders), which is enough to know
    the direction and not enough to pick a value.
    """
    sessions = [n for n in discover(args.root)
                if n.endswith("_solve") or n.endswith("_scramble")]
    if args.limit:
        sessions = sessions[:args.limit]
    cache = {}

    s_mins = [int(v) for v in args.s_min.split(",")]
    closes = [int(v) for v in args.close.split(",")]
    print(f"centre crop {args.centre}, mask {args.mask}")
    print(f"{len(sessions)} labelled sessions, "
          f"{len(s_mins)}x{len(closes)} settings\n")
    keys = (("fragmentation", True), ("max_blob", False), ("n_regions", True))
    print(f"  {'s_min':>6} {'close':>6} | " + " | ".join(
        f"{k[:9]:>9} in/out {'fDQ':>4}" for k, _ in keys))

    best = None
    for s_min in s_mins:
        for close in closes:
            rows = []
            for name in sessions:
                st = tail_window(os.path.join(args.root, name), args.window,
                                 args.skip_last, args.palette, s_min, close,
                                 args.centre, args.mask, cache)
                truth = "scrambled" if name.endswith("_scramble") else "solved"
                rows.append((name, truth, st))
            cells = []
            for key, hi in keys:
                ins = _separation(rows, key, hi)
                out = _holdout(rows, key, args.cut, hi)
                if not ins or not out:
                    cells.append(f"{'-':>20}")
                    continue
                cells.append(f"{ins[1]:>9.0%}{out[1]:>7.0%}{out[3]:>5.0%}")
                # Rank on the HOLDOUT catch rate, and only among settings
                # that hold the false-DQ line out of sample. An in-sample
                # peak here is the threshold memorising its worst session.
                if out[3] == 0 and (best is None or out[1] > best[0]):
                    best = (out[1], key, s_min, close, out[0], ins[1])
            print(f"  {s_min:>6} {close:>6} | " + " | ".join(cells))

    print(f"\n  in = in-sample catch (threshold from ALL legit solves)\n"
          f"  out = held-out catch, fDQ = held-out false-DQ "
          f"(threshold from sessions before {args.cut})")
    if best:
        print(f"\n  best HELD-OUT catch at zero held-out false-DQ: "
              f"{best[0]:.0%} on {best[1]} with s_min={best[2]} "
              f"close={best[3]} (threshold {best[4]:.3f}, in-sample "
              f"{best[5]:.0%})")
    else:
        print("\n  NO setting held the false-DQ line out of sample.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--root", required=True)
    s.add_argument("--palette", choices=("axis", "full"), default="axis")
    s.add_argument("--window", type=float, default=1.5,
                   help="seconds of tail to aggregate over")
    s.add_argument("--skip-last", type=float, default=0.5, dest="skip_last",
                   help="seconds to skip at the very end (hand on keyboard)")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--centre", type=float, default=CENTRE_FRAC)
    s.add_argument("--mask", choices=("hex", "ellipse", "box"), default=MASK)
    s.add_argument("--out", default=None)

    w = sub.add_parser("sweep")
    w.add_argument("--root", required=True)
    w.add_argument("--palette", choices=("axis", "full"), default="axis")
    w.add_argument("--window", type=float, default=1.5)
    w.add_argument("--skip-last", type=float, default=0.5, dest="skip_last")
    w.add_argument("--s-min", default="70,100,120,140,160", dest="s_min")
    w.add_argument("--close", default="3,7,11")
    w.add_argument("--centre", type=float, default=CENTRE_FRAC)
    w.add_argument("--mask", choices=("hex", "ellipse", "box"), default=MASK)
    w.add_argument("--cut", default="solve_20260731",
                   help="date-split: sessions naming-sorted below this "
                        "calibrate the threshold, the rest are the holdout")
    w.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()
    {"score": cmd_score, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
