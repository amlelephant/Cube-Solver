"""
common.py — shared provenance + scoring helpers for every paper measurement.

The single rule this file exists to enforce: **no number in the paper may
come from a session any evaluated checkpoint has seen.** `holdout()` derives
the evaluable set from the checkpoints' OWN recorded `train_session_names`
and `val_session_names`, intersected across every checkpoint passed, rather
than from a hand-maintained list. A hand-maintained list is exactly how this
project previously reported memorisation numbers as generalisation
(see move_detector/GAMEPLAN.md §4b).

Second rule, less obvious and it silently corrupted every existing eval
harness in the repo: the ground truth for a VISION model is the move as the
CAMERA saw it (`camera_notation`), not as the cube's core reported it
(`wca_notation`). A middle slice rotates the cube's internal frame away from
the camera, so after an `M` every BLE label names the wrong face. 7 of 88
sessions contain slices. `truth_word()` prefers `camera_notation`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
MD = REPO / "ble" / "move_detector"
SESSIONS = REPO / "ble" / "training_data"
DATA = Path(__file__).resolve().parents[1] / "data"
FIGS = Path(__file__).resolve().parents[1] / "figures"

# move_detector modules import each other by bare name and expect cwd == MD.
sys.path.insert(0, str(MD))

DAY_START, DAY_END = 9, 18   # "daytime" as eval_lighting.py defines it


# ---------------------------------------------------------------- provenance

def ckpt_seen(paths) -> set[str]:
    """Union of every session name any of these checkpoints trained OR
    validated on. Validation sessions count as seen: they picked the epoch."""
    seen: set[str] = set()
    for p in paths:
        c = torch.load(p, map_location="cpu", weights_only=False)
        seen |= set(c.get("train_session_names") or [])
        seen |= set(c.get("val_session_names") or [])
    return seen


def holdout(paths, kind: str | None = None) -> list[Path]:
    """Prepared sessions none of `paths` has seen, in name order.

    kind: None = all, "solve" = free solves only, "scramble" = prescribed
    scrambles only. The two are different regimes — a scramble is 20 slow
    prescribed moves with known truth, a solve is 80-150 fast ones — and
    pooling them flatters the mean.
    """
    seen = ckpt_seen(paths)
    out = []
    for npz in sorted(SESSIONS.glob("solve_*/detector_stream_color.npz")):
        d = npz.parent
        if d.name in seen or not (d / "moves.jsonl").exists():
            continue
        if kind == "solve" and not d.name.endswith("_solve"):
            continue
        if kind == "scramble" and not d.name.endswith("_scramble"):
            continue
        out.append(d)
    return out


# ------------------------------------------------------------------- truth

def truth_word(d: Path) -> list[str] | None:
    """Ground-truth move word in the CAMERA frame. None if unresolvable."""
    from reconstruct import WCA12
    recs = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    w = [r.get("camera_notation") or r.get("wca_notation") for r in recs]
    if not w or any(x is None or x not in WCA12 for x in w):
        return None
    return w


def cube_word(d: Path) -> list[str] | None:
    """Cube-frame word — what reconstruct.start_from_gt needs to build the
    start state, since its cube model is centre-relative."""
    from reconstruct import WCA12
    recs = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    w = [r.get("wca_notation") for r in recs]
    if not w or any(x is None or x not in WCA12 for x in w):
        return None
    return w


def session_meta(d: Path) -> dict:
    """Capture-side facts about a take, independent of any model."""
    fj = d / "frames.jsonl"
    ts0 = hour = None
    if fj.exists():
        with open(fj) as fh:
            ts0 = json.loads(fh.readline())["ts"]
        hour = time.localtime(ts0).tm_hour
    data = np.load(d / "detector_stream_color.npz", allow_pickle=True)
    onset = data["onset_idx"].astype(int)
    fps = float(data["fps"])
    n_frames = int(data["frames"].shape[0])
    gaps = np.diff(onset) if len(onset) > 1 else np.array([np.inf])
    dur = (onset[-1] - onset[0]) / fps if len(onset) > 1 else float("nan")
    cls = data["onset_class"].astype(int)
    same_close = int(sum(1 for i in range(len(onset) - 1)
                         if gaps[i] <= 2 and cls[i] == cls[i + 1]))
    return {
        "session": d.name,
        "hour": hour,
        "evening": None if hour is None else not (DAY_START <= hour < DAY_END),
        "fps": fps,
        "n_frames": n_frames,
        "n_moves": int(len(onset)),
        "tps": float(len(onset) - 1) / dur if dur and dur == dur else None,
        "crowded_frac": float((gaps <= 2).mean()) if len(onset) > 1 else 0.0,
        "ctc_floor": same_close / max(len(onset), 1),
        "crop_mode": str(data["crop_mode"]) if "crop_mode" in data else "?",
        "label_source": (str(data["label_source"])
                         if "label_source" in data else "ble"),
    }


# ------------------------------------------------------------------ scoring

def channel_split(gt: list[str], pred: list[str]) -> dict:
    """ok / sub / miss / phantom via Needleman-Wunsch, plus per-move accuracy.

    Accuracy is ok/|gt|: a phantom does not directly reduce it, which is why
    the phantom count is reported alongside rather than folded in. MER (the
    edit distance over |gt|) charges phantoms and is reported too.
    """
    from decode import align_sequences
    ops = align_sequences(gt, pred)
    c = {k: sum(1 for o, _, _ in ops if o == k)
         for k in ("ok", "sub", "miss", "phantom")}
    c["n_gt"] = len(gt)
    c["n_pred"] = len(pred)
    c["acc"] = c["ok"] / max(len(gt), 1)
    return c


def mer(gt: list[str], pred: list[str]) -> tuple[float, dict]:
    from ctc_decode import move_error_rate
    from reconstruct import WCA12
    return move_error_rate([WCA12.index(p) for p in pred],
                           [WCA12.index(g) for g in gt])


def sub_kind(gt_move: str, pred_move: str) -> str:
    """Classify a substitution: same face wrong direction ('inverse'),
    an adjacent face, or the opposite face. This is the error-axis question
    the repo's earlier notes kept re-litigating; it is cheap to just measure."""
    OPP = {"U": "D", "D": "U", "L": "R", "R": "L", "F": "B", "B": "F"}
    gf, pf = gt_move[0], pred_move[0]
    if gf == pf:
        return "inverse"
    if OPP[gf] == pf:
        return "opposite"
    return "adjacent"


# ------------------------------------------------------------------- output

def dump(name: str, obj) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / name
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"  wrote {p.relative_to(REPO)}")
    return p


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
