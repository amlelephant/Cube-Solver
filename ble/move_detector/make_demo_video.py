"""
make_demo_video.py — render a held-out solve into an annotated demo clip.

Picks a session no evaluated checkpoint has trained on (see
paper/data/holdout_meta.json / paper/README.md's "one rule"), decodes it
with the cached posteriorgram already computed for the paper
(paper/data/post/), and burns a live move-recognition log into a dead
region of the frame (below the cube, over the static laptop lid — verified
by eye across the session, never occluded by hands or the cube itself).

Session choice: solve_20260805_155829_solve. Not cherry-picked — it is the
*median*-accuracy daytime solve among the paper's held-out solves (see
paper/data/tab_persession.tex), so the clip is a representative outcome,
not the best one. Daytime specifically, per instruction, as the fairest
lighting condition (evening sessions run ~15-20pts lower — see
paper/README and results/2026-08-05 lighting_check.py).

Each log line's TRUE move is timed to the smart cube's own capture-clock
timestamp for that move (ground truth, never an input to the model); its
PRED column is timed to the frame the decoder actually emitted that class
at. A PHANTOM line (predicted move with no matching truth) is timed to the
model's own onset frame and flagged SPURIOUS — this is a false positive,
not a relabeling of a real move. The end-of-clip summary card is the exact
ok/sub/miss/phantom breakdown from decode.align_sequences, the same
function and the same numbers paper/scripts/m1_recognition.py reports
(cross-checked against paper/data/m1_recognition.json below).

This is the RAW per-move recognition stage — what the ticker visualises
move-by-move — not the group-theoretic beam-search reconstruction
(reconstruct.py) that runs after it and can still fix some of what's
flagged wrong here. Said explicitly on the summary card so the clip
doesn't overclaim relative to paper/.

    cd ble/move_detector
    ../../.venv/Scripts/python.exe make_demo_video.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ctc_decode import prefix_beam_decode, ctc_to_moves
from decode import align_sequences

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SESSION = "solve_20260805_155829_solve"
MODEL_TAG = "move_ctc_spd_s0"
SESSION_DIR = REPO / "ble" / "training_data" / SESSION
POST_CACHE = REPO / "paper" / "data" / "post" / f"{MODEL_TAG}__{SESSION}.npz"
M1_JSON = REPO / "paper" / "data" / "m1_recognition.json"
OUT_PATH = REPO / "media" / "demo_solve_verification.mp4"
BEAM = 16

# Dead zone: static laptop lid below the cube/hands, whole session (checked
# frames 0, 300, 700, 899, 1399 by eye). Frame is 1280x720. Right edge
# avoided — a small RGB keyboard glows there.
BOX = (24, 452, 1040, 700)  # x0, y0, x1, y1
N_LOG_LINES = 6
FLASH_FRAMES = 12

COLORS = {
    "ok": (90, 210, 90),
    "sub": (0, 165, 255),
    "miss": (160, 160, 160),
    "phantom": (50, 50, 235),
}
WHITE = (235, 235, 235)
DIM = (150, 150, 150)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_events():
    frame_recs = sorted(load_jsonl(SESSION_DIR / "frames.jsonl"), key=lambda r: r["idx"])
    ts_array = np.array([r["ts"] for r in frame_recs], dtype=np.float64)
    files = [r["file"] for r in frame_recs]
    fps = float(len(frame_recs) - 1) / (ts_array[-1] - ts_array[0])

    moves = load_jsonl(SESSION_DIR / "moves.jsonl")
    gt_names = [m.get("camera_notation") or m["wca_notation"] for m in moves]
    gt_ts = [m["timestamp"] for m in moves]

    class_prob = np.load(POST_CACHE)["class_prob"]
    assert class_prob.shape[0] == len(frame_recs), (
        f"posteriorgram has {class_prob.shape[0]} rows but session has "
        f"{len(frame_recs)} frames — cache/session mismatch")
    lp = np.log(np.maximum(class_prob, 1e-12))
    labels, frames_idx = prefix_beam_decode(lp, beam=BEAM)
    pred_moves = ctc_to_moves(class_prob, labels, frames_idx, fps=fps,
                              frame_times=ts_array)
    pred_names = [m["move"] for m in pred_moves]

    ref = next((r for r in json.loads(M1_JSON.read_text(encoding="utf-8"))
               if r["model"] == MODEL_TAG and r["session"] == SESSION), None)
    if ref is not None and ref["pred"] != pred_names:
        print("  WARNING: local beam decode does not match the cached "
              "paper/data/m1_recognition.json prediction — numbers below "
              "are the local decode, not the paper's.")

    ops = align_sequences(gt_names, pred_names)

    gt_i = pred_i = 0
    events = []
    for op, w, p in ops:
        ev = {"op": op}
        if op in ("ok", "sub", "miss"):
            ev["gt_move"] = gt_names[gt_i]
            frame = int(np.searchsorted(ts_array, gt_ts[gt_i]))
            ev["gt_frame"] = min(frame, len(frame_recs) - 1)
            gt_i += 1
        if op in ("ok", "sub", "phantom"):
            pm = pred_moves[pred_i]
            ev["pred_move"] = pm["move"]
            ev["pred_frame"] = pm["frame"]
            pred_i += 1
        ev["trigger_frame"] = ev["gt_frame"] if "gt_frame" in ev else ev["pred_frame"]
        events.append(ev)
    assert gt_i == len(gt_names) and pred_i == len(pred_names)
    events.sort(key=lambda e: e["trigger_frame"])

    counts = {"ok": 0, "sub": 0, "miss": 0, "phantom": 0}
    for e in events:
        counts[e["op"]] += 1
    n_gt, n_pred = len(gt_names), len(pred_names)
    acc = counts["ok"] / n_gt
    from decode import align_sequences as _al  # local import mirrors common.mer
    dist = counts["sub"] + counts["miss"] + counts["phantom"]
    mer = dist / n_gt

    return frame_recs, files, events, {
        "n_gt": n_gt, "n_pred": n_pred, "acc": acc, "mer": mer, **counts,
    }


def fmt_line(ev: dict) -> tuple[str, tuple]:
    op = ev["op"]
    t = ev.get("gt_frame")
    tstr = f"t={ev['_t']:5.1f}s" if "_t" in ev else "        "
    true_s = ev.get("gt_move", "-")
    pred_s = ev.get("pred_move", "-")
    tag = {"ok": "OK", "sub": "SUB (wrong)", "miss": "MISS (no call)",
           "phantom": "SPURIOUS"}[op]
    text = f"{tstr}  true:{true_s:<3s}  pred:{pred_s:<3s}  {tag}"
    return text, COLORS[op]


def draw_box(img, active_events, counts_so_far, n_gt_seen, n_gt_total,
            flash_left):
    x0, y0, x1, y1 = BOX
    overlay = img.copy()
    border = COLORS["phantom"] if flash_left > 0 else (70, 70, 70)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x1, y1), border, 2)

    seen_ok = counts_so_far["ok"]
    acc_txt = f"{100 * seen_ok / n_gt_seen:5.1f}%" if n_gt_seen else "  -  "
    header = (f"MOVE RECOGNITION (raw, live)   moves:{n_gt_seen:>3}/{n_gt_total}"
              f"   acc:{acc_txt}   sub:{counts_so_far['sub']}"
              f"  miss:{counts_so_far['miss']}  spurious:{counts_so_far['phantom']}")
    cv2.putText(img, header, (x0 + 14, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX,
               0.52, WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (x0 + 10, y0 + 34), (x1 - 10, y0 + 34), (70, 70, 70), 1)

    row_h = (y1 - y0 - 44) // N_LOG_LINES
    for i, ev in enumerate(active_events[-N_LOG_LINES:]):
        text, color = fmt_line(ev)
        y = y0 + 44 + i * row_h + int(row_h * 0.7)
        cv2.putText(img, text, (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1, cv2.LINE_AA)


def summary_card(stats: dict) -> np.ndarray:
    img = np.full((720, 1280, 3), (24, 20, 18), dtype=np.uint8)
    cx = 640

    def center(text, y, scale, color, thick=1):
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, (cx - w // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                   scale, color, thick, cv2.LINE_AA)

    center("SESSION COMPLETE", 130, 1.3, (255, 255, 255), 2)
    center("held-out session (never used in training) - daytime lighting",
          170, 0.55, DIM)
    center(f"session: {SESSION}   model: {MODEL_TAG}   beam width: {BEAM}",
          198, 0.5, DIM)

    rows = [
        (f"Ground-truth moves: {stats['n_gt']}", WHITE),
        (f"Predicted calls: {stats['n_pred']}", WHITE),
        ("", WHITE),
        (f"Correct (OK): {stats['ok']}", COLORS["ok"]),
        (f"Substitutions (wrong call): {stats['sub']}", COLORS["sub"]),
        (f"Missed (no call): {stats['miss']}", COLORS["miss"]),
        (f"Spurious (phantom call): {stats['phantom']}", COLORS["phantom"]),
        ("", WHITE),
        (f"Raw per-move accuracy: {stats['acc'] * 100:.1f}%", (255, 255, 255)),
        (f"Move error rate: {stats['mer'] * 100:.1f}%", (255, 255, 255)),
    ]
    y = 270
    for text, color in rows:
        if text:
            center(text, y, 0.75, color, 2)
        y += 42

    center("raw recognition stage - before group-theoretic state reconstruction",
          y + 20, 0.48, DIM)
    return img


def main():
    frame_recs, files, events, stats = build_events()
    for e in events:
        if "gt_frame" in e:
            e["_t"] = frame_recs[e["gt_frame"]]["ts"] - frame_recs[0]["ts"]
        else:
            e["_t"] = frame_recs[e["pred_frame"]]["ts"] - frame_recs[0]["ts"]

    print(f"  {SESSION}: {stats['n_gt']} gt moves, {stats['n_pred']} pred calls")
    print(f"  ok {stats['ok']}  sub {stats['sub']}  miss {stats['miss']}  "
          f"phantom {stats['phantom']}  acc {stats['acc']*100:.1f}%  "
          f"mer {stats['mer']*100:.1f}%")

    ts_array = np.array([r["ts"] for r in frame_recs])
    fps = float(len(frame_recs) - 1) / (ts_array[-1] - ts_array[0])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (1280, 720))

    by_frame: dict[int, list[dict]] = {}
    for e in events:
        by_frame.setdefault(e["trigger_frame"], []).append(e)

    active: list[dict] = []
    counts = {"ok": 0, "sub": 0, "miss": 0, "phantom": 0}
    n_gt_seen = 0
    flash_left = 0

    for i, rec in enumerate(frame_recs):
        img = cv2.imread(str(SESSION_DIR / "frames" / files[i]))
        if img is None:
            raise SystemExit(f"missing frame {files[i]}")
        for ev in by_frame.get(i, []):
            active.append(ev)
            counts[ev["op"]] += 1
            if "gt_frame" in ev:
                n_gt_seen += 1
            if ev["op"] == "phantom":
                flash_left = FLASH_FRAMES
        draw_box(img, active, counts, n_gt_seen, stats["n_gt"], flash_left)
        if flash_left > 0:
            flash_left -= 1
        vw.write(img)
        if i % 200 == 0:
            print(f"  frame {i}/{len(frame_recs)}")

    card = summary_card(stats)
    for _ in range(int(fps * 5)):
        vw.write(card)

    vw.release()
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  -> {OUT_PATH}  ({size_mb:.1f} MB, {len(frame_recs)} frames "
          f"+ {int(fps*5)} outro @ {fps:.1f}fps)")


if __name__ == "__main__":
    main()
