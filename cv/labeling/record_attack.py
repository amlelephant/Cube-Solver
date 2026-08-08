"""
record_attack.py — record deliberate swap-attack attempts (and matched legit
control runs) as labelled sessions for anticheat validation.

Roadmap Track C1 (LAUNCH_ROADMAP.md §4). These sessions are the HELD-OUT
VALIDATION set for the trajectory anomaly model — they are deliberately
never trained on. Training on recorded attacks would teach the model one
person's swap style; characterising normal motion from the ~69 legit
sessions and testing against these keeps the model honest against swap
techniques nobody thought to perform. Same discipline as verify_solve.py's
falsifiability decoys, which validate and never train.

Output matches the ble/training_data session layout exactly (frames/ +
frames.jsonl), so dump_trajectories.py, batch_test_guard.py and
continuity_guard.py all consume these with no changes. Alongside them it
writes attack.json:

    {"attack_type": "table_edge", "is_attack": true,
     "marks": [{"t": 12.4, "frame_idx": 372}],   # swap moments
     "started_at": ..., "fw":, "fh":, "n_frames":, "notes": ...}

`marks` is the ground truth the validation asks about: not merely "was this
session flagged" but "was it flagged AT the swap" — a model that DQs the
right session for the wrong reason has not actually detected anything.

MARK AFTERWARDS, NOT DURING. Every exploit here is an unsolved cube being
switched for a solved one, which needs both hands — there is no hand free
for the keyboard at the moment that matters, and a live keypress would
carry ~200-500ms of reaction lag regardless. So just record, then run
`mark_swap.py`, which proposes the swap moments from the tracked box's
appearance change and lets you confirm them frame-accurately:

    python record_attack.py --type table_edge
    cd ../detection && python dump_trajectories.py --root ../labeling/attack_sessions
    cd ../labeling && python mark_swap.py --session attack_sessions/table_edge_<stamp>

SPACE still works mid-recording if a hand happens to be free — it is a
convenience, not a requirement.

Record a matched control (--type legit) in the same sitting and lighting as
each attack batch: a false-DQ rate measured on old sessions in different
conditions is not the number the go/no-go gate needs (<1% false DQ).

Run from inside cv/labeling/:
    python record_attack.py --type table_edge
    python record_attack.py --type legit --notes "control for table_edge batch"

Attack types
------------
  table_edge   dip the cube below a mid-frame table edge, swap, raise it back
               (TODO.md item 1's known hole — the priority case)
  under_table  full below-frame exit and return with a different cube
  palm_swap    fast swap fully in view, hands together
  two_cube     both cubes visible at once
  edge_exit    leave frame at a border, return with the other cube
  legit        control run, no cheating (measures the false-DQ rate)

Controls
--------
  SPACE     optional — mark a swap moment if a hand is free (see above;
            mark_swap.py is the intended path)
  Q / Esc   stop recording
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

ATTACK_TYPES = ["table_edge", "under_table", "palm_swap", "two_cube",
                "edge_exit", "legit"]
JPEG_QUALITY = 85


def record(attack_type, output_dir, camera_index, notes):
    is_attack = attack_type != "legit"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_dir) / f"{attack_type}_{session_id}"
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # DirectShow + MJPG for the same reason live_guard_test.py does it:
    # Windows webcams streaming raw YUY2 at 720p get bandwidth-capped near
    # 10fps, and capture rate is a security parameter for this data — the
    # guard can only certify the interval it actually samples.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, probe = cap.read()
        if not ok:
            cap.release()
            cap = None
    else:
        cap = None
    if cap is None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, probe = cap.read()
        if not ok:
            raise SystemExit("Camera opened but returned no frame")
    fh, fw = probe.shape[:2]

    banner = ("ATTACK — perform the swap" if is_attack
              else "CONTROL — solve normally, do NOT cheat")
    print(f"\nRecording {attack_type} -> {frames_dir}")
    print(f"  {banner}")
    print("  Q/Esc = stop.  Both hands on the cube is expected — mark the "
          "swap afterwards with mark_swap.py.")
    print("  (SPACE marks a moment live, if a hand happens to be free.)\n")

    manifest = open(session_dir / "frames.jsonl", "w")
    marks = []
    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            now = time.time()
            fname = f"frame_{frame_idx:06d}_{now:.3f}.jpg"
            cv2.imwrite(str(frames_dir / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            manifest.write(json.dumps({"idx": frame_idx, "ts": now,
                                       "file": fname}) + "\n")

            preview = cv2.resize(frame, (640, 360))
            colour = (60, 60, 230) if is_attack else (60, 200, 60)
            cv2.putText(preview, f"{attack_type}  {frame_idx} frames",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(preview, f"marks: {len(marks)}   SPACE=mark  Q=stop",
                        (10, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1)
            if marks and now - t0 - marks[-1]["t"] < 1.0:
                cv2.putText(preview, "MARKED", (250, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3)
            cv2.imshow("record_attack", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                marks.append({"t": round(now - t0, 3), "frame_idx": frame_idx})
                print(f"  marked swap at t={now - t0:.2f}s (frame {frame_idx})")
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        manifest.close()

    meta = {
        "attack_type": attack_type,
        "is_attack": is_attack,
        "marks": marks,
        "started_at": datetime.fromtimestamp(t0).isoformat(),
        "n_frames": frame_idx,
        "fw": fw, "fh": fh,
        "notes": notes or "",
        "source": "record_attack.py",
    }
    with open(session_dir / "attack.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{frame_idx} frames, {len(marks)} live mark(s) -> {session_dir}")
    root = Path(output_dir).as_posix()
    if is_attack:
        print("\nNext — mark the swap (proposed for you, no frame-hunting):")
        print(f"  cd ../detection && python dump_trajectories.py --root ../labeling/{root}")
        print(f"  cd ../labeling && python mark_swap.py --session {session_dir.as_posix()}")
    else:
        print(f"\nNext: cd ../detection && python dump_trajectories.py "
              f"--root ../labeling/{root}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=ATTACK_TYPES,
                    dest="attack_type", help="attack type (or 'legit' control)")
    ap.add_argument("--output", default="attack_sessions")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    record(args.attack_type, args.output, args.camera, args.notes)


if __name__ == "__main__":
    main()
