"""
record_training.py

Decoupled recording: BLE events and webcam frames run in separate threads.
No blocking in the BLE loop — zero latency stacking.

Run after:  python postprocess_session.py --session <session_folder>
"""

import sys
import asyncio
import argparse
import json
import time
import threading
import queue
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from cube_ble import CubeConnection
from orientation_tracker import OrientationTracker

JPEG_QUALITY   = 85
FRAME_SKIP = 1          # save every Nth frame (1=all, 2=every other)

_stop_event    = threading.Event()
_preview_queue = queue.Queue(maxsize=4)


# ---------------------------------------------------------------------------
# BLE thread — receives moves, writes timestamps, never sleeps
# ---------------------------------------------------------------------------

async def _ble_session(args, session_dir: Path, tracker: OrientationTracker,
                       error_bucket: list):
    moves_path = session_dir / "moves.jsonl"
    orient_path = session_dir / "orientation.jsonl"
    move_count = 0
    start_str  = None
    end_str    = None
    battery    = None

    # --- IMU quaternion log ------------------------------------------------
    # Written from a callback on the BLE loop thread, NOT drained by the move
    # loop below (which blocks on `async for`). Kept in a separate file from
    # moves.jsonl because it is a different KIND of record: moves are discrete
    # events, this is a ~15Hz continuous signal, and prepare_data.py aligns
    # the two by wall clock exactly as it already aligns moves to frames.
    orient_file = None
    orient_stats = {"n": 0, "norm_sum": 0.0,
                    "norm_min": None, "norm_max": None}

    def _log_orientation(ev):
        if orient_file is None:
            return
        orient_file.write(json.dumps({
            "timestamp": ev.timestamp,
            "qw": ev.qw, "qx": ev.qx, "qy": ev.qy, "qz": ev.qz,
            "norm": ev.norm,
            "raw_hex": ev.raw_hex,
        }) + "\n")
        orient_stats["n"] += 1
        orient_stats["norm_sum"] += ev.norm
        if orient_stats["norm_min"] is None or ev.norm < orient_stats["norm_min"]:
            orient_stats["norm_min"] = ev.norm
        if orient_stats["norm_max"] is None or ev.norm > orient_stats["norm_max"]:
            orient_stats["norm_max"] = ev.norm

    want_orientation = not args.no_orientation
    if want_orientation:
        orient_file = open(orient_path, "w")

    try:
        async with CubeConnection(
                address=args.address,
                orientation=want_orientation,
                on_orientation=_log_orientation if want_orientation else None,
        ) as cube:
            battery = await cube.get_battery()
            if battery is not None:
                print(f"  Battery: {battery}%")

            print("  Getting start state...")
            start_obj = await cube.get_state()
            start_str = start_obj.to_kociemba() if start_obj else None
            print(f"  Start:   {start_str or '(unknown)'}")
            print(f"\n  Solving... Q in preview window or Ctrl+C to stop.\n")

            with open(moves_path, "w") as f:
                async for ble_move in cube.moves():
                    if _stop_event.is_set():
                        break

                    result = tracker.apply_move_event(ble_move)
                    move_count += 1

                    record = {
                        "move_num":     move_count,
                        "timestamp":    ble_move.timestamp,
                        "ble_color":    ble_move.face_color,
                        "ble_direction":ble_move.direction,
                        "ble_raw":      ble_move.raw_byte,
                        "wca_face":     result.wca_face     if result else None,
                        "wca_notation": result.wca_notation if result else None,
                        "front_color":  result.front_color  if result else None,
                        "top_color":    result.top_color    if result else None,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    print(f"  [{move_count:3d}] "
                          f"{ble_move.face_color:<8} {ble_move.direction:<4}  "
                          f"→  {result.wca_notation if result else '?':<4}  "
                          f"t={ble_move.timestamp:.3f}")

            print("\n  Getting end state...")
            end_obj = await cube.get_state()
            end_str = end_obj.to_kociemba() if end_obj else None
            print(f"  End (as the cube reports it): {end_str or '(unknown)'}")

    except Exception as e:
        error_bucket.append(e)
    finally:
        _stop_event.set()
        if orient_file is not None:
            orient_file.flush()
            orient_file.close()
        # The cube's own state report is NOT ground truth and must not be
        # recorded under a name that reads like it is.
        #
        # The cube tracks its state internally and that tracking drifts — a
        # flat battery does it, and on 2026-07-23 a cube at 0% battery
        # reported a constant 36 facelets wrong while emitting phantom
        # moves. A solve session ends solved by construction of the
        # exercise, so a report of NOT SOLVED at the end of one is evidence
        # about the cube, not about the solve.
        #
        # Nothing downstream consumes these fields — every decode targets
        # the solved state and builds its start state from the move word
        # (reconstruct.start_from_gt, verify_joint.decode_moves,
        # session_check.check_pair) — so the honest fix is to label them as
        # the claim they are and point at the check that is actually load
        # bearing.
        with open(session_dir / "ble_meta.json", "w") as f:
            json.dump({
                "cube_reported_start_state": start_str,
                "cube_reported_end_state":   end_str,
                "end_state_trusted":         False,
                "end_state_note": (
                    "The cube's own state tracking, not an observation. A "
                    "solve session ends solved by construction; check the "
                    "MOVE WORD instead with session_check.py, which applies "
                    "it to a solved cube and asserts it returns there."),
                "battery":     battery,
                "move_count":  move_count,
                # --- IMU stream health -------------------------------------
                # `quaternion_norm_*` is the acceptance test for the payload
                # layout guessed in cube_ble.QUAT_SCALE/QUAT_ORDER. A unit
                # quaternion has |q| == 1, so norms clustered at 1.0 confirm
                # the width/endianness/scale; anything else means the parse is
                # wrong and orientation.jsonl's raw_hex is the material to fix
                # it with. CHECK THIS BEFORE TRUSTING ANY ROTATION LABEL.
                "quaternion_enabled":  want_orientation,
                "quaternion_samples":  orient_stats["n"],
                "quaternion_norm_mean": (
                    orient_stats["norm_sum"] / orient_stats["n"]
                    if orient_stats["n"] else None),
                "quaternion_norm_min": orient_stats["norm_min"],
                "quaternion_norm_max": orient_stats["norm_max"],
                "quaternion_layout_verified": False,
            }, f, indent=2)

        if want_orientation:
            n = orient_stats["n"]
            if n == 0:
                print("\n  WARNING: orientation stream enabled but NO "
                      "quaternion messages arrived.\n"
                      "           Either the cube ignored CMD_ENABLE_ORIENTATION "
                      "(0x38) or\n"
                      "           MSG_ORIENTATION is not 0x03 on this firmware.")
            else:
                mean = orient_stats["norm_sum"] / n
                ok = 0.9 < mean < 1.1
                print(f"\n  IMU: {n} quaternion samples, "
                      f"|q| mean {mean:.4f} "
                      f"(min {orient_stats['norm_min']:.4f}, "
                      f"max {orient_stats['norm_max']:.4f})")
                print("       |q| ~= 1 — payload layout guess holds."
                      if ok else
                      "       |q| IS NOT ~1 — the payload layout guess is WRONG.\n"
                      "       Do not use these quaternions. Fix "
                      "cube_ble.QUAT_SCALE/QUAT_ORDER\n"
                      "       against orientation.jsonl's raw_hex first.")


def _run_ble_thread(args, session_dir, tracker, error_bucket):
    """Fresh thread = clean MTA COM apartment on Windows."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _ble_session(args, session_dir, tracker, error_bucket)
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Frame capture thread — saves every frame with timestamp, never touches BLE
# ---------------------------------------------------------------------------

def _frame_capture_thread(session_dir: Path, camera_index: int = 0):
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_log = session_dir / "frames.jsonl"

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_idx  = 0
    skip_count = 0

    with open(frames_log, "w") as log_file:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            ts = time.time()
            frame = cv2.flip(frame, 1)

            skip_count += 1
            if skip_count < FRAME_SKIP:
                try:
                    _preview_queue.put_nowait(frame)
                except queue.Full:
                    pass
                continue
            skip_count = 0

            fname = f"frame_{frame_idx:06d}_{ts:.3f}.jpg"
            cv2.imwrite(str(frames_dir / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            log_file.write(json.dumps({"idx": frame_idx, "ts": ts,
                                       "file": fname}) + "\n")
            if frame_idx % 30 == 0:
                log_file.flush()

            try:
                _preview_queue.put_nowait(frame)
            except queue.Full:
                pass

            frame_idx += 1

    cap.release()
    print(f"  Frame capture stopped. {frame_idx} frames saved.")


# ---------------------------------------------------------------------------
# Main — cv2 preview on main thread (required on Windows)
# ---------------------------------------------------------------------------

def main():
    global FRAME_SKIP
    parser = argparse.ArgumentParser(
        description="Record training data from Rubik's Connected + webcam"
    )
    parser.add_argument("--front",   type=str, required=True)
    parser.add_argument("--top",     type=str, required=True)
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument("--output",  type=str, default="training_data")
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--skip",    type=int, default=FRAME_SKIP,
                        help="Save every Nth frame (default 1=all)")
    parser.add_argument("--no-orientation", action="store_true",
                        help="Disable the cube's ~15Hz IMU quaternion stream. "
                             "It is ON by default here because it is the only "
                             "ground truth for whole-cube rotations, which BLE "
                             "move events cannot see. Turn it off to reproduce "
                             "pre-2026-08-08 recording conditions, or if the "
                             "extra BLE traffic is suspected of worsening move "
                             "timestamp collisions.")
    args = parser.parse_args()
    
    FRAME_SKIP = max(1, args.skip)
    

    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output) / f"solve_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    tracker = OrientationTracker()
    tracker.calibrate(args.front, args.top)

    with open(session_dir / "config.json", "w") as f:
        json.dump({
            "session_id":   session_id,
            "started_at":   datetime.now().isoformat(),
            "front_color":  args.front,
            "top_color":    args.top,
            "frame_skip":   FRAME_SKIP,
            "jpeg_quality": JPEG_QUALITY,
        }, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  CubeArena — Training Data Recorder")
    print(f"{'='*55}")
    print(f"  Front={args.front}  Top={args.top}  skip={FRAME_SKIP}")
    print(f"  Output: {session_dir}")
    print(f"{'='*55}\n")

    errors = []

    ble_thread = threading.Thread(
        target=_run_ble_thread,
        args=(args, session_dir, tracker, errors),
        daemon=True, name="ble-thread",
    )
    frame_thread = threading.Thread(
        target=_frame_capture_thread,
        args=(session_dir, args.camera),
        daemon=True, name="frame-thread",
    )

    ble_thread.start()
    frame_thread.start()

    # Preview on main thread
    try:
        while not _stop_event.is_set():
            try:
                frame = _preview_queue.get(timeout=0.1)
                preview = cv2.resize(frame, (640, 360))
                cv2.putText(preview, "Recording — Q or Ctrl+C to stop",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 220, 60), 2)
                cv2.imshow("CubeArena Recorder", preview)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _stop_event.set()
                break

    except KeyboardInterrupt:
        _stop_event.set()

    ble_thread.join(timeout=5)
    frame_thread.join(timeout=5)
    cv2.destroyAllWindows()

    print(f"\n  Session saved: {session_dir}")
    print(f"  Next step:")
    print(f"    python postprocess_session.py --session \"{session_dir}\"\n")

    if errors:
        print(f"  BLE error: {errors[0]}")


if __name__ == "__main__":
    main()