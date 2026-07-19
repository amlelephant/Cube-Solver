"""
record_frames.py — lightweight webcam recorder for cube-detection training
data (no BLE, no move tracking — that's ble/record_training.py's job).

Just points a webcam at a cube (e.g. during a manual solve) and saves every
Nth frame to disk. Output lands in the same session/frames/*.jpg layout
autolabel.py already expects, so the next step is always:

  python record_frames.py
  python autolabel.py propose --session sessions/solve_XXXXXX_XXXXXX

Controls
--------
  Q / Esc   — stop recording
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

JPEG_QUALITY = 85


def record(output_dir: str, camera_index: int, skip: int):
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_dir) / f"solve_{session_id}"
    frames_dir  = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}")

    print(f"\nRecording to {frames_dir}")
    print("Q or Esc to stop.\n")

    frame_idx  = 0
    skip_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            skip_count += 1
            save = skip_count >= skip
            if save:
                skip_count = 0
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_idx += 1

            preview = cv2.resize(frame, (640, 360))
            label = f"Recording ({frame_idx} saved) — Q to stop" if save else "Recording..."
            cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 220, 60), 2)
            cv2.imshow("record_frames", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n{frame_idx} frames saved to {frames_dir}")
    print(f"Next: python autolabel.py propose --session {session_dir.as_posix()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Record webcam frames for cube-detection training data")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip",   type=int, default=1,
                         help="Save every Nth frame (default 1 = all)")
    parser.add_argument("--output", type=str, default="sessions")
    args = parser.parse_args()

    record(args.output, args.camera, max(1, args.skip))


if __name__ == "__main__":
    main()
