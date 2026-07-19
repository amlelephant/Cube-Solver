"""
live_test.py

Live webcam test for the trained move classifier.
No BLE cube required — uses frame-delta motion detection to find move windows,
then runs predict() on the ordered frame window [before, mids..., after]
exactly as the model was trained (predict() adapts the window to the
loaded checkpoint's 1-diff or 4-diff input).

Controls:
    Q / Escape   quit
    R            reset prediction history
    D            toggle debug overlay (shows frame delta value)

Usage:
    python live_test.py
    python live_test.py --model move_classifier.pt --camera 0 --threshold 12
"""

import argparse
import collections
import time
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DELTA_THRESHOLD     = 10.0   # mean abs pixel diff to declare MOVING
                              # run with --debug and watch the delta value;
                              # set this just above your camera's noise floor
STILL_CONFIRM       = 4      # consecutive STILL frames before declaring settled
MOVING_CONFIRM      = 2      # consecutive MOVING frames to start a move window
BUFFER_SECONDS      = 1.5    # how many seconds of frames to keep in memory
HISTORY_DISPLAY     = 8      # number of recent predictions to show on screen
IMG_SIZE            = 224    # must match training

CLASS_NAMES = [
    "blue-CW",   "blue-CCW",
    "green-CW",  "green-CCW",
    "white-CW",  "white-CCW",
    "yellow-CW", "yellow-CCW",
    "red-CW",    "red-CCW",
    "orange-CW", "orange-CCW",
]

# BGR colours for each class label (for the on-screen display)
CLASS_COLORS = {
    "blue-CW":    (200, 80,  20 ),
    "blue-CCW":   (200, 80,  20 ),
    "green-CW":   (40,  180, 40 ),
    "green-CCW":  (40,  180, 40 ),
    "white-CW":   (230, 230, 230),
    "white-CCW":  (230, 230, 230),
    "yellow-CW":  (0,   220, 230),
    "yellow-CCW": (0,   220, 230),
    "red-CW":     (40,  40,  210),
    "red-CCW":    (40,  40,  210),
    "orange-CW":  (20,  140, 230),
    "orange-CCW": (20,  140, 230),
}


# ---------------------------------------------------------------------------
# Motion state machine
# ---------------------------------------------------------------------------

class MotionGate:
    """
    Tracks STILL / MOVING state using frame-delta on the full frame.
    Returns the ordered move window when a move completes.
    """

    def __init__(self, threshold: float, buffer_seconds: float,
                 still_confirm: int, moving_confirm: int):
        self.threshold      = threshold
        self.still_confirm  = still_confirm
        self.moving_confirm = moving_confirm

        self._state         = "STILL"
        self._confirm_buf   = collections.deque(maxlen=max(still_confirm,
                                                            moving_confirm))
        self._frame_buffer  = collections.deque()   # (timestamp, frame)
        self._buffer_maxlen = int(buffer_seconds * 30)  # approx 30fps

        self._last_still_frame  = None    # most recent confirmed STILL frame
        self._motion_window     = []      # (delta, frame) pairs during MOVING
        self._prev_gray         = None

    def update(self, frame: np.ndarray) -> tuple[str, float, tuple | None]:
        """
        Feed one frame.

        Returns:
            state   str    "STILL" | "MOVING"
            delta   float  raw frame-delta value
            result  tuple | None
                    On MOVING→STILL transition:
                    (frames, peak_delta) where frames is the ordered
                    5-frame window [before, mid x3, after]
                    Otherwise: None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._prev_gray is None:
            self._prev_gray = gray
            return "STILL", 0.0, None

        delta = float(np.mean(np.abs(gray - self._prev_gray)))
        self._prev_gray = gray

        # Accumulate raw readings
        raw = "MOVING" if delta > self.threshold else "STILL"
        self._confirm_buf.append(raw)

        moving_count = sum(1 for s in self._confirm_buf if s == "MOVING")
        still_count  = sum(1 for s in self._confirm_buf if s == "STILL")

        prev_state = self._state
        if moving_count >= self.moving_confirm:
            self._state = "MOVING"
        if still_count  >= self.still_confirm:
            self._state = "STILL"

        result = None

        if self._state == "MOVING":
            self._motion_window.append((delta, frame.copy()))

        elif self._state == "STILL":
            # Transition from MOVING → STILL: assemble the ordered window
            if prev_state == "MOVING" and self._motion_window:
                before = self._last_still_frame
                peak_delta = max(d for d, _ in self._motion_window)

                if before is not None:
                    # 3 evenly spaced motion frames (duplicates ok when
                    # the turn spanned fewer than 3 frames) + the current
                    # frame as 'after' — mirrors the training window
                    # [before, mid_00, mid_01, mid_02, after].
                    n    = len(self._motion_window)
                    mids = [self._motion_window[round(i * (n - 1) / 2)][1]
                            for i in range(3)]
                    result = ([before] + mids + [frame.copy()], peak_delta)

                self._motion_window.clear()

            self._last_still_frame = frame.copy()

        return self._state, delta, result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_state_bar(frame: np.ndarray, state: str, delta: float,
                   threshold: float, debug: bool):
    h, w = frame.shape[:2]

    color = (0, 60, 200) if state == "MOVING" else (0, 200, 60)
    cv2.putText(frame, state, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    if debug:
        bar_w = min(int(delta * 12), w - 100)
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 64), color, -1)
        cv2.rectangle(frame, (10 + int(threshold * 12), 46),
                      (10 + int(threshold * 12) + 2, 68), (255, 255, 255), -1)
        cv2.putText(frame, f"delta={delta:.1f}  thr={threshold:.1f}",
                    (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)


def draw_prediction(frame: np.ndarray, cls_name: str, conf: float):
    """Draw the most recent prediction prominently at the top."""
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(cls_name, (200, 200, 200))

    # Dark background bar
    cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)

    cv2.putText(frame, cls_name, (14, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{conf*100:.0f}%", (14, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def draw_history(frame: np.ndarray, history: list[tuple[str, float]]):
    """Draw a scrolling list of recent predictions on the right side."""
    h, w = frame.shape[:2]
    x    = w - 180
    y0   = 40

    cv2.putText(frame, "Recent:", (x, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    for i, (name, conf) in enumerate(reversed(history[-HISTORY_DISPLAY:])):
        y     = y0 + i * 22
        color = CLASS_COLORS.get(name, (200, 200, 200))
        alpha = max(0.35, 1.0 - i * 0.12)
        faded = tuple(int(c * alpha) for c in color)
        cv2.putText(frame, f"{name}  {conf*100:.0f}%",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, faded, 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    # Load model
    try:
        from train_move_classifier import predict, load_for_inference
    except ImportError:
        sys.exit("train_move_classifier.py not found in current directory.")

    print(f"\nLoading model from {args.model} ...")
    try:
        load_for_inference(args.model)
        print("Model loaded.\n")
    except FileNotFoundError:
        sys.exit(f"Model not found: {args.model}\n"
                 f"Train first: python train_move_classifier.py --sessions training_data/solve_*/")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    gate    = MotionGate(
        threshold      = args.threshold,
        buffer_seconds = BUFFER_SECONDS,
        still_confirm  = STILL_CONFIRM,
        moving_confirm = MOVING_CONFIRM,
    )
    history = []
    debug   = args.debug
    last_pred_name = ""
    last_pred_conf = 0.0

    print("Controls:  Q/Escape=quit   R=reset history   D=toggle debug\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Motion gate update
        state, delta, move_window = gate.update(frame)

        # If a complete move window was detected, run the classifier
        if move_window is not None:
            window_frames, peak_delta = move_window
            cls_id, cls_name, conf = predict(window_frames, args.model)
            history.append((cls_name, conf))
            last_pred_name = cls_name
            last_pred_conf = conf
            print(f"  Detected: {cls_name:<14}  conf={conf*100:.0f}%"
                  f"  (peak_delta={peak_delta:.1f})")

        # Draw UI
        draw_state_bar(display, state, delta, args.threshold, debug)
        if last_pred_name:
            draw_prediction(display, last_pred_name, last_pred_conf)
        if history:
            draw_history(display, history)

        # Instructions
        cv2.putText(display, "Q=quit  R=reset  D=debug",
                    (12, display.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

        cv2.imshow("CubeArena — Live Move Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            history.clear()
            last_pred_name = ""
            last_pred_conf = 0.0
            print("  History cleared.")
        elif key == ord("d"):
            debug = not debug
            print(f"  Debug overlay: {'on' if debug else 'off'}")

    cap.release()
    cv2.destroyAllWindows()

    if history:
        print(f"\nSession summary ({len(history)} predictions):")
        from collections import Counter
        for name, count in Counter(n for n,_ in history).most_common():
            avg_conf = sum(c for n,c in history if n==name) / count
            print(f"  {name:<14}  {count:3d}×  avg conf {avg_conf*100:.0f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live webcam test for the move classifier"
    )
    parser.add_argument("--model",     type=str,   default="move_classifier.pt")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--threshold", type=float, default=DELTA_THRESHOLD,
                        help="Frame-delta motion threshold (default 10.0). "
                             "Run with --debug and turn slowly to calibrate.")
    parser.add_argument("--debug",     action="store_true",
                        help="Show delta bar and threshold line overlay")
    args = parser.parse_args()
    run(args)