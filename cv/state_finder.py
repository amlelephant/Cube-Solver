"""
state_finder.py

Full 6-face cube state scanner with live webcam UI.

Flow
----
1. Open webcam
2. For each of the 6 faces (U, R, F, D, L, B):
   a. Show live feed with detection overlay
   b. User presses SPACE to capture the current frame
   c. cube_detector finds the face and extracts 9 sticker patches
   d. ensemble classifies each sticker
   e. Result shown on screen — user can retry if unhappy
3. After all 6 faces scanned, assemble state string
4. Validate with kociemba group-theory check
5. Print solution

Controls
--------
  SPACE   — scan current face
  R       — redo the current face
  Q       — quit
  D       — toggle debug overlay (shows detection method, HSV values)
"""

import cv2
import numpy as np
import sys

# Some terminals (default Windows console codepages) can't encode the
# unicode arrows/checkmarks used in this script's output — force UTF-8
# so printing them doesn't crash the process.
sys.stdout.reconfigure(encoding="utf-8")

from cube_detector import detect_and_extract, draw_sticker_overlay
from ensemble      import classify_face, calibrate, reset_calibration, calibration_status
from color_classifier import (
    CLASSES, COLOR_BGR, COLOR_TO_FACE,
)

# Vendored pure-Python two-phase solver (cv/twophase/, MIT licensed —
# see cv/twophase/LICENSE.txt). Used in place of the 'kociemba' PyPI
# package, which ships no Windows wheel and requires a C compiler to
# build from source.
import twophase as kociemba

# ---------------------------------------------------------------------------
# Face scanning order (WCA standard)
# ---------------------------------------------------------------------------
FACE_ORDER = ["U", "R", "F", "D", "L", "B"]

FACE_INFO = {
    "U": {"name": "Up",    "center_color": "white",  "hint": "white center facing you"},
    "R": {"name": "Right", "center_color": "red",    "hint": "red center facing you"},
    "F": {"name": "Front", "center_color": "green",  "hint": "green center facing you"},
    "D": {"name": "Down",  "center_color": "yellow", "hint": "yellow center facing you"},
    "L": {"name": "Left",  "center_color": "orange", "hint": "orange center facing you"},
    "B": {"name": "Back",  "center_color": "blue",   "hint": "blue center facing you"},
}

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
WIN_NAME  = "CubeArena — State Finder"
UI_WIDTH  = 900
UI_HEIGHT = 600
CAM_W     = 640
CAM_H     = 480

# Panel on the right side of the window
PANEL_X   = CAM_W + 10
PANEL_W   = UI_WIDTH - CAM_W - 20

# Colors (BGR)
C_WHITE   = (245, 245, 240)
C_GRAY    = (130, 130, 130)
C_DARK    = (40,  40,  40 )
C_GREEN   = (50,  200, 80 )
C_YELLOW  = (0,   220, 230)
C_RED     = (60,  60,  210)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ScanState:
    def __init__(self):
        self.face_idx     = 0           # which face we're currently on
        self.scanned      = {}          # face_key -> {"colors": [...], "confs": [...]}
        self.last_frame   = None        # most recent raw frame
        self.last_result  = None        # most recent detection result dict
        self.last_colors  = None        # colors from last capture
        self.last_confs   = None        # confidences from last capture
        self.debug_mode   = False
        self.status_msg   = ""
        self.status_color = C_WHITE

    @property
    def current_face(self):
        if self.face_idx < len(FACE_ORDER):
            return FACE_ORDER[self.face_idx]
        return None

    @property
    def done(self):
        return len(self.scanned) == 6


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_text(img, text, x, y, color=C_WHITE, scale=0.55, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_mini_face(canvas, colors, confs, x, y, cell=18, label=""):
    """Draw a 3×3 colored grid representing a scanned face."""
    for row in range(3):
        for col in range(3):
            idx  = row * 3 + col
            name = colors[idx] if idx < len(colors) else "unknown"
            bgr  = COLOR_BGR.get(name, (90, 90, 90))
            conf = confs[idx]   if idx < len(confs)  else 0.0

            rx = x + col * cell
            ry = y + row * cell
            cv2.rectangle(canvas, (rx, ry), (rx + cell - 2, ry + cell - 2), bgr, -1)

            # Yellow outline if low confidence
            border = (0, 200, 255) if conf < 0.5 else (40, 40, 40)
            cv2.rectangle(canvas, (rx, ry), (rx + cell - 2, ry + cell - 2), border, 1)

    if label:
        draw_text(canvas, label, x, y + 3 * cell + 12, C_GRAY, 0.45)


def build_ui_frame(cam_frame, state):
    """
    Compose the full UI frame:
      Left side: camera feed (with detection overlay if available)
      Right side: status panel
    """
    canvas = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 22)

    # ---- Left: camera feed --------------------------------------------------
    cam_resized = cv2.resize(cam_frame, (CAM_W, CAM_H))
    canvas[:CAM_H, :CAM_W] = cam_resized

    # ---- Right panel --------------------------------------------------------
    px = PANEL_X
    py = 20

    # Title
    draw_text(canvas, "CubeArena", px, py, C_WHITE, 0.8, 2)
    py += 26
    draw_text(canvas, "Cube State Finder", px, py, C_GRAY, 0.5)
    py += 30

    # Divider
    cv2.line(canvas, (px, py), (px + PANEL_W, py), (55, 55, 55), 1)
    py += 14

    # Current face prompt
    face_key = state.current_face
    if face_key and not state.done:
        info      = FACE_INFO[face_key]
        center_c  = info["center_color"]
        bgr       = COLOR_BGR.get(center_c, C_GRAY)

        draw_text(canvas, f"Face {state.face_idx + 1} of 6", px, py, C_GRAY, 0.5)
        py += 20
        draw_text(canvas, f"{info['name']} ({center_c})", px, py, C_WHITE, 0.65, 1)
        py += 18
        draw_text(canvas, info["hint"], px, py, bgr, 0.48)
        py += 28

        # Most recent capture preview
        if state.last_colors:
            draw_text(canvas, "Last capture:", px, py, C_GRAY, 0.45)
            py += 14
            draw_mini_face(canvas, state.last_colors, state.last_confs, px, py)
            py += 3 * 18 + 24

    # Scanned faces summary
    if state.scanned:
        draw_text(canvas, "Scanned faces:", px, py, C_GRAY, 0.45)
        py += 16

        summary_x = px
        for fi, fk in enumerate(FACE_ORDER):
            if fk in state.scanned:
                data = state.scanned[fk]
                sx   = summary_x + (fi % 3) * 80
                sy   = py + (fi // 3) * 82
                draw_mini_face(canvas, data["colors"], data["confs"],
                               sx, sy, cell=16, label=fk)

        py += 2 * 82 + 10

    # Status message
    if state.status_msg:
        draw_text(canvas, state.status_msg, px, min(py + 10, UI_HEIGHT - 80),
                  state.status_color, 0.5)

    # Controls footer
    footer_y = UI_HEIGHT - 55
    cv2.line(canvas, (px, footer_y - 6), (px + PANEL_W, footer_y - 6), (55, 55, 55), 1)
    draw_text(canvas, "SPACE  Scan face",    px, footer_y,      C_GRAY, 0.45)
    draw_text(canvas, "R  Redo face",        px, footer_y + 16, C_GRAY, 0.45)
    draw_text(canvas, "D  Debug overlay",    px, footer_y + 32, C_GRAY, 0.45)
    draw_text(canvas, "Q  Quit",             px, footer_y + 48, C_GRAY, 0.45)

    # Progress bar at bottom of cam area
    total_faces   = 6
    filled_faces  = len(state.scanned)
    bar_w         = CAM_W
    bar_h         = 6
    bar_y         = CAM_H - bar_h
    filled_w      = int(bar_w * filled_faces / total_faces)
    cv2.rectangle(canvas, (0, bar_y), (bar_w, bar_y + bar_h), (50, 50, 50), -1)
    if filled_w > 0:
        cv2.rectangle(canvas, (0, bar_y), (filled_w, bar_y + bar_h), C_GREEN, -1)

    return canvas


def draw_debug_overlay(canvas, state):
    """Overlay HSV values and method tag if debug mode is on."""
    if not state.debug_mode or not state.last_result:
        return
    result = state.last_result
    method = result.get("method", "")
    draw_text(canvas, f"detect={method}", 8, CAM_H - 16, C_YELLOW, 0.45)

    cal = calibration_status()
    cal_txt = (f"orange/red boundary: calibrated (R={cal['red']} L={cal['orange']})"
               if cal["active"] else
               "orange/red boundary: not yet calibrated (scan R and L faces)")
    draw_text(canvas, cal_txt, 8, CAM_H - 2, C_YELLOW, 0.4)


# ---------------------------------------------------------------------------
# Core scan action
# ---------------------------------------------------------------------------
def scan_current_face(frame, state):
    """
    Run detection + ensemble on the current frame.
    Updates state.last_* fields and state.status_msg.
    Returns True if the scan was usable (even if some stickers are uncertain).
    """
    result = detect_and_extract(frame)
    state.last_result = result

    if not result["found"] or len(result["patches"]) != 9:
        state.status_msg   = "No cube face detected — adjust position"
        state.status_color = C_RED
        return False

    colors, confs, details = classify_face(result["patches"])

    # The center sticker's true color is known in advance (fixed WCA
    # scheme — see FACE_INFO), independent of what the classifier guessed.
    # Use it to calibrate the orange/red boundary to this session's actual
    # camera + lighting (see ensemble.calibrate).
    center_color  = FACE_INFO[state.current_face]["center_color"]
    center_detail = details[4]
    if "h" in center_detail:
        calibrate(center_color, center_detail["h"], center_detail["s"], center_detail["v"])

    unknown_count = colors.count("unknown")
    low_conf      = sum(1 for c in confs if c < 0.45)

    state.last_colors = colors
    state.last_confs  = confs

    if unknown_count > 2:
        state.status_msg   = f"{unknown_count} stickers unrecognized — try again"
        state.status_color = C_RED
        return False

    if low_conf > 3:
        state.status_msg   = f"Low confidence on {low_conf} stickers — accepted (check result)"
        state.status_color = C_YELLOW
    else:
        state.status_msg   = "Face scanned successfully"
        state.status_color = C_GREEN

    return True


# ---------------------------------------------------------------------------
# State string assembly and solving
# ---------------------------------------------------------------------------
def build_state_string(scanned):
    """
    Assemble the 54-char URFDLB state string from scanned face data.
    Returns (state_string, error_message).
    """
    state_str = ""
    for face_key in FACE_ORDER:
        if face_key not in scanned:
            return None, f"Missing face: {face_key}"
        colors = scanned[face_key]["colors"]
        for color in colors:
            letter = COLOR_TO_FACE.get(color)
            if letter is None:
                return None, f"Unknown color '{color}' on face {face_key}"
            state_str += letter

    if len(state_str) != 54:
        return None, f"State string wrong length: {len(state_str)}"

    return state_str, None


def validate_and_solve(state_str):
    """
    Run the kociemba solver. Returns (solution, error).
    """
    try:
        solution = kociemba.solve(state_str)
        return solution, None
    except Exception as e:
        return None, str(e)


def print_results(scanned):
    """Print final state string and solution to terminal."""
    print("\n" + "=" * 55)
    print("  CUBE STATE FINDER — RESULTS")
    print("=" * 55)

    state_str, err = build_state_string(scanned)
    if err:
        print(f"  Error building state: {err}")
        return

    print(f"  State string : {state_str}")

    # Per-face breakdown
    print("\n  Face breakdown:")
    for face_key in FACE_ORDER:
        data   = scanned[face_key]
        colors = data["colors"]
        confs  = data["confs"]
        letters = "".join(COLOR_TO_FACE.get(c, "?") for c in colors)
        avg_conf = sum(confs) / len(confs) if confs else 0
        print(f"    {face_key}: {letters}  (avg confidence: {avg_conf:.2f})")

    print()
    solution, err = validate_and_solve(state_str)
    if err:
        print(f"  Solver error: {err}")
        print("  This usually means the scanned state is physically impossible.")
        print("  Re-scan any face that seemed uncertain.")
    else:
        if solution == "":
            print("  Cube is already solved!")
        else:
            moves = solution.split()
            print(f"  Solution ({len(moves)} moves):")
            print(f"    {solution}")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    reset_calibration()  # fresh camera/lighting calibration for this session

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = ScanState()

    # Check if CNN model is available
    from cnn_classifier import MODEL_PATH
    import os
    if os.path.exists(MODEL_PATH):
        state.status_msg   = "CNN model loaded. Ready to scan."
        state.status_color = C_GREEN
        print(f"[INFO] CNN model found at '{MODEL_PATH}'")
    else:
        state.status_msg   = "No CNN model — using HSV only. Run: python cnn_classifier.py --train"
        state.status_color = C_YELLOW
        print(f"[INFO] No CNN model at '{MODEL_PATH}'. Running HSV-only mode.")
        print(f"       To train: python cnn_classifier.py --train")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, UI_WIDTH, UI_HEIGHT)

    print("\nControls: SPACE=scan  R=redo  D=debug  Q=quit\n")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("ERROR: Lost webcam feed.")
            break

        frame = cv2.flip(raw_frame, 1)  # mirror for natural feel

        # Always run detection on the live feed for the overlay
        live_result = detect_and_extract(frame)
        display_frame = live_result["debug_frame"].copy()

        # Draw sticker overlay if we have a recent good scan
        if (live_result["found"]
                and state.last_colors
                and live_result["quad"] is not None):
            draw_sticker_overlay(
                display_frame,
                live_result["quad"],
                state.last_colors,
                state.last_confs,
            )

        ui = build_ui_frame(display_frame, state)
        draw_debug_overlay(ui, state)

        cv2.imshow(WIN_NAME, ui)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("d"):
            state.debug_mode = not state.debug_mode

        elif key == ord("r") and not state.done:
            # Redo current face
            if state.current_face in state.scanned:
                del state.scanned[state.current_face]
                state.face_idx  -= 1
            state.last_colors  = None
            state.last_confs   = None
            state.status_msg   = f"Redoing face {state.current_face}"
            state.status_color = C_YELLOW

        elif key == ord(" ") and not state.done:
            face_key = state.current_face
            ok = scan_current_face(frame, state)

            if ok:
                state.scanned[face_key] = {
                    "colors": state.last_colors,
                    "confs":  state.last_confs,
                }
                state.face_idx += 1

                if state.done:
                    print_results(state.scanned)
                    state.status_msg   = "All faces scanned! See terminal for solution."
                    state.status_color = C_GREEN
                else:
                    next_face = FACE_ORDER[state.face_idx]
                    next_info = FACE_INFO[next_face]
                    state.status_msg   = f"Next: {next_info['name']} face ({next_info['hint']})"
                    state.status_color = C_WHITE

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()