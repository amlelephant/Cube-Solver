"""
orientation_tracker.py

Tracks which face of the Rubik's cube is facing the camera at any given
moment, and translates raw BLE move events (which are cube-relative, i.e.
"the red face turned CW") into camera-relative WCA notation (e.g. "R").

The problem
-----------
Your smart cube reports moves in terms of CENTER COLORS, not WCA letters.
"Red face CW" always means the same physical face turned — but whether that
translates to R, L, U, D, F, or B in WCA notation depends entirely on how
the cube is oriented in your hands relative to the camera.

The solution
------------
Maintain a VirtualCube — a software model of which center color is currently
on each of the six WCA faces (U/D/F/B/L/R). When you start a session you do
a one-time calibration: tell the tracker which color is currently on U, F,
and R (you pick those three and it derives the other three automatically).

After that, every move event from the BLE cube is:
  1. Looked up in the current orientation map  →  WCA face letter
  2. Applied to the virtual cube state          →  orientation updated
  3. Returned as a WCA-notation string          →  "R", "U'", "F2", etc.

Whole-cube rotations (x, y, z) work the same way: they are detected as
moves that affect all six face centers simultaneously and update the
orientation map without emitting a WCA sticker move.

Calibration
-----------
The simplest reliable calibration for your use case:

  1. Hold the cube so the CAMERA can see one face clearly.
  2. Call  tracker.calibrate(front_color, top_color)
     passing the colors of the face pointing AT the camera (F)
     and the face pointing UP (U).
  3. The tracker derives R automatically (cross product of F and U vectors).

You only need to do this once per solve session, and only when you pick up
the cube. After that you can rotate however you like — the tracker keeps up.

IMU drift correction
--------------------
The BLE cube also streams MsgOrientation quaternions at 15/sec. These are
re-enabled in this module (we disabled them in cube_ble.py for the basic
recording use case). After every 20 moves, the tracker optionally snaps its
internal orientation to the IMU quaternion — this catches any accumulated
error from missed move events.

Usage
-----
    from orientation_tracker import OrientationTracker, COLORS
    from cube_ble import CubeConnection, MoveEvent

    tracker = OrientationTracker()

    # One-time calibration: green face toward camera, white face on top
    tracker.calibrate(front_color="green", top_color="white")

    async with CubeConnection() as cube:
        async for ble_move in cube.moves():
            result = tracker.apply_move(ble_move)
            if result.is_rotation:
                print(f"Whole-cube rotation: {result.rotation_name}")
            else:
                print(f"WCA move: {result.wca_notation}")
                print(f"  front face is now: {tracker.front_color}")
"""
import sys
sys.coinit_flags=0

import math
from dataclasses import dataclass
from typing import Optional

# Dependency-free by design, so importing it can never fail on the capture
# machine — see its module docstring.
from slice_frame import CameraFrame, is_slice_pair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = ("white", "yellow", "red", "orange", "blue", "green")

# WCA face letters
U, D, F, B, R, L = "U", "D", "F", "B", "R", "L"
FACES = (U, D, F, B, R, L)

# Opposite face pairs
OPPOSITE = {U: D, D: U, F: B, B: F, R: L, L: R}

# ---------------------------------------------------------------------------
# Orientation state
# ---------------------------------------------------------------------------

class FaceMap:
    """
    Maps each WCA face letter to the color currently on that face.
    e.g. {U: "white", D: "yellow", F: "green", B: "blue", R: "red", L: "orange"}

    This is the core state. When the cube is in standard WCA orientation:
      U=white, D=yellow, F=green, B=blue, R=red, L=orange
    """

    def __init__(self, mapping: dict):
        assert set(mapping.keys()) == set(FACES), f"Need all 6 faces: {FACES}"
        assert set(mapping.values()) == set(COLORS[:6]) or \
               len(set(mapping.values())) == 6, "Need all 6 distinct colors"
        self._m = dict(mapping)

    @classmethod
    def standard(cls):
        """WCA standard orientation: white U, yellow D, green F, blue B, red R, orange L."""
        return cls({U:"white", D:"yellow", F:"green", B:"blue", R:"red", L:"orange"})

    @classmethod
    def from_front_and_top(cls, front_color: str, top_color: str) -> "FaceMap":
        """
        Derive a complete FaceMap from just the front (F) and top (U) colors.
        The remaining four faces are fully determined by these two.
        Raises ValueError if front/top are opposite colors (physically impossible).
        """
        # Build the standard color→face lookup
        standard = cls.standard()
        std_color_to_face = {v: k for k, v in standard._m.items()}

        # Find what WCA rotation moves the standard cube to this orientation
        # Strategy: find the rotation sequence that puts front_color on F and top_color on U
        # then apply it to the standard face map.

        # Enumerate all 24 valid orientations and find the matching one
        for orientation in _all_24_orientations():
            if (orientation[F] == front_color and orientation[U] == top_color):
                return cls(orientation)

        raise ValueError(
            f"No valid orientation has {front_color} on F and {top_color} on U. "
            f"Are they opposite colors?"
        )

    def color_on(self, face: str) -> str:
        """Return the color currently on a given WCA face."""
        return self._m[face]

    def face_of(self, color: str) -> str:
        """Return which WCA face a given color is currently on."""
        for face, col in self._m.items():
            if col == color:
                return face
        raise KeyError(f"Color {color!r} not in map")

    def apply_face_rotation(self, wca_face: str, direction: str) -> "FaceMap":
        """
        Apply a single face rotation (not a whole-cube rotation).
        This does NOT change which color is on which face — face rotations
        only move stickers, not centers. Centers stay put.
        Returns self unchanged (face map only tracks centers).
        """
        return self  # centers don't move on face turns

    def apply_whole_rotation(self, axis: str, turns: int) -> "FaceMap":
        """
        Apply a whole-cube rotation (x, y, or z) and return the new FaceMap.

        x  = rotate around R-L axis (U face tips toward camera / F)
        y  = rotate around U-D axis (F face tips to R / right)
        z  = rotate around F-B axis (U face tips to R)

        turns: 1=CW, -1=CCW, 2=180
        """
        m = dict(self._m)
        turns = turns % 4

        for _ in range(turns):
            if axis == "x":
                # U→F, F→D, D→B, B→U  (R and L stay)
                m = {
                    U: self._m[B], F: self._m[U],
                    D: self._m[F], B: self._m[D],
                    R: self._m[R], L: self._m[L],
                }
            elif axis == "y":
                # F→R, R→B, B→L, L→F  (U and D stay)
                m = {
                    F: self._m[L], R: self._m[F],
                    B: self._m[R], L: self._m[B],
                    U: self._m[U], D: self._m[D],
                }
            elif axis == "z":
                # U→R, R→D, D→L, L→U  (F and B stay)
                m = {
                    U: self._m[L], R: self._m[U],
                    D: self._m[R], L: self._m[D],
                    F: self._m[F], B: self._m[B],
                }
            self = FaceMap(m)

        return FaceMap(m)

    def __repr__(self):
        return (f"FaceMap(U={self._m[U]}, D={self._m[D]}, "
                f"F={self._m[F]}, B={self._m[B]}, "
                f"R={self._m[R]}, L={self._m[L]})")


# ---------------------------------------------------------------------------
# The 24 valid cube orientations (precomputed)
# ---------------------------------------------------------------------------

def _all_24_orientations() -> list[dict]:
    """
    Generate all 24 valid orientations of a cube by systematically applying
    whole-cube rotations to the standard orientation.
    """
    seen = set()
    result = []

    def orientation_key(m):
        return tuple(m[f] for f in FACES)

    def apply_x(m):
        return {U:m[B], F:m[U], D:m[F], B:m[D], R:m[R], L:m[L]}

    def apply_y(m):
        return {F:m[L], R:m[F], B:m[R], L:m[B], U:m[U], D:m[D]}

    # Start from standard
    std = {U:"white", D:"yellow", F:"green", B:"blue", R:"red", L:"orange"}

    queue = [std]
    while queue:
        current = queue.pop(0)
        key = orientation_key(current)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(current))
        queue.append(apply_x(current))
        queue.append(apply_y(current))

    return result


# ---------------------------------------------------------------------------
# Move translation
# ---------------------------------------------------------------------------

# When a face turns, how does that affect the ADJACENT faces' sticker layout?
# For the purposes of center tracking: face turns don't move centers.
# Only whole-cube rotations move centers.
# So we only need to detect whole-cube rotations and translate them.

# BLE move byte → (face_color, turns)
# turns: 1=CW, -1=CCW (from the perspective of looking AT that face)
BLE_MOVE_MAP = {
    0x00: ("blue",   1),
    0x01: ("blue",  -1),
    0x02: ("green",  1),
    0x03: ("green", -1),
    0x04: ("white",  1),
    0x05: ("white", -1),
    0x06: ("yellow", 1),
    0x07: ("yellow",-1),
    0x08: ("red",    1),
    0x09: ("red",   -1),
    0x0A: ("orange", 1),
    0x0B: ("orange",-1),
}

# Whole-cube rotation detection:
# If the cube's physical orientation changes but the internal state doesn't,
# the driver reports it as a face turn of a face that — given current orientation —
# corresponds to a rotation axis.
#
# However: the Rubik's Connected does NOT report whole-cube rotations as BLE events.
# It only reports actual face turns. So if you do x/y/z rotations, the BLE cube
# keeps track internally and the subsequent face turn events still correctly
# identify which color face turned — but if you re-orient the cube between moves,
# the BLE labels remain accurate (they always say WHICH COLOR turned, not which
# WCA face turned). Our job is just to map color→WCA face using our FaceMap.
#
# So there is NO whole-cube rotation detection needed from BLE events alone.
# The FaceMap handles it: as you re-orient, you call tracker.set_orientation()
# or use the IMU quaternion snap, and the mapping updates.


@dataclass
class MoveResult:
    """Result of processing one BLE move event through the orientation tracker."""
    ble_color:    str        # which color face the BLE cube reports turned
    ble_turns:    int        # 1=CW, -1=CCW from the face's own perspective
    wca_face:     str        # U/D/F/B/L/R in current camera orientation
    wca_notation: str        # full WCA notation: "R", "R'", "U2", etc.
    front_color:  str        # which color is on the F face at time of this move
    top_color:    str        # which color is on the U face at time of this move
    is_rotation:  bool = False
    rotation_name: str = ""
    # The move as the CAMERA saw it. Equal to wca_notation until a
    # middle-slice turn rotates the cube's centres away from the camera —
    # after that the cube keeps reporting in its own rotated frame and only
    # this field names the face that physically moved. See ble/slice_frame.py.
    camera_notation: str = ""
    # The camera<-cube orientation in force for this move, for the record.
    orientation: str = "identity"


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class OrientationTracker:
    """
    Maintains the cube's current orientation and translates BLE moves to WCA.

    Typical usage:
        tracker = OrientationTracker()
        tracker.calibrate(front_color="green", top_color="white")
        # ... for each BLE move event:
        result = tracker.apply_ble_move(raw_byte)
        print(result.wca_notation)
    """

    def __init__(self):
        self._face_map: Optional[FaceMap] = None
        self._move_history: list[MoveResult] = []
        self._calibrated = False
        # Camera<-cube frame. The cube reports relative to its centres, and a
        # middle-slice turn moves them; without this every label after the
        # first slice names the wrong face. See ble/slice_frame.py.
        self._frame = CameraFrame()
        self._prev: tuple[str, float] | None = None

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, front_color: str, top_color: str):
        """
        Set the current cube orientation by specifying which color is on
        the front face (pointing at camera) and which is on top.

        Call this:
          - At the start of every solve session
          - Any time you pick up the cube and reorient it significantly
          - After the IMU quaternion snap (internal use)

        Example:
            # Hold cube so green face is toward camera, white face is on top
            tracker.calibrate(front_color="green", top_color="white")
        """
        front_color = front_color.lower().strip()
        top_color   = top_color.lower().strip()

        if front_color not in COLORS:
            raise ValueError(f"Unknown color: {front_color!r}. Must be one of {COLORS}")
        if top_color not in COLORS:
            raise ValueError(f"Unknown color: {top_color!r}. Must be one of {COLORS}")
        if front_color == top_color:
            raise ValueError("Front and top cannot be the same color")

        self._face_map = FaceMap.from_front_and_top(front_color, top_color)
        self._calibrated = True
        # Calibration IS the statement that the two frames agree right now.
        self._frame.reset()
        self._prev = None
        print(f"[orientation] Calibrated: F={front_color}, U={top_color}")
        print(f"[orientation] Full map: {self._face_map}")

    def set_orientation_from_imu(self, qw: float, qx: float, qy: float, qz: float):
        """
        Snap the orientation from an IMU quaternion.
        Called periodically (every ~20 moves) to correct accumulated drift.

        The quaternion from the GoCube is relative to the cube's power-on
        orientation, NOT relative to gravity/north. So we use it as a
        differential correction rather than an absolute reference.

        This finds which of the 24 valid orientations is closest to the
        quaternion and snaps to that.
        """
        # Convert quaternion to rotation matrix
        R = _quat_to_rotation_matrix(qw, qx, qy, qz)

        # The "camera direction" in the cube's reference frame at power-on
        # We define it as the Z-axis (0, 0, 1) pointing toward the camera.
        # After rotation R, the face that is now closest to Z is the front face.
        z_axis = (R[0][2], R[1][2], R[2][2])

        # Map axis directions to faces:
        # The standard WCA orientation has these face normal vectors:
        #   U=(0,1,0), D=(0,-1,0), F=(0,0,1), B=(0,0,-1), R=(1,0,0), L=(-1,0,0)
        face_normals = {
            U: (0, 1, 0), D: (0, -1, 0),
            F: (0, 0, 1), B: (0, 0, -1),
            R: (1, 0, 0), L: (-1, 0, 0),
        }

        # Which face normal is most aligned with Z (camera direction)?
        # This tells us which WCA face of the STANDARD cube points toward camera
        best_face = max(face_normals, key=lambda f: _dot(face_normals[f], z_axis))

        # What color is currently on that face?
        if self._face_map:
            new_front = self._face_map.color_on(best_face)
        else:
            new_front = "green"  # default assumption

        # Similarly find which face points up (aligned with Y)
        y_axis = (R[0][1], R[1][1], R[2][1])
        best_up_face = max(face_normals, key=lambda f: _dot(face_normals[f], y_axis))
        if self._face_map:
            new_top = self._face_map.color_on(best_up_face)
        else:
            new_top = "white"

        # Snap
        try:
            new_map = FaceMap.from_front_and_top(new_front, new_top)
            self._face_map = new_map
        except ValueError:
            pass  # If snap produces invalid orientation, keep existing

    # ── Move processing ───────────────────────────────────────────────────────

    def apply_ble_move(self, raw_byte: int,
                       timestamp: Optional[float] = None) -> Optional[MoveResult]:
        """
        Process a single raw BLE FaceRotation byte.
        Returns a MoveResult, or None if the byte is unrecognised.

        `timestamp` (seconds) enables slice detection. The cube reports a
        middle-slice turn as a PAIR of opposite-face quarter turns with
        opposite handedness, arriving within one notification tick, because
        the core rotates with the slice — and that rotation carries the
        centres, which is what the cube's whole coordinate system is built
        on. Without a timestamp the pair cannot be told from two genuinely
        separate turns, so `camera_notation` degrades to `wca_notation` and
        the caller gets the pre-2026-08-03 behaviour rather than a guess.

        Both halves of a pair are labelled under the frame in force BEFORE
        the rotation, and the rotation takes effect for the move after them
        — the centres finish moving when the slice finishes. That is the
        same rule slice_frame.annotate applies offline, deliberately, so a
        session labelled live and one backfilled later cannot disagree.
        """
        if not self._calibrated:
            raise RuntimeError(
                "Tracker not calibrated. Call tracker.calibrate(front_color, top_color) first."
            )

        mapping = BLE_MOVE_MAP.get(raw_byte)
        if mapping is None:
            return None

        face_color, turns = mapping
        wca_face = self._face_map.face_of(face_color)
        wca_notation = _to_notation(wca_face, turns)

        result = MoveResult(
            ble_color    = face_color,
            ble_turns    = turns,
            wca_face     = wca_face,
            wca_notation = wca_notation,
            front_color  = self._face_map.color_on(F),
            top_color    = self._face_map.color_on(U),
            camera_notation = self._frame.camera_name(wca_notation),
            orientation  = self._frame.describe(),
        )

        if timestamp is not None:
            if self._prev is not None and is_slice_pair(
                    self._prev[0], self._prev[1], wca_notation, timestamp):
                self._frame.apply_slice(self._prev[0], wca_notation)
                self._prev = None          # a pair consumes both halves
            else:
                self._prev = (wca_notation, timestamp)

        self._move_history.append(result)
        return result

    def apply_move_event(self, move_event) -> Optional[MoveResult]:
        """
        Process a MoveEvent from cube_ble.py.
        Convenience wrapper around apply_ble_move that forwards the event's
        own timestamp, which is what makes slice detection work.
        """
        return self.apply_ble_move(move_event.raw_byte,
                                   getattr(move_event, "timestamp", None))

    @property
    def camera_frame(self) -> str:
        """The current camera<-cube orientation, 'identity' when they agree."""
        return self._frame.describe()

    @property
    def frame_drifted(self) -> bool:
        """True once a slice has rotated the cube's frame off the camera's."""
        return not self._frame.is_identity

    # ── Manual orientation update ─────────────────────────────────────────────

    def notify_reoriented(self, front_color: str, top_color: str):
        """
        Call this when you manually re-orient the cube between moves
        (e.g. you rotated the whole cube to show a different face to the camera).
        This is the same as calibrate() but without the "fresh session" semantics.
        """
        self.calibrate(front_color, top_color)

    # ── State queries ─────────────────────────────────────────────────────────

    @property
    def front_color(self) -> Optional[str]:
        """Color of the face currently pointing toward the camera."""
        if not self._face_map:
            return None
        return self._face_map.color_on(F)

    @property
    def top_color(self) -> Optional[str]:
        if not self._face_map:
            return None
        return self._face_map.color_on(U)

    @property
    def face_map(self) -> Optional[FaceMap]:
        return self._face_map

    @property
    def move_count(self) -> int:
        return len(self._move_history)

    def get_move_sequence(self) -> list[str]:
        """Return the WCA notation of all moves recorded this session."""
        return [m.wca_notation for m in self._move_history]

    def reset_history(self):
        """Clear move history (call between solves)."""
        self._move_history = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_notation(face: str, turns: int) -> str:
    """Convert WCA face + turns to standard notation string."""
    turns = turns % 4
    if turns == 1:
        return face
    elif turns == 3:
        return face + "'"
    elif turns == 2:
        return face + "2"
    return ""  # 0 turns = no-op


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _quat_to_rotation_matrix(w, x, y, z):
    """Convert unit quaternion to 3x3 rotation matrix."""
    return [
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests():
    print("Running OrientationTracker tests...\n")

    # Test 1: all 24 orientations are generated
    orients = _all_24_orientations()
    assert len(orients) == 24, f"Expected 24 orientations, got {len(orients)}"
    print(f"  [PASS] Generated all 24 valid cube orientations")

    # Test 2: standard calibration
    t = OrientationTracker()
    t.calibrate("green", "white")
    assert t.front_color == "green"
    assert t.top_color   == "white"
    print(f"  [PASS] Calibration: F=green, U=white")

    # Test 3: in standard WCA orientation, red-CW = R
    t2 = OrientationTracker()
    t2.calibrate("green", "white")  # standard WCA
    result = t2.apply_ble_move(0x08)  # red CW
    assert result.wca_face     == R,   f"Expected R, got {result.wca_face}"
    assert result.wca_notation == "R", f"Expected 'R', got {result.wca_notation}"
    print(f"  [PASS] Red CW → R (standard orientation)")

    # Test 4: red-CCW = R'
    t3 = OrientationTracker()
    t3.calibrate("green", "white")
    result = t3.apply_ble_move(0x09)  # red CCW
    assert result.wca_notation == "R'", f"Expected R', got {result.wca_notation}"
    print(f"  [PASS] Red CCW → R' (standard orientation)")

    # Test 5: rotated orientation — cube flipped so red face is toward camera
    t4 = OrientationTracker()
    t4.calibrate("red", "white")   # red on F, white on U
    result = t4.apply_ble_move(0x08)  # red CW — but red is now F, not R
    assert result.wca_face == F, f"Expected F, got {result.wca_face}"
    print(f"  [PASS] Red CW with red on F → F (rotated orientation)")

    # Test 6: FaceMap from_front_and_top consistency
    fm = FaceMap.from_front_and_top("green", "white")
    assert fm.color_on(F) == "green"
    assert fm.color_on(U) == "white"
    assert fm.color_on(D) == "yellow"
    assert fm.color_on(B) == "blue"
    assert fm.color_on(R) == "red"
    assert fm.color_on(L) == "orange"
    print(f"  [PASS] FaceMap correctly derives all 6 faces from F+U")

    # Test 7: whole-cube rotation updates face map
    fm2 = FaceMap.from_front_and_top("green", "white")
    fm2_y = fm2.apply_whole_rotation("y", 1)  # y rotation: F→R, so green→R
    assert fm2_y.color_on(R) == "green", f"After y rotation, green should be on R, got {fm2_y.color_on(R)}"
    assert fm2_y.color_on(F) == "orange", f"After y rotation, orange should be on F, got {fm2_y.color_on(F)}"
    print(f"  [PASS] Whole-cube y-rotation correctly updates face map")

    # Test 8: notation encoding
    assert _to_notation("R",  1) == "R"
    assert _to_notation("U", -1) == "U'"
    assert _to_notation("F",  2) == "F2"
    assert _to_notation("R",  3) == "R'"
    print(f"  [PASS] WCA notation encoding")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orientation tracker utilities")
    parser.add_argument("--test",      action="store_true", help="Run self-tests")
    parser.add_argument("--live",      action="store_true", help="Live demo with BLE cube")
    parser.add_argument("--front",     type=str, default="green", help="Front face color")
    parser.add_argument("--top",       type=str, default="white",  help="Top face color")
    parser.add_argument("--address",   type=str, default=None)
    args = parser.parse_args()

    if args.test:
        _run_tests()

    elif args.live:
        import asyncio
        from cube_ble import CubeConnection

        async def live_demo():
            tracker = OrientationTracker()
            tracker.calibrate(args.front, args.top)
            print(f"\nListening for moves (F={args.front}, U={args.top})...")
            print("Ctrl+C to stop\n")

            async with CubeConnection(address=args.address) as cube:
                async for ble_move in cube.moves():
                    result = tracker.apply_move_event(ble_move)
                    if result:
                        print(f"  BLE: {ble_move.face_color:<8} {ble_move.direction:<4}  →  "
                              f"WCA: {result.wca_notation:<4}  "
                              f"(F={result.front_color}, U={result.top_color})")

        asyncio.run(live_demo())

    else:
        parser.print_help()