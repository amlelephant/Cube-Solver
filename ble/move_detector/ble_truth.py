"""
ble_truth.py

Ground truth from the smart cube, recorded live and in parallel with the
webcam, so a verification take can say WHICH move was wrong rather than
only that something was.

Why this exists
---------------
`verify_solve.py`'s phase 1 gets ground truth by prescribing a scramble and
assuming you performed it correctly.  That assumption is doing real work and
it is invisible when it breaks: perform move 14 backwards and the cube is
not in the state phase 2 asserts it was, so phase 2 fails for a human reason
that looks exactly like a pipeline failure.  Phase 2 has no per-move truth
at all — that is the point of it as a *product* test, but it makes it
useless as a *debugging* test, and a 60-move solve is not something anyone
can recall afterwards.

The BLE cube closes both gaps for free.  It reports every turn it feels,
timestamped on the same wall clock as the webcam frames, so:

  * phase 1 learns the scramble you ACTUALLY performed, and can verify
    against that instead of against the one it printed;
  * phase 2 gets a full per-move ground truth for a free solve, which no
    other path in this repo can produce;
  * both phases can name the specific move where the pipeline and reality
    diverged.

This is a debugging instrument, not part of the product claim.  The claim is
still "decided from webcam frames alone" — the BLE stream is never an input
to the detector, classifier or decode, only a label to score them against.
`record_training.py` already pairs BLE with the webcam this way; the
difference here is that nothing is written to disk and the connection is
held across both phases of one sitting.

Threading
---------
The BLE stack owns an asyncio loop and must have a clean MTA COM apartment
on Windows, so it runs on its own thread (same reasoning as
`record_training.py`).  Moves land in a list under a lock; the capture
thread never blocks on BLE and BLE never blocks on capture.  A move drain
task and `get_state()` requests coexist on that loop, which is safe because
`CubeConnection` routes them through separate futures.

    truth = BleTruth(front="blue", top="yellow")
    if truth.start():
        ...                                   # record with the webcam
        word = truth.words_between(t0, t1)    # WCA moves in that window
        truth.stop()
"""

import sys
from pathlib import Path

# ble/ holds cube_ble, orientation_tracker and win_compat.
_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

# MUST precede cv2/pywin32 anywhere in the process — see win_compat's
# docstring. Importing this module early is how verify_solve.py satisfies
# that, since live_detect.py pulls in cv2.
import win_compat  # noqa: F401,E402

import asyncio     # noqa: E402
import threading   # noqa: E402
import time        # noqa: E402

from cube_ble import CubeConnection                            # noqa: E402
from orientation_tracker import OrientationTracker, COLORS     # noqa: E402

# A solved cube in the vendor's colour-fixed URFDLB facelet convention.
# Orientation-independent: whatever way up a solved cube is held, every
# face is uniform, so this comparison needs no camera-frame relabelling.
SOLVED_FACELETS = "".join(c * 9 for c in "URFDLB")

# Moves are timestamped when their BLE notification is parsed, which trails
# the physical turn by the cube's own reporting latency. Widening the
# window at both ends keeps a turn made just before SPACE-stop from being
# dropped for arriving just after it.
WINDOW_PAD = 0.35


class BleTruth:
    """
    A live BLE move log, windowable by wall-clock time.

    Every public method is safe to call from the capture thread. `start()`
    returns False rather than raising if no cube can be reached — a
    verification take without ground truth is degraded, not impossible, and
    the caller decides which it wants.
    """

    def __init__(self, front: str, top: str, address: str | None = None,
                 echo: bool = True):
        for name, colour in (("front", front), ("top", top)):
            if colour not in COLORS:
                raise ValueError(
                    f"--{name} {colour!r} is not a cube colour; expected one "
                    f"of {', '.join(COLORS)}")
        self.front, self.top, self.address = front, top, address
        self.echo = echo

        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cube: CubeConnection | None = None

        self.connected = False
        self.battery: int | None = None
        self.error: Exception | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 30.0) -> bool:
        """Connect and begin logging. Blocks until connected or failed."""
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ble-truth")
        self._thread.start()
        self._ready.wait(timeout=timeout)
        if not self.connected and self.error is None:
            self.error = TimeoutError(
                f"cube did not connect within {timeout:.0f}s")
        return self.connected

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        self.connected = False

    def _run(self):
        """Fresh thread = clean MTA COM apartment, as record_training.py."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._session())
        except Exception as exc:            # connection refused, no adapter
            self.error = exc
        finally:
            self.connected = False
            self._ready.set()               # unblock start() on failure
            try:
                loop.close()
            except Exception:
                pass

    async def _session(self):
        tracker = OrientationTracker()
        tracker.calibrate(self.front, self.top)
        try:
            async with CubeConnection(address=self.address) as cube:
                self._cube = cube
                self.battery = await cube.get_battery()
                self.connected = True
                self._ready.set()

                # The drain runs as its own task rather than inline, so the
                # loop stays free to service get_state() from the capture
                # thread — and so stop() does not have to wait for the next
                # turn to be felt before it can shut down.
                drain = asyncio.create_task(self._drain(cube, tracker))
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)
                drain.cancel()
                try:
                    await drain
                except asyncio.CancelledError:
                    pass
        except Exception as exc:
            self.error = exc
        finally:
            self.connected = False
            self._ready.set()

    async def _drain(self, cube, tracker):
        async for ev in cube.moves():
            result = tracker.apply_move_event(ev)
            # Field names mirror record_training.py's moves.jsonl exactly, so
            # a saved take is readable by postprocess_session.py /
            # prepare_data.py with no special-casing.
            rec = {"ts": ev.timestamp,
                   "wca": result.wca_notation if result else None,
                   "ble_color": ev.face_color,
                   "ble_direction": ev.direction,
                   "ble_raw": ev.raw_byte,
                   "wca_face": result.wca_face if result else None,
                   "front_color": result.front_color if result else self.front,
                   "top_color": result.top_color if result else self.top}
            with self._lock:
                self._events.append(rec)
            if self.echo:
                n = len(self._events)
                print(f"    BLE [{n:3d}] {ev.face_color:<7} "
                      f"{ev.direction:<3} -> {rec['wca'] or '?'}")

    # -- reading -----------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def moves_between(self, t0: float, t1: float,
                      pad: float = WINDOW_PAD) -> list[dict]:
        """Logged moves whose timestamps fall inside a recording window."""
        with self._lock:
            return [e for e in self._events if t0 - pad <= e["ts"] <= t1 + pad]

    def words_between(self, t0: float, t1: float,
                      pad: float = WINDOW_PAD) -> list[str]:
        """
        The same window as a WCA move word.

        Half turns are dropped with a warning rather than silently: the
        classifier's 12 classes are the quarter turns, so a `U2` in the
        truth would be scored wrong no matter how the pipeline performed
        and would corrupt the measurement it is supposed to anchor. The
        cube reports each physical half turn as two events anyway; this
        only fires if the tracker coalesced them.
        """
        out, dropped = [], []
        for e in self.moves_between(t0, t1, pad):
            w = e["wca"]
            if w is None:
                dropped.append("?")
            elif w.endswith("2"):
                dropped.append(w)
            else:
                out.append(w)
        if dropped:
            print(f"  WARNING: dropped {len(dropped)} non-quarter-turn BLE "
                  f"event(s) from the truth: {' '.join(dropped[:8])}")
        return out

    def snapshot_state(self, timeout: float = 6.0) -> str | None:
        """
        Ask the cube for its full 54-sticker state, from any thread.

        Returns the vendor's colour-fixed URFDLB facelet string, which is
        directly comparable to SOLVED_FACELETS but NOT to a camera-relative
        state from reconstruct.py — for that, build the state from the move
        word instead (`RC.seq_to_state(word)`), which is already in camera
        coordinates because the orientation tracker put it there.
        """
        if not self.connected or self._loop is None or self._cube is None:
            return None
        try:
            fut = asyncio.run_coroutine_threadsafe(self._cube.get_state(),
                                                   self._loop)
            state = fut.result(timeout=timeout)
        except Exception:
            return None
        return state.to_kociemba() if state else None

    def is_solved(self, timeout: float = 6.0) -> bool | None:
        """True/False, or None when the cube did not answer."""
        s = self.snapshot_state(timeout)
        return None if s is None else s == SOLVED_FACELETS

    def solved_report(self, timeout: float = 6.0) -> tuple[bool | None, int]:
        """
        (is_solved, facelets_wrong) — the distance matters as much as the
        verdict.

        A bare "NOT SOLVED" cannot distinguish a cube genuinely one turn
        short from a cube whose own internal state tracking has drifted,
        and those need opposite responses: turn the last face, versus stop
        trusting the ground truth entirely. One quarter turn misplaces 6
        facelets (and any n-turn scramble misplaces a multiple of ~6-ish),
        so a large count means the cube is confused rather than unsolved.
        This is not hypothetical — a cube reporting 0% battery did exactly
        that on 2026-07-23, alongside emitting phantom moves.
        """
        s = self.snapshot_state(timeout)
        if s is None:
            return None, 0
        wrong = sum(a != b for a, b in zip(s, SOLVED_FACELETS))
        return s == SOLVED_FACELETS, wrong

    # -- health ------------------------------------------------------------

    # A cube turning faster than this is not a person. The good takes on
    # 2026-07-23 bottomed out at a 60ms inter-move gap with a 181-270ms
    # median; the failure mode was ~10 identical moves per second with the
    # cube sitting still, so anything under 50ms on the SAME face is the
    # hardware talking, not the hand.
    SPAM_GAP = 0.05

    def health(self) -> dict:
        """
        Signs that the cube's own reporting cannot be trusted.

        Never drops events — a filter that silently discards real moves
        would corrupt the ground truth in the one direction that is
        impossible to notice. This only reports, so the take can be judged
        rather than quietly believed.
        """
        with self._lock:
            events = list(self._events)
        bursts = 0
        for a, b in zip(events, events[1:]):
            if b["ts"] - a["ts"] < self.SPAM_GAP and b["wca"] == a["wca"]:
                bursts += 1
        low_battery = self.battery is not None and self.battery <= 15
        return {"events": len(events), "repeat_bursts": bursts,
                "battery": self.battery, "low_battery": low_battery,
                "suspect": bursts > 0 or low_battery}

    def warn_if_unhealthy(self) -> bool:
        """Print a diagnosis if the cube looks unreliable. True if suspect."""
        h = self.health()
        if not h["suspect"]:
            return False
        print(f"\n  CUBE HEALTH WARNING")
        if h["low_battery"]:
            print(f"    battery {h['battery']}% — a low cube both invents "
                  f"moves and lets its own\n    internal state drift, which "
                  f"corrupts the ground truth AND makes the\n    solved "
                  f"check report nonsense. Charge it before trusting a take.")
        if h["repeat_bursts"]:
            print(f"    {h['repeat_bursts']} repeated move(s) under "
                  f"{self.SPAM_GAP*1000:.0f}ms on the same face — that is "
                  f"the\n    phantom-move signature, not a human hand. The "
                  f"move log for this take\n    is not trustworthy.")
        return True


# ---------------------------------------------------------------------------
# Comparing a performed word against a prescribed one
# ---------------------------------------------------------------------------

def compare_words(prescribed: list[str], performed: list[str],
                  max_show: int = 12) -> dict:
    """
    Did the human do what they were told?

    Reported separately from every model metric on purpose. A mis-performed
    scramble and a mis-read scramble produce the same downstream symptom —
    phase 2 verifying against a state the cube was never in — and the only
    way to tell them apart is to ask the cube what actually happened.
    """
    from decode import align_sequences

    ops = align_sequences(prescribed, performed)
    counts = {k: sum(1 for o, _, _ in ops if o == k)
              for k in ("ok", "sub", "miss", "phantom")}
    match = performed == prescribed

    print(f"\n  performed vs prescribed: ", end="")
    if match:
        print(f"IDENTICAL ({len(performed)} moves) — the cube is in the "
              f"state\n    the next phase will assert.")
        return {"match": True, "ops": ops, "counts": counts}

    print(f"DIFFERENT")
    print(f"    prescribed {len(prescribed)} moves, "
          f"you performed {len(performed)}")
    print(f"    {counts['sub']} turned the wrong way or wrong face, "
          f"{counts['miss']} skipped, "
          f"{counts['phantom']} extra")

    shown, i = 0, 0
    for op, want, got in ops:
        if op != "ok":
            if shown == 0:
                print(f"\n    {'#':>4}  {'prescribed':<11} {'you did'}")
            if shown < max_show:
                mark = {"sub": "X", "miss": "-", "phantom": "+"}[op]
                print(f"    {i+1:>3}{mark} {want or '.':<11} {got or '.'}")
            shown += 1
        if want is not None:
            i += 1
    if shown > max_show:
        print(f"    ... and {shown - max_show} more")

    print(f"\n    Verifying against what you ACTUALLY did, not what was "
          f"printed.\n    The cube's real state is the one that matters.")
    return {"match": False, "ops": ops, "counts": counts}
