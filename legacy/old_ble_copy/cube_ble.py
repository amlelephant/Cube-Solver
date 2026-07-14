"""
cube_ble.py

Connects to a Rubik's Connected (GoCube) cube over Bluetooth and streams
move events to Python in real time.

The Rubik's Connected is a GoCube hardware variant. Its BLE protocol is
fully documented at: https://github.com/oddpetersson/gocube-protocol

What this module provides
-------------------------
  scan()          - find nearby GoCube devices and print their addresses
  CubeConnection  - async context manager that connects and streams events

Usage — one-shot scan:
    python cube_ble.py --scan

Usage — live move stream:
    python cube_ble.py --stream

Usage — record a full solve to a JSON file:
    python cube_ble.py --record solve_001.json

Usage — in your own code:
    import asyncio
    from cube_ble import CubeConnection

    async def main():
        async with CubeConnection() as cube:
            state = await cube.get_state()
            print("Starting state:", state)

            async for event in cube.moves():
                print(event)   # {"face": "R", "dir": "CW", "ts": 1234567890.123}

    asyncio.run(main())

Requirements:
    pip install bleak
"""

import asyncio
import json
import sys
import time
import argparse
from dataclasses import dataclass, asdict
from typing import AsyncIterator, Optional

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False

# ---------------------------------------------------------------------------
# GoCube BLE constants (from protocol documentation)
# ---------------------------------------------------------------------------

PRIMARY_SERVICE  = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
RX_CHARACTERISTIC = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write commands here
TX_CHARACTERISTIC = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # subscribe here for events

# Commands to write to RX
CMD_GET_STATE   = bytes([0x33])
CMD_GET_BATTERY = bytes([0x32])
CMD_GET_CUBE_TYPE = bytes([0x56])
CMD_DISABLE_ORIENTATION = bytes([0x37])  # kills the 15/sec quaternion spam
CMD_ENABLE_ORIENTATION  = bytes([0x38])

# Message types in TX notifications
MSG_ROTATION    = 0x01
MSG_STATE       = 0x02
MSG_ORIENTATION = 0x03
MSG_BATTERY     = 0x05

# FaceRotation byte → (face_name, direction)
# The face name here is the CENTER COLOR, not WCA notation.
# We translate to WCA notation using the cube's current orientation below.
FACE_ROTATION_MAP = {
    0x00: ("blue",   "CW"),
    0x01: ("blue",   "CCW"),
    0x02: ("green",  "CW"),
    0x03: ("green",  "CCW"),
    0x04: ("white",  "CW"),
    0x05: ("white",  "CCW"),
    0x06: ("yellow", "CW"),
    0x07: ("yellow", "CCW"),
    0x08: ("red",    "CW"),
    0x09: ("red",    "CCW"),
    0x0A: ("orange", "CW"),
    0x0B: ("orange", "CCW"),
}

# Center color → WCA face letter
# WCA convention: U=white, D=yellow, F=green, B=blue, R=red, L=orange
COLOR_TO_WCA = {
    "white":  "U",
    "yellow": "D",
    "green":  "F",
    "blue":   "B",
    "red":    "R",
    "orange": "L",
}

# State message color bytes → color name
STATE_COLOR_MAP = {
    0x00: "blue",
    0x01: "green",
    0x02: "white",
    0x03: "yellow",
    0x04: "red",
    0x05: "orange",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MoveEvent:
    face_color: str    # e.g. "red"
    face_wca:   str    # e.g. "R"
    direction:  str    # "CW" or "CCW"
    notation:   str    # standard notation e.g. "R", "R'", "U2" (2 not yet detected)
    timestamp:  float  # time.time()
    raw_byte:   int    # raw FaceRotation byte from protocol

    def to_dict(self):
        return asdict(self)


@dataclass
class CubeState:
    """Full 54-sticker state from MsgState."""
    faces: dict        # {"blue": [c,c,c,...], "green": [...], ...}
    timestamp: float

    def to_kociemba(self):
        """
        Convert to kociemba state string (URFDLB order).
        Returns None if state is incomplete.
        """
        color_to_face = {"white":"U","red":"R","green":"F",
                         "yellow":"D","orange":"L","blue":"B"}
        face_order = ["white","red","green","yellow","orange","blue"]
        result = ""
        for face_color in face_order:
            stickers = self.faces.get(face_color, [])
            if len(stickers) != 9:
                return None
            for color in stickers:
                result += color_to_face.get(color, "?")
        return result if len(result) == 54 and "?" not in result else None


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------

def _parse_message(data: bytearray):
    """
    Parse a raw BLE notification byte array.
    Returns (msg_type, payload) or (None, None) on parse failure.

    Common message format:
      [0]        0x2A  prefix
      [1]        length (includes prefix but excludes suffix)
      [2]        message type
      [3..n-1]   payload
      [n]        checksum = sum(bytes[0..n-1]) % 0x100
      [n+1,n+2]  0x0D 0x0A suffix
    """
    if len(data) < 5:
        return None, None
    if data[0] != 0x2A:
        return None, None

    length   = data[1]
    msg_type = data[2]
    payload  = data[3:length - 1]

    # Validate checksum
    expected_checksum = sum(data[:length - 1]) % 0x100
    actual_checksum   = data[length - 1]
    if expected_checksum != actual_checksum:
        return None, None

    return msg_type, payload


def _parse_rotation(payload: bytearray) -> list[MoveEvent]:
    """
    Parse MsgRotation payload.
    Payload is n*2 bytes: [FaceRotation, Orientation] per move.
    Multiple moves can be batched in one message.
    """
    events = []
    ts = time.time()

    for i in range(0, len(payload) - 1, 2):
        face_byte = payload[i]
        # orientation_byte = payload[i + 1]  # not needed for move logging

        mapping = FACE_ROTATION_MAP.get(face_byte)
        if mapping is None:
            continue

        face_color, direction = mapping
        face_wca  = COLOR_TO_WCA.get(face_color, "?")
        notation  = face_wca if direction == "CW" else face_wca + "'"

        events.append(MoveEvent(
            face_color = face_color,
            face_wca   = face_wca,
            direction  = direction,
            notation   = notation,
            timestamp  = ts,
            raw_byte   = face_byte,
        ))

    return events


def _parse_state(payload: bytearray) -> Optional[CubeState]:
    """
    Parse MsgState payload (60 bytes).
    Returns CubeState with all 6 faces populated.
    """
    if len(payload) < 54:
        return None

    face_order = ["blue","green","white","yellow","red","orange"]
    faces = {}

    for i, face_name in enumerate(face_order):
        offset = i * 9
        # First byte is the center (always the face color)
        center = STATE_COLOR_MAP.get(payload[offset], "?")
        # Next 8 bytes are the surrounding stickers (clockwise from top-left)
        stickers = [STATE_COLOR_MAP.get(payload[offset + j], "?")
                    for j in range(1, 9)]
        # Reconstruct full 9-sticker face: [TL, T, TR, R, BR, B, BL, L, CENTER]
        # Reorder to row-major: top-left to bottom-right
        # Protocol order: TL=1, T=2, TR=3, R=4, BR=5, B=6, BL=7, L=8, Center=0
        tl, t, tr, r, br, b, bl, l_ = stickers
        full_face = [tl, t, tr, l_, center, r, bl, b, br]
        faces[face_name] = full_face

    return CubeState(faces=faces, timestamp=time.time())


# ---------------------------------------------------------------------------
# CubeConnection
# ---------------------------------------------------------------------------

class CubeConnection:
    """
    Async context manager for a GoCube BLE connection.

    Usage:
        async with CubeConnection(address="XX:XX:XX:XX:XX:XX") as cube:
            # address is optional — auto-scans if omitted
            async for move in cube.moves():
                print(move.notation)
    """

    def __init__(self, address: str = None, timeout: float = 10.0):
        self.address    = address
        self.timeout    = timeout
        self._client    = None
        self._move_queue: asyncio.Queue = asyncio.Queue()
        self._state_future: Optional[asyncio.Future] = None
        self._battery_future: Optional[asyncio.Future] = None

    async def __aenter__(self):
        if not BLEAK_AVAILABLE:
            raise RuntimeError(
                "bleak not installed. Run: pip install bleak"
            )

        if self.address is None:
            print("Scanning for GoCube / Rubik's Connected...")
            self.address = await _find_cube(self.timeout)
            if self.address is None:
                raise RuntimeError(
                    "No GoCube found.\n"
                    "Make sure the cube is on (twist a face) and not connected\n"
                    "to another device (close the Rubik's app first)."
                )
            print(f"Found cube at {self.address}")

        self._client = BleakClient(self.address, timeout=self.timeout)
        await self._client.connect()
        print(f"Connected to {self.address}")

        # Subscribe to TX notifications
        await self._client.start_notify(TX_CHARACTERISTIC, self._on_notification)

        # Disable orientation spam (15/sec quaternion messages we don't need)
        await self._client.write_gatt_char(RX_CHARACTERISTIC,
                                           CMD_DISABLE_ORIENTATION,
                                           response=False)

        return self

    async def __aexit__(self, *args):
        if self._client and self._client.is_connected:
            await self._client.stop_notify(TX_CHARACTERISTIC)
            await self._client.disconnect()
            print("Disconnected.")

    def _on_notification(self, sender, data: bytearray):
        """Called by bleak on every BLE notification from the cube."""
        msg_type, payload = _parse_message(data)
        if msg_type is None:
            return

        if msg_type == MSG_ROTATION:
            moves = _parse_rotation(payload)
            for move in moves:
                self._move_queue.put_nowait(move)

        elif msg_type == MSG_STATE:
            state = _parse_state(payload)
            if state and self._state_future and not self._state_future.done():
                self._state_future.get_event_loop().call_soon_threadsafe(
                    self._state_future.set_result, state
                )

        elif msg_type == MSG_BATTERY:
            if payload and self._battery_future and not self._battery_future.done():
                level = payload[0]
                self._battery_future.get_event_loop().call_soon_threadsafe(
                    self._battery_future.set_result, level
                )

    async def moves(self) -> AsyncIterator[MoveEvent]:
        """Async generator that yields MoveEvent as each face is turned."""
        while True:
            move = await self._move_queue.get()
            yield move

    async def get_state(self) -> Optional[CubeState]:
        """Request the current full 54-sticker state. Waits up to 3 seconds."""
        loop = asyncio.get_event_loop()
        self._state_future = loop.create_future()
        await self._client.write_gatt_char(RX_CHARACTERISTIC,
                                           CMD_GET_STATE,
                                           response=False)
        try:
            state = await asyncio.wait_for(self._state_future, timeout=3.0)
            return state
        except asyncio.TimeoutError:
            print("WARNING: get_state timed out — no response from cube")
            return None
        finally:
            self._state_future = None

    async def get_battery(self) -> Optional[int]:
        """Request battery percentage. Waits up to 3 seconds."""
        loop = asyncio.get_event_loop()
        self._battery_future = loop.create_future()
        await self._client.write_gatt_char(RX_CHARACTERISTIC,
                                           CMD_GET_BATTERY,
                                           response=False)
        try:
            level = await asyncio.wait_for(self._battery_future, timeout=3.0)
            return level
        except asyncio.TimeoutError:
            return None
        finally:
            self._battery_future = None


# ---------------------------------------------------------------------------
# Scanner helper
# ---------------------------------------------------------------------------

async def _find_cube(timeout: float = 10.0) -> Optional[str]:
    """
    Scan for BLE devices advertising the GoCube primary service UUID.
    Returns the address of the first matching device, or None.
    """
    found = None

    def callback(device, adv_data):
        nonlocal found
        if found:
            return
        # Match by service UUID
        if PRIMARY_SERVICE.lower() in [s.lower() for s in (adv_data.service_uuids or [])]:
            found = device.address
        # Also match by device name prefix as a fallback
        name = (device.name or "").lower()
        if any(x in name for x in ["gocube", "rubik", "cube"]):
            found = device.address

    async with BleakScanner(callback):
        deadline = time.time() + timeout
        while not found and time.time() < deadline:
            await asyncio.sleep(0.2)

    return found


async def scan_and_print():
    """Print all nearby BLE devices that look like cubes."""
    if not BLEAK_AVAILABLE:
        print("ERROR: bleak not installed. Run: pip install bleak")
        return

    print("Scanning 8 seconds... (twist your cube to wake it up)")
    devices = await BleakScanner.discover(timeout=8.0)

    cubes = []
    others = []
    for d in devices:
        name = (d.name or "").lower()
        if any(x in name for x in ["gocube", "rubik", "cube"]):
            cubes.append(d)
        else:
            others.append(d)

    if cubes:
        print(f"\nLikely cube devices ({len(cubes)}):")
        for d in cubes:
            print(f"  {d.address}  {d.name}")
    else:
        print("\nNo obvious cube devices found.")
        print("If your cube is on, it may advertise under a generic name.")
        print("All devices seen:")
        for d in others[:15]:
            print(f"  {d.address}  {d.name}")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

async def _stream(address=None):
    """Print every move in real time."""
    async with CubeConnection(address=address) as cube:
        battery = await cube.get_battery()
        if battery is not None:
            print(f"Battery: {battery}%")

        state = await cube.get_state()
        if state:
            ks = state.to_kociemba()
            print(f"Starting state: {ks or '(could not parse)'}")

        print("\nListening for moves... (Ctrl+C to stop)\n")
        async for move in cube.moves():
            print(f"  {move.notation:<4}  ({move.face_color} {move.direction})  "
                  f"t={move.timestamp:.3f}")


async def _record(output_path: str, address=None):
    """
    Record a full solve session to a JSON file.

    Format:
    {
      "start_state": "<kociemba string>",
      "end_state":   "<kociemba string>",
      "moves": [
        {"notation": "R", "face_color": "red", "direction": "CW",
         "timestamp": 1234567890.123, "raw_byte": 8},
        ...
      ]
    }
    """
    moves_log = []

    async with CubeConnection(address=address) as cube:
        battery = await cube.get_battery()
        if battery is not None:
            print(f"Battery: {battery}%")

        print("Getting start state...")
        start_state = await cube.get_state()
        start_ks = start_state.to_kociemba() if start_state else None
        print(f"Start state: {start_ks or '(unknown)'}")

        print("\nSolving... (Ctrl+C when done)\n")

        try:
            async for move in cube.moves():
                moves_log.append(move.to_dict())
                move_num = len(moves_log)
                print(f"  [{move_num:3d}] {move.notation}")
        except KeyboardInterrupt:
            print("\nStopped. Getting end state...")

        end_state = await cube.get_state()
        end_ks = end_state.to_kociemba() if end_state else None
        print(f"End state: {end_ks or '(unknown)'}")

    record = {
        "start_state": start_ks,
        "end_state":   end_ks,
        "move_count":  len(moves_log),
        "moves":       moves_log,
    }

    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nSaved {len(moves_log)} moves to {output_path}")
    if start_ks and end_ks:
        print(f"Start: {start_ks}")
        print(f"End:   {end_ks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Connect to Rubik's Connected / GoCube via Bluetooth"
    )
    parser.add_argument("--scan",    action="store_true",
                        help="Scan for nearby cube devices")
    parser.add_argument("--stream",  action="store_true",
                        help="Stream moves live to terminal")
    parser.add_argument("--record",  type=str, metavar="FILE",
                        help="Record a solve session to a JSON file")
    parser.add_argument("--address", type=str, default=None,
                        help="Bluetooth address to connect to (skip scan)")
    args = parser.parse_args()

    if not BLEAK_AVAILABLE:
        print("ERROR: bleak not installed.")
        print("Run: pip install bleak")
        sys.exit(1)

    if args.scan:
        asyncio.run(scan_and_print())
    elif args.stream:
        asyncio.run(_stream(args.address))
    elif args.record:
        asyncio.run(_record(args.record, args.address))
    else:
        parser.print_help()