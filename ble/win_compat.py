"""
win_compat.py

Must be the FIRST import in any script that uses bleak on Windows.
Sets COM threading to MTA before any other package can set it to STA.

    import win_compat   # ← line 1, before everything else
    import asyncio
    import cv2
    ...

Why this exists:
    bleak's WinRT backend requires MTA (Multi-Threaded Apartment) COM threading.
    Packages like pywin32, opencv-python, and others silently initialize COM to
    STA (Single-Threaded Apartment) on import, which breaks bleak's async
    callbacks with: "Thread is configured for Windows GUI but callbacks are
    not working."

    sys.coinit_flags = 0 tells pythoncom to use MTA when it initializes.
    It must be set before pythoncom is imported by anything, which means
    it must be the very first thing that runs.

    uninitialize_sta() is a belt-and-suspenders call that forcibly resets
    the COM model even if something already initialized it to STA.
"""

import sys

if sys.platform == "win32":
    # Best effort: some Python/pywin32 setups don't expose sys.coinit_flags on
    # 3.13, so we fall back to explicitly initializing the COM apartment as MTA
    # before bleak starts scanning or connecting.
    try:
        sys.coinit_flags = 0
    except AttributeError:
        pass

    try:
        from bleak.backends.winrt.util import uninitialize_sta
        uninitialize_sta()
    except (ImportError, AttributeError, OSError):
        pass

    try:
        import pythoncom
        pythoncom.CoInitializeEx(0)  # COINIT_MULTITHREADED
    except Exception:
        pass