import asyncio
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import win_compat
from cube_ble import CubeConnection


def test_move_notification_from_thread_is_queued():
    cube = CubeConnection()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_test():
        await asyncio.sleep(0)
        move = None

        async def consume():
            nonlocal move
            move = await cube.moves().__anext__()

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.01)

        def notify():
            cube._on_notification(None, bytearray([0x2A, 0x05, 0x01, 0x00, 0x00, 0x00]))

        t = threading.Thread(target=notify)
        t.start()
        t.join()

        await asyncio.wait_for(consumer, timeout=1.0)
        assert move is not None
        assert move.raw_byte == 0x00

    loop.run_until_complete(run_test())
    loop.close()
