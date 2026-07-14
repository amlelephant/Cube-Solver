import asyncio
import threading
import win_compat
from cube_ble import CubeConnection

async def main():
    cube = CubeConnection()

    def notify():
        cube._on_notification(None, bytearray([0x2A, 0x05, 0x01, 0x00, 0x00, 0x00]))

    t = threading.Thread(target=notify)
    t.start()
    t.join()
    print('done')

asyncio.run(main())
