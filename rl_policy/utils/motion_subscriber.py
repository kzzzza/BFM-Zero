"""ZMQ subscriber for streaming ``MotionFrameMessage`` to the online tracking policy.

A daemon thread blocks on ``recv()`` and pushes deserialized messages into a
bounded ``collections.deque``. The policy loop drains this queue with
``poll()`` (non-blocking) at its 50 Hz cadence.

CRITICAL: this subscriber uses ``CONFLATE=0``. Conflation would silently drop
intermediate frames, which breaks the finite-difference body velocity inside
``BackwardObsBuilder`` and corrupts the computed z. If frames legitimately
arrive out of order or late, ``OnlineZProvider`` handles it by resetting; we
must not pre-filter them here.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import zmq
from loguru import logger

from utils.common import MotionFrameMessage, PORTS


class MotionSubscriber:
    def __init__(
        self,
        port: int = PORTS["motion_frame"],
        ip: str = "localhost",
        maxlen: int = 64,
        recv_timeout_ms: int = 10,
    ):
        self.port = int(port)
        self.ip = ip

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        # Do NOT enable CONFLATE — see module docstring.
        self._socket.connect(f"tcp://{ip}:{port}")
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))

        self._queue: deque[MotionFrameMessage] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        logger.info(f"MotionSubscriber listening on tcp://{ip}:{port}")

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                return
            except Exception as exc:
                logger.warning(f"MotionSubscriber recv error: {exc}")
                continue

            try:
                msg = MotionFrameMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"MotionSubscriber failed to decode frame: {exc}")
                continue

            self._queue.append(msg)

    def poll(self) -> Optional[MotionFrameMessage]:
        """Pop the oldest queued frame, or ``None`` if the queue is empty."""
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def drain_latest(self) -> Optional[MotionFrameMessage]:
        """Discard everything except the most recent queued frame and return it.

        Useful for recovering after an underrun where the policy thread fell
        behind the producer.
        """
        last: Optional[MotionFrameMessage] = None
        while True:
            try:
                last = self._queue.popleft()
            except IndexError:
                break
        return last

    def qsize(self) -> int:
        return len(self._queue)

    def close(self) -> None:
        self._stop.set()
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
        self._thread.join(timeout=1.0)
