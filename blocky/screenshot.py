"""Screen capture module.

Grabs the full screen (or a specified monitor) at a configurable interval
and hands the raw image to whoever needs it.
"""

import io
import time
import threading
from typing import Callable

import mss
import numpy as np
from PIL import Image

from config import SCREENSHOT_INTERVAL_SECONDS


class ScreenCapture:
    """Periodically captures the screen and invokes a callback with the frame."""

    def __init__(
        self,
        on_frame: Callable[[np.ndarray, float], None],
        interval: float = SCREENSHOT_INTERVAL_SECONDS,
        monitor: int = 0,
    ):
        """
        Args:
            on_frame: Called with (frame_bgr, timestamp) every interval.
            interval: Seconds between captures.
            monitor: MSS monitor index (0 = all monitors combined).
        """
        self._on_frame = on_frame
        self._interval = interval
        self._monitor = monitor
        self._running = False
        self._thread: threading.Thread | None = None

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            self._thread = None

    def capture_once(self) -> np.ndarray:
        """Take a single screenshot and return it as a BGR numpy array."""
        with mss.mss() as sct:
            mon = sct.monitors[self._monitor]
            raw = sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.rgb)
            return np.array(img)[:, :, ::-1]  # RGB -> BGR for OpenCV compat

    def capture_png_bytes(self) -> bytes:
        """Take a single screenshot and return PNG bytes (for sending to LLM)."""
        with mss.mss() as sct:
            mon = sct.monitors[self._monitor]
            raw = sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.rgb)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    @property
    def screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the target monitor."""
        with mss.mss() as sct:
            mon = sct.monitors[self._monitor]
            return mon["width"], mon["height"]

    # -- internals ------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            ts = time.time()
            frame = self.capture_once()
            try:
                self._on_frame(frame, ts)
            except Exception as exc:
                print(f"[ScreenCapture] callback error: {exc}")
            elapsed = time.time() - ts
            sleep_time = max(0, self._interval - elapsed)
            time.sleep(sleep_time)
