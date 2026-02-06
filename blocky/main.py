#!/usr/bin/env python3
"""AI Ad Blocker — main orchestrator.

Flow:
  1. Every 5 seconds, capture a screenshot.
  2. Send it to the LLM to detect ad regions (bounding boxes).
  3. For each detected region, extract its pixel template and register it
     with the pixel tracker.
  4. Between LLM calls, the tracker relocates every registered ad region
     using fast template matching — no further LLM calls needed.
  5. A transparent overlay draws opaque blocks over every tracked ad region
     in real time, so the user never sees the ads.

The LLM is only called once every 5 seconds (or whatever the configured
interval is).  All the frame-to-frame following is done purely with
OpenCV pixel matching, which is essentially free.
"""

import argparse
import signal
import sys
import threading
import time

import numpy as np

from config import DEBUG, SCREENSHOT_INTERVAL_SECONDS
from screenshot import ScreenCapture
from detector import AdDetector, AdRegion
from tracker import PixelTracker
from overlay import ScreenOverlay


class AdBlocker:
    """Ties together screenshot → detection → tracking → overlay."""

    def __init__(self, headless: bool = False):
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._headless = headless

        # Components
        self._capture = ScreenCapture(
            on_frame=self._on_frame,
            interval=SCREENSHOT_INTERVAL_SECONDS,
        )
        self._detector = AdDetector()
        self._tracker = PixelTracker()
        self._overlay: ScreenOverlay | None = None

        if not headless:
            self._overlay = ScreenOverlay(get_regions=self._get_block_rects)

        # Detection thread runs LLM calls off the capture thread
        self._detect_flag = threading.Event()
        self._detect_thread: threading.Thread | None = None
        self._running = False

        self._stats = {"llm_calls": 0, "ads_found": 0, "tracker_updates": 0}

    # -- public API -----------------------------------------------------------

    def run(self) -> None:
        """Start all components and block until interrupted."""
        self._running = True

        print("[AdBlocker] Starting screen capture...")
        self._capture.start()

        print("[AdBlocker] Starting tracker update loop...")
        self._tracker_thread = threading.Thread(
            target=self._tracker_loop, daemon=True
        )
        self._tracker_thread.start()

        print("[AdBlocker] Starting detection thread...")
        self._detect_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self._detect_thread.start()

        if self._overlay is not None:
            print("[AdBlocker] Starting overlay...")
            self._overlay.start()

        print("[AdBlocker] Running. Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        print("\n[AdBlocker] Shutting down...")
        self._running = False
        self._detect_flag.set()  # unblock detection thread
        self._capture.stop()
        if self._overlay is not None:
            self._overlay.stop()
        self._print_stats()

    # -- callbacks & loops ----------------------------------------------------

    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Called by ScreenCapture every interval with a new frame."""
        with self._lock:
            self._latest_frame = frame
        # Signal the detection thread that a new frame is ready
        self._detect_flag.set()

    def _detection_loop(self) -> None:
        """Runs LLM detection on each new frame from the capture thread."""
        while self._running:
            self._detect_flag.wait(timeout=SCREENSHOT_INTERVAL_SECONDS + 1)
            self._detect_flag.clear()

            if not self._running:
                break

            with self._lock:
                frame = self._latest_frame
            if frame is None:
                continue

            try:
                png_bytes = self._capture.capture_png_bytes()
                print("[AdBlocker] Sending screenshot to AI for ad detection...")
                regions = self._detector.detect(png_bytes)
                self._stats["llm_calls"] += 1

                if regions:
                    print(
                        f"[AdBlocker] Detected {len(regions)} ad region(s): "
                        f"{[r.label for r in regions]}"
                    )
                    self._register_detections(frame, regions)
                else:
                    print("[AdBlocker] No ads detected in this frame.")
            except Exception as exc:
                print(f"[AdBlocker] Detection error: {exc}")

    def _register_detections(
        self, frame: np.ndarray, regions: list[AdRegion]
    ) -> None:
        """Register newly detected ad regions with the pixel tracker."""
        # Clear old tracked regions and replace with fresh detections.
        # This ensures we pick up new ads and drop ones that disappeared.
        self._tracker.clear()
        for region in regions:
            try:
                rid = self._tracker.register(
                    frame,
                    x=region.x, y=region.y,
                    w=region.w, h=region.h,
                    label=region.label,
                )
                self._stats["ads_found"] += 1
                if DEBUG:
                    print(f"  [Tracker] Registered {rid}: {region}")
            except ValueError as exc:
                print(f"  [Tracker] Skipping region: {exc}")

    def _tracker_loop(self) -> None:
        """High-frequency loop that updates tracked region positions."""
        while self._running:
            with self._lock:
                frame = self._latest_frame
            if frame is not None and self._tracker.active_regions():
                self._tracker.update(frame)
                self._stats["tracker_updates"] += 1
            time.sleep(0.05)  # ~20 fps tracking

    def _get_block_rects(self) -> list[tuple[int, int, int, int]]:
        """Return current bounding boxes to block (called by overlay)."""
        return [r.bbox for r in self._tracker.active_regions()]

    # -- debug ----------------------------------------------------------------

    def _print_stats(self) -> None:
        print(f"[AdBlocker] Stats: {self._stats}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-powered system-level ad blocker"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the screen overlay (useful for testing/debug)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help=f"Screenshot interval in seconds (default: {SCREENSHOT_INTERVAL_SECONDS})",
    )
    args = parser.parse_args()

    if args.interval is not None:
        import config
        config.SCREENSHOT_INTERVAL_SECONDS = args.interval

    blocker = AdBlocker(headless=args.headless)

    # Graceful shutdown on SIGTERM
    signal.signal(signal.SIGTERM, lambda *_: blocker.stop())

    blocker.run()


if __name__ == "__main__":
    main()
