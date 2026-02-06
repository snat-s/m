"""Pixel-based ad region tracker.

After the AI detector identifies ad regions in a frame, this module captures
a pixel "fingerprint" of each region. On subsequent frames it uses OpenCV
template matching to relocate the region purely from pixel data — no further
LLM calls needed.  This handles scrolling, minor layout shifts, and
animations cheaply and in real time.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import TRACKER_MATCH_THRESHOLD, TRACKER_SEARCH_MARGIN


@dataclass
class TrackedRegion:
    """A single tracked ad region with its pixel template."""
    id: str
    x: int
    y: int
    w: int
    h: int
    label: str
    template: np.ndarray          # BGR patch grabbed from the detection frame
    last_seen: float = 0.0        # timestamp of last successful match
    lost_count: int = 0           # consecutive frames where match failed
    active: bool = True
    confidence: float = 1.0       # last match score

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class TrackerState:
    """Holds all currently tracked regions."""
    regions: dict[str, TrackedRegion] = field(default_factory=dict)
    _next_id: int = 0

    def next_id(self) -> str:
        rid = f"ad_{self._next_id}"
        self._next_id += 1
        return rid


class PixelTracker:
    """Tracks previously-detected ad regions across frames using pixel matching."""

    def __init__(
        self,
        match_threshold: float = TRACKER_MATCH_THRESHOLD,
        search_margin: int = TRACKER_SEARCH_MARGIN,
        max_lost_frames: int = 10,
    ):
        self._match_threshold = match_threshold
        self._search_margin = search_margin
        self._max_lost = max_lost_frames
        self.state = TrackerState()

    # -- public API -----------------------------------------------------------

    def register(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        label: str = "ad",
    ) -> str:
        """Register a new ad region from the current frame.

        Extracts the pixel template and starts tracking it.
        Returns the tracking id.
        """
        fh, fw = frame.shape[:2]
        # Clamp to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            raise ValueError("Region too small to track")

        template = frame[y1:y2, x1:x2].copy()
        rid = self.state.next_id()
        self.state.regions[rid] = TrackedRegion(
            id=rid,
            x=x1, y=y1, w=x2 - x1, h=y2 - y1,
            label=label,
            template=template,
            last_seen=time.time(),
        )
        return rid

    def update(self, frame: np.ndarray) -> list[TrackedRegion]:
        """Update all tracked regions against a new frame.

        Returns a list of regions that are still active (found in this frame).
        """
        active: list[TrackedRegion] = []
        to_remove: list[str] = []

        for rid, region in self.state.regions.items():
            if not region.active:
                continue

            found = self._find_region(frame, region)
            if found:
                nx, ny, score = found
                region.x = nx
                region.y = ny
                region.confidence = score
                region.last_seen = time.time()
                region.lost_count = 0
                active.append(region)
            else:
                region.lost_count += 1
                if region.lost_count >= self._max_lost:
                    region.active = False
                    to_remove.append(rid)
                else:
                    # Still report it at last-known position (might just be
                    # a momentary occlusion or transition frame).
                    active.append(region)

        for rid in to_remove:
            del self.state.regions[rid]

        return active

    def clear(self) -> None:
        """Drop all tracked regions (e.g. before a fresh detection pass)."""
        self.state.regions.clear()

    def active_regions(self) -> list[TrackedRegion]:
        return [r for r in self.state.regions.values() if r.active]

    # -- internals ------------------------------------------------------------

    def _find_region(
        self, frame: np.ndarray, region: TrackedRegion
    ) -> tuple[int, int, float] | None:
        """Try to locate ``region.template`` in ``frame`` near its last position.

        Returns (x, y, score) or None if not found above threshold.
        """
        fh, fw = frame.shape[:2]
        th, tw = region.template.shape[:2]

        # Define a search window around last known position
        sx1 = max(0, region.x - self._search_margin)
        sy1 = max(0, region.y - self._search_margin)
        sx2 = min(fw, region.x + region.w + self._search_margin)
        sy2 = min(fh, region.y + region.h + self._search_margin)

        # Make sure the search window is at least as big as the template
        if (sx2 - sx1) < tw or (sy2 - sy1) < th:
            # Fall back to full-frame search
            sx1, sy1 = 0, 0
            sx2, sy2 = fw, fh

        search_area = frame[sy1:sy2, sx1:sx2]

        if search_area.shape[0] < th or search_area.shape[1] < tw:
            return None

        result = cv2.matchTemplate(search_area, region.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= self._match_threshold:
            abs_x = sx1 + max_loc[0]
            abs_y = sy1 + max_loc[1]
            return (abs_x, abs_y, float(max_val))

        return None
