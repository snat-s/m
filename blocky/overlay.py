"""Screen overlay module.

Draws opaque rectangles over detected ad regions using a transparent
Tkinter window that sits on top of everything.  The overlay positions
are updated in real time from the tracker state.
"""

import threading
import tkinter as tk
from typing import Callable

from config import OVERLAY_COLOR, OVERLAY_REFRESH_RATE


class ScreenOverlay:
    """Transparent full-screen window that draws opaque blocks over ad regions."""

    def __init__(
        self,
        get_regions: Callable[[], list[tuple[int, int, int, int]]],
        color: tuple[int, int, int, int] = OVERLAY_COLOR,
        refresh_rate: float = OVERLAY_REFRESH_RATE,
    ):
        """
        Args:
            get_regions: Callable returning list of (x, y, w, h) to block.
            color: RGBA fill color for blocking rectangles.
            refresh_rate: How often (seconds) to refresh overlay positions.
        """
        self._get_regions = get_regions
        self._color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        self._refresh_ms = int(refresh_rate * 1000)
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Launch the overlay in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except tk.TclError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # -- internals ------------------------------------------------------------

    def _run_tk(self) -> None:
        self._root = tk.Tk()
        self._root.title("AI AdBlocker Overlay")

        # Full-screen transparent window on top of everything
        self._root.attributes("-fullscreen", True)
        self._root.attributes("-topmost", True)
        self._root.wait_visibility(self._root)

        # Make the window background transparent
        # On Linux with compositing:
        self._root.attributes("-alpha", 1.0)
        try:
            # This makes the window "click-through" on Linux (X11)
            self._root.wm_attributes("-type", "dock")
        except tk.TclError:
            pass

        # Use a transparent background color as the "pass-through" color
        transparent_color = "gray1"  # near-black that we'll make transparent
        self._root.config(bg=transparent_color)
        try:
            self._root.wm_attributes("-transparentcolor", transparent_color)
        except tk.TclError:
            # Fallback: the window won't be click-through but will still work
            pass

        self._canvas = tk.Canvas(
            self._root,
            bg=transparent_color,
            highlightthickness=0,
            cursor="none",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Start the refresh loop
        self._refresh()
        self._root.mainloop()

    def _refresh(self) -> None:
        if not self._running or self._canvas is None:
            return

        self._canvas.delete("adblock")

        regions = self._get_regions()
        for (x, y, w, h) in regions:
            self._canvas.create_rectangle(
                x, y, x + w, y + h,
                fill=self._color_hex,
                outline=self._color_hex,
                tags="adblock",
            )

        if self._root is not None:
            self._root.after(self._refresh_ms, self._refresh)
