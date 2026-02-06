"""Configuration for the AI ad blocker."""

import os

# --- Screenshot ---
SCREENSHOT_INTERVAL_SECONDS = 5

# --- AI Detection ---
# Uses Anthropic Claude vision by default. Set your API key:
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Minimum confidence (0-1) to consider a detection valid
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# --- Pixel Tracker ---
# How closely a tracked region must match to keep following it (0-1).
# Lower = more lenient matching. Uses normalized cross-correlation.
TRACKER_MATCH_THRESHOLD = 0.6

# How many pixels to expand the search area around the last known position
# when looking for a shifted ad region (handles small scrolls between frames).
TRACKER_SEARCH_MARGIN = 200

# --- Overlay ---
# Color used to cover ad regions (R, G, B, A).  Fully opaque by default.
OVERLAY_COLOR = (30, 30, 30, 255)

# How often (seconds) the overlay positions are refreshed from tracker state.
OVERLAY_REFRESH_RATE = 0.05  # 50 ms – smooth following

# --- General ---
# Set to True to write annotated debug screenshots to /tmp/adblocker_debug/
DEBUG = os.environ.get("ADBLOCKER_DEBUG", "0") == "1"
