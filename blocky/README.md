# blocky

AI-powered system-level ad blocker. Uses vision LLM to detect ads on screen, then locks onto their pixels and tracks them in real time — no repeated AI calls needed.

## How it works

1. **Screenshot** (`screenshot.py`) — captures the screen every 5 seconds
2. **Detect** (`detector.py`) — sends the screenshot to Claude's vision API, gets back bounding boxes of every ad region
3. **Track** (`tracker.py`) — extracts the pixel signature of each detected ad and uses OpenCV template matching to follow it across frames, even when the user scrolls. No more LLM calls after the initial detection
4. **Block** (`overlay.py`) — draws opaque rectangles over tracked ad regions using a transparent fullscreen overlay

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
# Run with overlay (blocks ads on screen)
python main.py

# Headless mode (detection + tracking, no overlay — good for testing)
python main.py --headless

# Custom screenshot interval
python main.py --interval 10

# Debug mode (saves annotated frames to /tmp/adblocker_debug/)
ADBLOCKER_DEBUG=1 python main.py
```

## Requirements

- Python 3.11+
- Linux with X11 (for the transparent overlay)
- Anthropic API key
- A display server (for `mss` screen capture)
