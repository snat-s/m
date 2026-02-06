"""AI-based advertisement detector.

Sends a screenshot to an LLM with vision capabilities and receives back
bounding-box coordinates for every region classified as an advertisement.
"""

import base64
import json
import re
from dataclasses import dataclass

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DETECTION_CONFIDENCE_THRESHOLD,
)


@dataclass
class AdRegion:
    """A detected advertisement region on screen."""
    x: int       # top-left x
    y: int       # top-left y
    w: int       # width
    h: int       # height
    confidence: float
    label: str   # short description e.g. "banner ad", "sponsored post"

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


DETECTION_PROMPT = """\
You are an advertisement detector. Analyze this screenshot and identify every \
region that is an advertisement, sponsored content, promotional banner, or \
marketing material.

For EACH ad region found, return a JSON object with these fields:
- x: top-left x coordinate (pixels from left edge of screenshot)
- y: top-left y coordinate (pixels from top edge of screenshot)
- w: width in pixels
- h: height in pixels
- confidence: your confidence this is an ad (0.0 to 1.0)
- label: short description (e.g. "banner ad", "sponsored post", "video ad overlay")

Return your answer as a JSON array. If there are NO ads, return an empty array [].

Example output:
```json
[
  {"x": 120, "y": 50, "w": 728, "h": 90, "confidence": 0.95, "label": "banner ad"},
  {"x": 800, "y": 300, "w": 300, "h": 250, "confidence": 0.87, "label": "sidebar sponsored"}
]
```

IMPORTANT: Return ONLY the JSON array, no other text. Be precise with coordinates.
"""


class AdDetector:
    """Detects ads in screenshots using LLM vision."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or ANTHROPIC_API_KEY
        if not key:
            raise ValueError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY env var or pass api_key=."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model or ANTHROPIC_MODEL

    def detect(self, png_bytes: bytes) -> list[AdRegion]:
        """Send a PNG screenshot to the LLM and return detected ad regions."""
        b64_image = base64.standard_b64encode(png_bytes).decode("utf-8")

        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": DETECTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = message.content[0].text.strip()
        return self._parse_response(raw_text)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> list[AdRegion]:
        # Strip markdown fences if present
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[AdDetector] Failed to parse LLM response:\n{text}")
            return []

        if not isinstance(items, list):
            return []

        regions: list[AdRegion] = []
        for item in items:
            try:
                region = AdRegion(
                    x=int(item["x"]),
                    y=int(item["y"]),
                    w=int(item["w"]),
                    h=int(item["h"]),
                    confidence=float(item.get("confidence", 1.0)),
                    label=str(item.get("label", "ad")),
                )
                if region.confidence >= DETECTION_CONFIDENCE_THRESHOLD:
                    regions.append(region)
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[AdDetector] Skipping malformed region {item}: {exc}")

        return regions
