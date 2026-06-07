"""Vision helpers: resolve `image_url` content into PIL Images.

Supports:

- `data:image/<fmt>;base64,<payload>` — decoded in-process. Compute-node
  safe (no network).
- `file:///abs/path.png` — loaded from local disk.
- `https://…` — fetched with `requests`. *Will fail on compute nodes
  isolated from the public internet.* Pass base64 instead.

Returns a `PIL.Image.Image` ready for the model's processor.
"""
from __future__ import annotations

import base64
import io
from urllib.parse import urlparse

from PIL import Image


def load_image(url: str) -> Image.Image:
    parsed = urlparse(url)
    if parsed.scheme == "data":
        # data:image/png;base64,xxxxx
        _, _, payload = url.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if parsed.scheme == "file":
        return Image.open(parsed.path).convert("RGB")
    if parsed.scheme in ("http", "https"):
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("Install 'requests' to fetch http(s) images") from e
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r} (url={url[:80]!r})")


__all__ = ["load_image"]
