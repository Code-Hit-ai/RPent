"""Opt-in per-request evidence without headers or inline image bodies."""

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path


def sanitize(value):
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if not isinstance(value, dict):
        return value
    if (
        value.get("type") == "input_image"
        and isinstance(value.get("image_url"), str)
        and value["image_url"].startswith("data:")
    ):
        data = base64.b64decode(value["image_url"].split(",", 1)[1])
        return dict(
            value,
            image_url={"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
        )
    return {k: sanitize(v) for k, v in value.items()}


def capture_body(path, raw, encoding):
    if encoding == "gzip":
        raw = gzip.decompress(raw)
    elif encoding == "zstd":
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    Path(path).write_text(json.dumps(sanitize(json.loads(raw)), ensure_ascii=False))
