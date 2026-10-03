# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay sanitized request captures using hash-verified original images."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

from .codex import CodexAdapter
from .features import Features
from .session import ContextSession


def restore(value, images):
    """Restore only image bodies whose hashes match the recorded content."""
    if isinstance(value, list):
        return [restore(x, images) for x in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "input_image" and isinstance(value.get("image_url"), dict):
        ref = value["image_url"]
        data = images[ref["sha256"]]
        if len(data) != ref["bytes"]:
            raise ValueError("recorded image size does not match")
        return dict(
            value, image_url="data:image/png;base64," + base64.b64encode(data).decode()
        )
    return {k: restore(v, images) for k, v in value.items()}


def main() -> None:
    """Write selection metrics without calling a model or simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--encoder",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("image", "text", "image_text", "action", "all"),
        default="image",
    )
    args = parser.parse_args()
    images = {}
    for path in args.episode.rglob("*.png"):
        if path.is_file():
            data = path.read_bytes()
            images[hashlib.sha256(data).hexdigest()] = data
    adapter = CodexAdapter(tuple(args.cameras.split(",")))
    session = ContextSession(adapter, Features(args.encoder), mode=args.mode)
    full = ContextSession(adapter, None)
    with args.output.open("w") as output:
        for path in sorted((args.episode / "requests").glob("*.json")):
            request = restore(json.loads(path.read_text())["body"], images)
            assert full.process(request) is request
            before = json.dumps(request)
            result = session.process(request)
            assert json.dumps(request) == before, "original request mutated"
            # Every original non-image field and tool pair must be unchanged.
            records = adapter.extract(request)
            for record in records:
                block = result["input"][record.item]["output"][record.block]
                if block.get("type") != "input_image":
                    assert not record.protected
            expected = adapter.render(
                request,
                records,
                {d["id"] for d in session.stats["records"] if d["selected"]}
                if "records" in session.stats
                else {r.id for r in records},
            )
            if "text" in session.stats:
                text_records, _ = adapter.extract_text(expected)
                expected = adapter.render_text(
                    expected,
                    text_records,
                    {
                        d["id"]
                        for d in session.stats["text"]["records"]
                        if d["selected"]
                    },
                )
            if "action" in session.stats:
                action_records = adapter.extract_action(expected)
                expected = adapter.render_action(
                    expected,
                    action_records,
                    {
                        d["id"]
                        for d in session.stats["action"]["records"]
                        if d["selected"]
                    },
                )
            assert expected == result
            output.write(json.dumps(dict(request=path.name, **session.stats)) + "\n")


if __name__ == "__main__":
    main()
