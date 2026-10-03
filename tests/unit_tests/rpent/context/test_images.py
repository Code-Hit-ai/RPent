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

"""Offline protocol and selection tests."""

import base64
import copy
import gzip
import io
import json

import pytest

from rpent.context.codex import CodexAdapter
from rpent.context.features import Features
from rpent.context.proxy import rewrite_body
from rpent.context.selection import select_by_nc
from rpent.context.session import ContextSession


def request():
    Image = pytest.importorskip("PIL.Image")

    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    items = [{"type": "reasoning", "encrypted_content": "opaque"}]
    for i in range(6):
        items.extend(
            [
                {"type": "custom_tool_call", "call_id": str(i), "input": "observe"},
                {
                    "type": "custom_tool_call_output",
                    "call_id": str(i),
                    "id": f"out{i}",
                    "output": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "step": i,
                                    "state": {},
                                    "log": {"command": {"action": "observe"}},
                                    "task_language": "pick",
                                }
                            ),
                        },
                        {"type": "input_image", "image_url": url},
                    ],
                },
            ]
        )
    return {"input": items, "model": "example", "stream": True}


class FakeFeatures(Features):
    def __init__(self):
        super().__init__("unused")

    def similarity(self, text, query):
        return 0.5


def test_nc_edges_and_recency():
    assert select_by_nc({"a": 1, "b": 1}, 0.5) == {"b"}
    assert select_by_nc({"a": 0}, 0.5) == {"a"}
    assert select_by_nc({}, 0.5) == set()
    with pytest.raises(ValueError):
        select_by_nc({}, 0)


def test_full_preserves_compressed_wire_bytes():
    raw = gzip.compress(json.dumps(request()).encode())
    session = ContextSession(CodexAdapter(), None)
    assert rewrite_body(raw, "gzip", session) == (raw, False)


def test_selection_only_changes_old_images():
    source = request()
    original = copy.deepcopy(source)
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="image")
    result = session.process(source)
    assert source == original
    assert 0 < session.stats["images_after"] < 6
    assert session.stats["nc"] >= 0.5
    for i, (before, after) in enumerate(zip(source["input"], result["input"])):
        if before != after:
            assert i < 9  # latest two tool cycles remain intact
            before = copy.deepcopy(before)
            before["output"][1] = after["output"][1]
            assert before == after
    assert (
        session.process(source) == result
    )  # repeated full history does not change novelty


def test_unknown_camera_and_incremental_requests_preserved():
    source = request()
    session = ContextSession(CodexAdapter(), FakeFeatures(), mode="image")
    assert session.process(source) is source
    source["previous_response_id"] = "server-held-history"
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="image")
    assert session.process(source) is source


def test_latest_observation_survives_readonly_calls():
    source = request()
    for i in range(3):
        source["input"].append(
            {"type": "custom_tool_call_output", "call_id": f"read{i}", "output": []}
        )
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="image")
    result = session.process(source)
    assert result["input"][12] == source["input"][12]


def test_modified_body_is_uncompressed_json():
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="image")
    raw = gzip.compress(json.dumps(request()).encode())
    body, changed = rewrite_body(raw, "gzip", session)
    assert changed
    assert json.loads(body)["stream"] is True


def test_dhash_horizontal_direction():
    Image = pytest.importorskip("PIL.Image")
    features = Features("unused")
    hashes = []
    for name, row in (
        ("forward", list(range(0, 225, 25))),
        ("reverse", list(range(200, -1, -25))),
    ):
        image = Image.new("L", (9, 8))
        image.putdata(row * 8)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        hashes.append(features.dhash(name, stream.getvalue()))
    assert hashes == [(1 << 64) - 1, 0]


def test_omitted_images_do_not_return_or_participate_in_comparison():
    source = request()
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="image")
    session.process(source)
    removed = {
        r["id"]
        for r in session.stats["records"]
        if not r["selected"] and r.get("reason") != "initial_observation"
    }
    assert removed
    # Append a new native interaction, retaining the complete original input.
    new_call, new_output = copy.deepcopy(source["input"][-2:])
    new_call["call_id"] = new_output["call_id"] = "next"
    new_output["id"] = "out-next"
    source["input"].extend([new_call, new_output])
    session.process(source)
    for record in session.stats["records"]:
        if record["id"] in removed:
            assert not record["selected"]
            assert record["score"] is None
            assert record["novelty"] is None
            assert record["reason"] == "previously_omitted"
    result = session.process(source)
    assert session.process(source) == result


def test_initial_observation_is_only_kept_while_protected():
    source = request()
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="image")
    first = dict(source, input=source["input"][:3])
    assert session.process(first) is first
    record = session.stats["records"][0]
    assert record["selected"] and record["score"] is None
    session.process(source)
    initial = session.stats["records"][0]
    assert not initial["selected"]
    assert initial["score"] is None
    assert initial["reason"] == "initial_observation"
    # First non-initial frame has no eligible predecessor; step zero is excluded.
    assert session.stats["records"][1]["novelty"] is None
    assert session.stats["records"][1]["score"] == 0.25
    # Identical subsequent image has measured zero novelty, not missing novelty.
    assert session.stats["records"][2]["novelty"] == 0.0
    assert session.stats["records"][2]["score"] == 0.25

import copy
import pytest
from rpent.context.codex import CodexAdapter
from rpent.context.session import ContextSession


@pytest.mark.parametrize("style", ["custom_tool_call", "function_call"])
def test_native_formats_preserve_protocol(style):
    source = request()
    for item in source["input"]:
        if item["type"] == "custom_tool_call":
            item["type"] = style
            if style == "function_call":
                item["arguments"] = "{}"
                item.pop("input")
        elif item["type"] == "custom_tool_call_output":
            item["type"] = style + "_output"
    original = copy.deepcopy(source)
    adapter = CodexAdapter(("front",))
    assert len(adapter.extract(source)) == 6
    assert len(adapter.extract_action(source)) == 6
    session = ContextSession(adapter, FakeFeatures(), mode="all")
    result = session.process(source)
    assert source == original
    assert 0 < session.stats["images_after"] < 6
    for before, after in zip(source["input"], result["input"]):
        assert before["type"] == after["type"]
        assert before.get("call_id") == after.get("call_id")
        if before["type"] in ("reasoning", style):
            assert before == after
    assert result["input"][-1] == source["input"][-1]

def test_mismatched_native_pair_is_not_selected():
    source = request()
    for item in source["input"]:
        if item["type"] == "custom_tool_call":
            item["type"] = "function_call"
    adapter = CodexAdapter(("front",))
    assert adapter.extract(source) == []
    assert adapter.extract_action(source) == []


def test_missing_novelty_does_not_outscore_a_relevant_changed_image():
    class ScoredFeatures(FakeFeatures):
        def similarity(self, text, query):
            action = json.loads(text)["action"]
            return {"old": 0.430, "changed": 0.285}.get(action, 0.0)

        def dhash(self, digest, data):
            return 0 if data == b"old" else (1 << 24) - 1

    source = request()
    for i in range(6):
        output = source["input"][2 + 2 * i]
        snapshot = json.loads(output["output"][0]["text"])
        snapshot["log"]["command"]["action"] = (
            "old" if i == 1 else "changed" if i == 2 else "other"
        )
        output["output"][0]["text"] = json.dumps(snapshot)
        data = b"old" if i <= 1 else b"changed"
        output["output"][1]["image_url"] = (
            "data:image/png;base64," + base64.b64encode(data).decode()
        )
    original = copy.deepcopy(source)
    session = ContextSession(
        CodexAdapter(("front",)), ScoredFeatures(), mode="image", rho=0.5
    )
    result = session.process(source)
    records = {r["id"]: r for r in session.stats["records"]}
    assert not records["out1:1"]["selected"]
    assert records["out2:1"]["selected"]
    assert records["out1:1"]["novelty"] is None
    assert records["out2:1"]["novelty"] == 0.375
    assert records["out4:1"]["selected"] and records["out5:1"]["selected"]
    assert source == original
    assert session.process(source) == result
