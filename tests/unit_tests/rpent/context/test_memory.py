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

"""Memory parent restoration, re-entry and prompt isolation contracts."""

import copy
import json
from types import SimpleNamespace

import pytest

from rpent.context.memory import END, START, MemorySelector
from rpent.context.session import ContextSession


class Features:
    def __init__(self):
        self.scores = {}

    def similarity(self, text, query):
        return self.scores.get(text, 0.1)


class Adapter:
    task = "Pick object"

    def extract_action(self, request):
        return [SimpleNamespace(snapshot={"task_language": self.task})]


def setup(tmp_path):
    blocks = [
        {"id": "identify", "text": "Object is blue; distinguish it."},
        {"id": "place", "text": "Place inside basket."},
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(blocks))
    memory = "\n" + "\n".join(block["text"] for block in blocks) + "\n"
    request = {
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Rules"}]},
            {
                "role": "user",
                "id": "u1",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Goal\n{START}{memory}{END}\nFixed rules",
                    }
                ],
            },
            {"type": "reasoning", "encrypted_content": "opaque"},
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Found the basket. Pick object."}
                ],
            },
            {
                "type": "custom_tool_call_output",
                "output": [{"type": "input_text", "text": "DO NOT USE THIS AS QUERY"}],
            },
        ],
        "tools": [{"name": "move"}],
    }
    return MemorySelector(path), request, Features(), Adapter()


def test_parent_completion_and_only_public_sentence_query(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"distinguish it.": 0.9}
    original = copy.deepcopy(request)
    result, stats = selector.process(request, adapter, features)
    assert stats["query"] == "Pick object."
    assert stats["selected_blocks"] == ["identify"]
    assert stats["selection_nc"] >= 0.5
    assert "Object is blue; distinguish it." in result["input"][1]["content"][0]["text"]
    assert "Place inside basket." not in result["input"][1]["content"][0]["text"]
    assert request == original
    restored = copy.deepcopy(result)
    restored["input"][1]["content"][0]["text"] = original["input"][1]["content"][0][
        "text"
    ]
    assert restored == original


def test_deselected_experience_can_return(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"distinguish it.": 0.9}
    first, stats = selector.process(request, adapter, features)
    assert stats["omitted_blocks"] == ["place"]
    features.scores = {"Place inside basket.": 0.9}
    second, stats = selector.process(first, adapter, features)
    assert stats["selected_blocks"] == ["place"]
    features.scores = {"distinguish it.": 0.9}
    third, stats = selector.process(second, adapter, features)
    assert stats["selected_blocks"] == ["identify"]
    assert "distinguish it." in third["input"][1]["content"][0]["text"]


def test_unknown_task_keeps_all_and_empty_query_selects_none(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    adapter.task = ""
    result, stats = selector.process(request, adapter, features)
    assert result is request
    assert stats["selected_blocks"] == ["identify", "place"]
    adapter.task = "Pick object"
    request["input"][3]["content"] = []
    _, stats = selector.process(request, adapter, features)
    assert stats["selected_blocks"] == []
    assert stats["query"] == ""


def test_full_memory_threshold_and_missing_region(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    selector.rho = 1.0
    result, stats = selector.process(request, adapter, features)
    assert result is request
    assert stats["selection_nc"] == 1.0
    request["input"][1]["content"][0]["text"] = "Missing markers"
    with pytest.raises(ValueError, match="managed user region"):
        selector.process(request, adapter, features)


def test_full_session_preserves_memory_stats_and_retries(tmp_path):
    selector, request, features, _ = setup(tmp_path)
    from rpent.context.codex import CodexAdapter

    session = ContextSession(CodexAdapter(), features, memory_selector=selector)
    first = session.process(request)
    stats = copy.deepcopy(session.stats["memory"])
    assert session.process(request) == first
    assert session.stats["memory"] == stats
    assert session.stats["decision"]["reused"]


@pytest.mark.parametrize(
    "content",
    [
        "【抓取】\n夹住目标并抬起。\n\n【搬运】\n保持物体高于障碍物。\n",
        "【Grasp】\nGrasp the package.\n\n【Transfer】\nKeep clearance above groceries.\n",
    ],
)
def test_titled_demonstration_loads_complete_blocks(tmp_path, content):
    from rpent.context.memory import append_memory, load_memory_blocks

    path = tmp_path / "memory.txt"
    path.write_text(content, encoding="utf-8")
    blocks = load_memory_blocks(path)
    assert [b["id"] for b in blocks] == ["block_001", "block_002"]
    assert blocks[0]["text"] == content.split("\n\n")[0]
    prompt = append_memory("Fixed rules", blocks)
    assert prompt.startswith("Fixed rules")
    assert prompt.count(START) == prompt.count(END) == 1
    assert blocks[0]["text"] in prompt
    assert blocks[1]["text"] in prompt
    assert append_memory("Fixed rules", []) == "Fixed rules"


@pytest.mark.parametrize(
    "content",
    [
        "Unstructured memory",
        "【Empty】\n",
        "Preface\n【Topic】\nBody",
        "【Topic】\n[RPENT_MEMORY] bad marker",
    ],
)
def test_invalid_demonstration_rejected(tmp_path, content):
    from rpent.context.memory import load_memory_blocks

    path = tmp_path / "memory.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_memory_blocks(path)


def test_initial_prompt_must_match_selector_library(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    adapter.task = ""
    request["input"][1]["content"][0]["text"] = f"{START}Old experience{END}"
    with pytest.raises(ValueError, match="differs from selector library"):
        selector.process(request, adapter, features)


def test_demo_region_survives_request_rewrite_with_protocol_unchanged(tmp_path):
    from rpent.context.codex import CodexAdapter
    from rpent.context.memory import append_memory, load_memory_blocks
    from rpent.context.proxy import rewrite_body

    path = tmp_path / "demo.txt"
    path.write_text(
        "【Identify】\nIdentify the package.\n\n【Place】\nLower inside basket.\n"
    )
    blocks = load_memory_blocks(path)
    request = {
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "Fixed protocol"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": append_memory("Goal", blocks)}
                ],
            },
            {"type": "reasoning", "encrypted_content": "opaque"},
        ],
        "tools": [{"name": "move"}],
    }
    session = ContextSession(
        CodexAdapter(), Features(), memory_selector=MemorySelector(path)
    )
    raw = json.dumps(request).encode()
    processed, changed = rewrite_body(raw, "", session)
    assert not changed
    assert processed == raw
    request["input"] += [
        {
            "type": "custom_tool_call",
            "call_id": "c1",
            "name": "exec",
            "input": "observe",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "c1",
            "output": [
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {
                            "step": 0,
                            "state": {},
                            "log": {"command": None, "result": None},
                            "task_language": "Pick package",
                        }
                    ),
                }
            ],
        },
    ]
    # No public plan yet: no positive relevance. Only the experience region may change.
    original = copy.deepcopy(request)
    processed, changed = rewrite_body(json.dumps(request).encode(), "", session)
    assert changed
    result = json.loads(processed)
    assert session.stats["memory"]["selected_blocks"] == []
    result["input"][1] = original["input"][1]
    assert result == original
