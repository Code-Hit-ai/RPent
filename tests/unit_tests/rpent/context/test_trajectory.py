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

"""Feedback accounting preserves missing data and retry boundaries."""

import copy
import gzip
import json

from rpent.context.trajectory import assemble, read_response, write_trajectory


def record(request_id="0001"):
    return {
        "request_id": request_id,
        "response_id": "response-" + request_id,
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "timing": {"status": 200, "latency_s": 2},
        "selection": {
            "decision": {
                "id": 1,
                "thresholds": {"image": 0.2, "text": 0.1, "action": 0.1},
                "reused": False,
            },
            "records": [{"id": "old-image", "selected": False}],
        },
    }


def test_usage_is_read_once_even_if_repeated_in_stream(tmp_path):
    event = {
        "response": {"id": "r", "usage": {"input_tokens": 100, "output_tokens": 10}}
    }
    raw = ("data: " + json.dumps(event) + "\n\n") * 2 + "data: [DONE]\n"
    path = tmp_path / "response"
    path.write_bytes(gzip.compress(raw.encode()))
    usage, _, response_id = read_response(path, "gzip")
    assert usage["input_tokens"] == 100
    assert response_id == "r"
    path.write_text(json.dumps({"object": "response", **event["response"]}))
    assert read_response(path)[0] == usage


def test_attempts_group_by_decision_without_fabricating_usage():
    first = record()
    first["usage"] = None
    first["timing"]["transport_error"] = True
    second = record("0002")
    second["selection"]["decision"]["reused"] = True
    result = assemble([first, second], {"environment_success": True}, episode_id="e")
    assert len(result["decisions"]) == 1
    assert len(result["decisions"][0]["attempts"]) == 2
    assert result["decisions"][0]["attempts"][0]["usage"] is None
    assert "missing_usage" in result["issues"]
    assert "transport_failure" in result["issues"]


def test_agent_claim_cannot_supply_missing_environment_verdict(tmp_path):
    path = write_trajectory(
        tmp_path, [record()], {"task_result": {"finish": {"status": "success"}}}
    )
    result = json.loads(path.read_text())
    assert not result["audit_complete"]
    assert result["outcome"]["environment_success"] is None
    assert "missing_environment_verdict" in result["issues"]


def test_duplicate_response_id_and_fallback_are_explicit():
    first = record()
    second = copy.deepcopy(first)
    second["request_id"] = "0002"
    fallback = {"request_id": "0003", "selection": {"fallback": "processing_error"}}
    result = assemble(
        [first, second, fallback], {"environment_success": True}, episode_id="e"
    )
    assert "duplicate_response_id" in result["issues"]
    assert "compression_fallback" in result["issues"]


def test_environment_failure_with_complete_usage_is_valid_audit():
    result = assemble([record()], {"environment_success": False}, episode_id="e")
    assert result["audit_complete"]
    assert result["outcome"]["environment_success"] is False
    assert result["decisions"][0]["attempts"][0]["selection"]["image"][
        "omitted_ids"
    ] == ["old-image"]
