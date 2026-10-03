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

"""Contracts for causal observations, retries and failed selection rollback."""

import copy
import json
import math

import pytest

from rpent.context.codex import CodexAdapter
from rpent.context.controller import Decision, FixedController, Thresholds
from rpent.context.session import ContextSession
from tests.unit_tests.rpent.context.test_images import FakeFeatures, request


class RecordingController:
    def __init__(self):
        self.observations = []

    def decide(self, observation):
        self.observations.append(observation)
        return Decision(Thresholds(0.2, 0.1, 0.1), policy_version="test")


def test_retry_reuses_decision_and_observation_without_pruning_again():
    controller = RecordingController()
    session = ContextSession(
        CodexAdapter(("front",)), FakeFeatures(), mode="all", controller=controller
    )
    source = request()
    original = copy.deepcopy(source)
    result = session.process(source)
    first = copy.deepcopy(session.stats["decision"])
    assert session.process(copy.deepcopy(source)) == result
    assert len(controller.observations) == 1
    assert session.stats["decision"] == dict(first, reused=True)
    # Caller mutation cannot corrupt the retained retry result.
    result["input"].clear()
    assert session.process(source)["input"]
    assert source == original


def test_observation_is_causal_and_scores_are_explicitly_lagged():
    controller = RecordingController()
    session = ContextSession(
        CodexAdapter(("front",)), FakeFeatures(), mode="all", controller=controller
    )
    source = request()
    first = dict(source, input=source["input"][:7])
    session.process(first)
    obs = controller.observations[0]
    features = dict(zip(obs.names, obs.values))
    assert features["image.records_log"] == math.log1p(3)
    assert features["image.previous_nc_missing"] == 1
    previous = copy.deepcopy(session.stats)
    session.process(source)
    second = dict(
        zip(controller.observations[1].names, controller.observations[1].values)
    )
    assert second["image.records_log"] == math.log1p(6)
    assert second["image.previous_retained_fraction"] == (
        sum(r["selected"] for r in previous["records"]) / 3
    )
    assert all(math.isfinite(x) for x in controller.observations[1].values)
    assert "data:image" not in json.dumps(session.stats["decision"])


def test_failed_later_selector_rolls_back_incremental_state(monkeypatch):
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="all")
    fresh = ContextSession(CodexAdapter(("front",)), FakeFeatures(), mode="all")
    action_class = type(session._action)
    original = action_class.process

    def fail(*args, **kwargs):
        raise RuntimeError("injected action failure")

    monkeypatch.setattr(action_class, "process", fail)
    with pytest.raises(RuntimeError):
        session.process(request())
    assert not session._seen and not session._text._seen
    assert session._decision_count == 0
    monkeypatch.setattr(action_class, "process", original)
    assert session.process(request()) == fresh.process(request())


@pytest.mark.parametrize("bad", [0, -1, 1.1, float("nan"), float("inf"), True])
def test_invalid_thresholds_fail_before_selection(bad):
    with pytest.raises(ValueError):
        FixedController(Thresholds(bad, 0.1, 0.1))
