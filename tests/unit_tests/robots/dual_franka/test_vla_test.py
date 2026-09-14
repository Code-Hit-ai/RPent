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

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.dual_franka.vla_test import DeploymentTest, run_console


def make_session(tmp_path, *, fail=False, terminate=False):
    calls = []

    def observation():
        calls.append("observe")
        return {
            "states": np.zeros(20, np.float32),
            "main_images": np.zeros((4, 4, 3), np.uint8),
            "extra_view_images": np.zeros((2, 4, 4, 3), np.uint8),
        }

    actions = np.tile([0.5, 0, 0.5, 1, 0, 0, 0, 1, 0, 0] * 2, (20, 1)).astype(
        np.float32
    )

    def predict(obs, options):
        calls.append("predict")
        assert obs["task_descriptions"] == "pick"
        return actions.copy()

    def execute(a):
        calls.append("execute")
        if fail:
            raise RuntimeError("RPC timeout")
        return {"terminated": terminate, "observation": observation()}

    env = SimpleNamespace(
        get_observation=observation,
        chunk_step=execute,
        get_robot_state=lambda: {"state": "after"},
    )
    workspace = {
        "ee_pose_limit_min": [[0, -1, 0]] * 2,
        "ee_pose_limit_max": [[1, 1, 1]] * 2,
    }
    s = DeploymentTest(
        env,
        SimpleNamespace(predict=predict),
        tmp_path,
        workspace,
        "pick",
        {"config": {"openpi": {"action_chunk": 20}}},
    )
    return s, calls, actions


def test_infer_records_without_execution(tmp_path):
    s, calls, _ = make_session(tmp_path)
    result = s.chunk()
    assert calls == ["observe", "predict"]
    assert not result["executed"]
    record = json.loads((tmp_path / "vla_000001.json").read_text())
    assert record["prompt"] == "pick"
    with np.load(tmp_path / "vla_000001.npz", allow_pickle=False) as a:
        assert a["root/actions"].shape == (20, 20)
        assert a["root/input/main_images"].dtype == np.uint8


def test_step_reinfers_instead_of_reusing_preview(tmp_path):
    s, calls, _ = make_session(tmp_path)
    s.chunk()
    s.chunk(True)
    assert calls.count("predict") == 2 and calls.count("execute") == 1
    assert "robot_state_after" in json.loads((tmp_path / "vla_000002.json").read_text())


@pytest.mark.parametrize("kind", ["nan", "shape", "workspace", "rotation"])
def test_bad_actions_never_execute(tmp_path, kind):
    s, calls, a = make_session(tmp_path)
    if kind == "nan":
        a[0, 0] = np.nan
    if kind == "shape":
        a = a[:1]
    if kind == "workspace":
        a[0, 10] = 2
    if kind == "rotation":
        a[0, 3:9] = 0
    s.model.predict = lambda *args, **kw: a
    with pytest.raises(ValueError):
        s.chunk(True)
    assert "execute" not in calls
    assert not json.loads((tmp_path / "vla_000001.json").read_text())["ok"]


def test_failure_aborts_run_and_blocks_further_motion(tmp_path):
    s, calls, _ = make_session(tmp_path, fail=True)
    lines = iter(["run 3", "step", "quit"])
    run_console(s, read=lambda _: next(lines), emit=lambda _: None)
    assert calls.count("execute") == 1
    assert s.execution_uncertain


def test_termination_stops_run(tmp_path):
    s, calls, _ = make_session(tmp_path, terminate=True)
    lines = iter(["run 3", "quit"])
    run_console(s, read=lambda _: next(lines), emit=lambda _: None)
    assert calls.count("execute") == 1


@pytest.mark.parametrize(
    "line", ["run 0", "run 21", "run abc", "step 3", "infer 2", "prompt "]
)
def test_invalid_commands_do_nothing(tmp_path, line):
    s, calls, _ = make_session(tmp_path)
    lines = iter([line, "quit"])
    run_console(s, read=lambda _: next(lines), emit=lambda _: None)
    assert calls == []


def test_task_registered():
    from robots.dual_franka.tasks import get_dual_franka_task

    assert get_dual_franka_task(104).name == "vla_deployment_test"


def test_terminated_episode_requires_reset(tmp_path):
    s, calls, _ = make_session(tmp_path, terminate=True)
    s.chunk(True)
    with pytest.raises(RuntimeError, match="Episode ended"):
        s.chunk(True)
    assert calls.count("execute") == 1


def test_dual_client_uses_live_state_instead_of_cached_reset():
    from robots.dual_franka.env_client import DualFrankaEnvClient

    client = object.__new__(DualFrankaEnvClient)
    client._last_states = np.zeros(20)
    client._client = SimpleNamespace(call=lambda *a, **k: {"states": np.ones(20)})
    np.testing.assert_array_equal(client.get_observation()["states"], np.ones(20))


@pytest.mark.parametrize("instruction", [None, "diagnostic prompt"])
def test_session_accepts_standard_config_without_local_deployment(
    tmp_path, monkeypatch, instruction
):
    import argparse

    import yaml

    from robots.dual_franka import vla_test
    from robots.dual_franka.tasks import CLEAN_DESK_VLA_PROMPT

    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump({"workspace": {}}))
    closed = []
    observed = []
    args = argparse.Namespace(
        robot_config=str(config_path),
        task_id=104,
        instruction=instruction,
        vla_model_path="checkpoint",
        vla_repo_id="dataset",
    )
    model = SimpleNamespace(
        status=lambda **kw: {"config": {"openpi": {"action_chunk": 20}}}
    )
    spec = SimpleNamespace(
        parse_config=lambda args: SimpleNamespace(output_dir=tmp_path / "run"),
        init_runtime=lambda *a: (
            [SimpleNamespace(stop=lambda: closed.append(True))],
            {"model": model, "env": SimpleNamespace(get_camera_meta=lambda: {})},
        ),
    )
    monkeypatch.setattr(vla_test, "get_robot_spec", lambda: spec)
    monkeypatch.setattr(
        vla_test, "run_console", lambda session: observed.append(session.prompt)
    )
    assert vla_test.run_session(args) == 0
    assert observed == [instruction or CLEAN_DESK_VLA_PROMPT]
    assert closed == [True]


def test_vla_status_queries_metadata_without_inference():
    from unittest.mock import Mock

    from rpent.robots.components.pi05_vla_client import Pi05VLAClient

    metadata = {"config": {"openpi": {"action_chunk": 20}}}
    rpc = Mock()
    rpc.call.return_value = metadata
    client = Pi05VLAClient(rpc, embodiment="dual_franka")
    assert client.status(timeout_s=5) == metadata
    rpc.call.assert_called_once_with("vla.status", timeout_s=5)
