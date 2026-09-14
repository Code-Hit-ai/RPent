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

from types import SimpleNamespace

import pytest

from robots.dual_franka.manual import parse_command, run_console
from robots.dual_franka.tools import TOOLS_SPEC

SPECS = {s["name"]: s for s in TOOLS_SPEC}


@pytest.mark.parametrize(
    "line",
    [
        'move_delta {"arm":"both","delta_xyz":[0,0,1]}',
        'move_delta {"arm":"right","delta_xyz":[0,NaN,1]}',
        'move_delta {"arm":"right","delta_xyz":[0,1]}',
        'close_gripper {"arm":"right","extra":1}',
        'vla_right_grasp {"prompt":"test","max_chunks":0}',
    ],
)
def test_invalid_commands_do_not_dispatch(line):
    with pytest.raises(ValueError):
        parse_command(line, SPECS)


@pytest.mark.parametrize("vla", [False, True])
def test_console_uses_shared_dispatcher_only_for_explicit_calls(vla):
    calls = []

    def execute(name, args):
        calls.append((name, args))
        return SimpleNamespace(content_blocks=[{"type": "text", "text": "ok"}])

    toolkit = SimpleNamespace(get_tools_spec=lambda: TOOLS_SPEC, execute_tool=execute)
    lines = iter(
        [
            "help",
            'move_delta {"arm":"right","delta_xyz":[0,0,0.01]}',
            'vla_right_grasp {"prompt":"test","max_chunks":1}',
            "quit",
        ]
    )
    run_console(
        toolkit, enable_vla=vla, read=lambda _: next(lines), emit=lambda _: None
    )
    assert calls[0] == ("move_delta", {"arm": "right", "delta_xyz": [0, 0, 0.01]})
    assert len(calls) == (2 if vla else 1)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("state", ("view_env_state", {})),
        ("cameras", ("view_camera_meta", {})),
        (
            "move right 0 0 0.01",
            ("move_delta", {"arm": "right", "delta_xyz": [0.0, 0.0, 0.01]}),
        ),
        (
            "rotate left 0 0 .05",
            ("rotate_delta", {"arm": "left", "delta_rpy": [0.0, 0.0, 0.05]}),
        ),
        ("close left", ("close_gripper", {"arm": "left"})),
        ("open right", ("open_gripper", {"arm": "right"})),
    ],
)
def test_short_commands(line, expected):
    assert parse_command(line, SPECS) == expected


@pytest.mark.parametrize(
    "line",
    [
        "move right .10 .10 0",
        "move right .101 0 0",
        'move_delta {"arm":"right","delta_xyz":[0,0,0.101]}',
        "rotate left .8 0 0",
        "move right nan 0 0",
        "move right 0 0",
        "open both",
        "state unexpected",
        'rotate_delta {"arm":"left","delta_rpy":[0,0,0.8]}',
    ],
)
def test_limits_apply_to_aliases_and_json(line):
    with pytest.raises(ValueError):
        parse_command(line, SPECS)


def test_failure_returns_to_prompt_without_retry():
    calls = []
    messages = []

    def execute(name, args):
        calls.append((name, args))
        return SimpleNamespace(
            result={"ok": False, "final_error_rad": 0.042}, content_blocks=[]
        )

    toolkit = SimpleNamespace(get_tools_spec=lambda: TOOLS_SPEC, execute_tool=execute)
    lines = iter(["rotate left 0 0 .05", "quit"])
    run_console(toolkit, read=lambda _: next(lines), emit=messages.append)
    assert len(calls) == 1
    assert any("0.042" in msg for msg in messages)


def test_manual_task_does_not_need_vla():
    from robots.dual_franka.tasks import get_dual_franka_task

    assert get_dual_franka_task(103).name == "manual_primitive_test"


@pytest.mark.parametrize(
    "line",
    [
        "move right .10 0 0",
        "move left .06 .08 0",
        'move_delta {"arm":"right","delta_xyz":[0,0,-0.10]}',
    ],
)
def test_ten_centimeter_moves_are_accepted(line):
    name, args = parse_command(line, SPECS)
    assert name == "move_delta"
    assert len(args["delta_xyz"]) == 3


@pytest.mark.parametrize(
    "line",
    [
        "rotate left 0 0 0.785398",
        'rotate_delta {"arm":"right","delta_rpy":[0,0,-0.7853981633974483]}',
    ],
)
def test_forty_five_degree_rotation_accepted(line):
    assert parse_command(line, SPECS)[0] == "rotate_delta"


def test_combined_rotation_limit():
    with pytest.raises(ValueError):
        parse_command("rotate left .6 .6 0", SPECS)
