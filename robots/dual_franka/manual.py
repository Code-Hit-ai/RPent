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

"""Manual terminal frontend over the same toolkit used by agent planners."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from typing import Any

from robots.dual_franka.robot_spec import get_robot_spec, get_toolkit
from robots.dual_franka.tools import TOOLS_SPEC
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("dual_franka.manual")


def parse_command(line: str, specs: dict) -> tuple[str, dict]:
    parts = line.strip().split()
    aliases = {
        "state": "view_env_state",
        "cameras": "view_camera_meta",
        "move": "move_delta",
        "rotate": "rotate_delta",
        "open": "open_gripper",
        "close": "close_gripper",
    }
    if parts and parts[0] in aliases:
        name = aliases[parts[0]]
        if parts[0] in ("move", "rotate"):
            if len(parts) != 5:
                raise ValueError("Usage: move/rotate left|right x y z")
            key = "delta_xyz" if parts[0] == "move" else "delta_rpy"
            values = {"arm": parts[1], key: [float(x) for x in parts[2:]]}
        elif parts[0] in ("open", "close"):
            if len(parts) != 2:
                raise ValueError("Usage: open/close left|right")
            values = {"arm": parts[1]}
        else:
            if len(parts) != 1:
                raise ValueError("No arguments expected")
            values = {}
        raw = json.dumps(values)
    else:
        name, _, raw = line.strip().partition(" ")
    if name not in specs:
        raise ValueError(f"Unknown/unavailable tool: {name}. Type help.")
    args = json.loads(raw.strip() or "{}")
    if not isinstance(args, dict):
        raise ValueError("Arguments must be a JSON object")
    schema = specs[name]["input_schema"]
    props = schema.get("properties", {})
    if set(args) - set(props):
        raise ValueError("Unknown arguments: " + str(set(args) - set(props)))
    for key in schema.get("required", []):
        if key not in args:
            raise ValueError(f"Missing argument: {key}")

    def validate(value, rule):
        kind = rule.get("type")
        if kind == "string" and not isinstance(value, str):
            raise ValueError("Expected string")
        if kind == "integer" and (type(value) is not int):
            raise ValueError("Expected integer")
        if kind == "number" and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            raise ValueError("Expected finite number")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError("Invalid choice")
        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError("Below minimum")
        if "maximum" in rule and value > rule["maximum"]:
            raise ValueError("Above maximum")
        if kind == "array":
            if not isinstance(value, list):
                raise ValueError("Expected array")
            if (
                not rule.get("minItems", 0)
                <= len(value)
                <= rule.get("maxItems", math.inf)
            ):
                raise ValueError("Invalid array length")
            for item in value:
                validate(item, rule["items"])

    for key, value in args.items():
        validate(value, props[key])
    for key, limit in [("delta_xyz", 0.10), ("delta_rpy", math.pi / 4)]:
        if key in args and math.sqrt(sum(x * x for x in args[key])) > limit + 1e-12:
            raise ValueError(f"{key} vector norm must be <= {limit}; command rejected")
    return name, args


def run_console(
    toolkit: Any,
    *,
    enable_vla: bool = False,
    read: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> None:
    """Execute explicit operator commands through the toolkit."""
    robot_tools = {s["name"] for s in TOOLS_SPEC} | {"reset", "vla_grasp"}
    specs = {
        s["name"]: s
        for s in toolkit.get_tools_spec()
        if s["name"] in robot_tools and (enable_vla or not s["name"].startswith("vla_"))
    }
    emit(
        "手动模式：state | cameras | move/rotate left|right x y z | open/close left|right | reset | quit"
    )
    emit(
        "平移单位 m，向量长度≤0.10；旋转单位 rad，向量长度≤π/4（45°）。也支持工具名 + JSON。"
    )
    emit(
        "No automatic task sequence. view_env_state reads the latest recorded snapshot."
    )
    while True:
        try:
            line = read("manual> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("quit", "exit"):
            break
        if not line:
            continue
        if line == "help":
            for name, spec in specs.items():
                emit(name + " " + json.dumps(spec["input_schema"], ensure_ascii=False))
            continue
        try:
            name, args = parse_command(line, specs)
        except (ValueError, TypeError) as exc:
            emit(f"Invalid command: {exc}")
            continue
        # Exactly the agent dispatcher, including its operation lock and state logging.
        started = time.monotonic()
        result = toolkit.execute_tool(name, args)
        elapsed = time.monotonic() - started
        payload = getattr(result, "result", {})
        if isinstance(payload, dict):
            fields = (
                "ok",
                "error",
                "arm",
                "requested_delta_xyz",
                "requested_delta_rpy",
                "start_tcp_pose",
                "final_tcp_pose",
                "final_error_m",
                "final_error_rad",
                "steps_used",
                "target_gripper_open",
            )
            summary = {k: payload[k] for k in fields if k in payload}
            if name == "move_delta":
                summary["tolerance_m"] = 0.005
                if "start_tcp_pose" in payload and "final_tcp_pose" in payload:
                    summary["actual_delta_xyz_m"] = [
                        b - a
                        for a, b in zip(
                            payload["start_tcp_pose"][:3], payload["final_tcp_pose"][:3]
                        )
                    ]

            if name == "rotate_delta":
                summary["tolerance_rad"] = 0.04
            if summary:
                emit(json.dumps(summary, ensure_ascii=False, default=str))
                emit(f"耗时 {elapsed:.2f}s；等待下一条命令，不自动重试。")
                continue
        for block in result.content_blocks:
            if block.get("type") == "text":
                emit(block["text"])


def main() -> int:
    """Run the diagnostic command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    spec = get_robot_spec()
    spec.add_cli_args(parser, use_dashboard=False)
    parser.add_argument("--output-dir", default=None)
    parser.set_defaults(task_id=103)
    args = parser.parse_args()
    return run_session(args)


def run_session(args: argparse.Namespace) -> int:
    """Own the diagnostic runtime and close it when the console exits."""
    spec = get_robot_spec()
    if args.task_id == 103 and args.vla_endpoint:
        raise ValueError("task103 is primitive-only; remove --vla-endpoint")
    config = spec.parse_config(args)
    output = init_output_dir(config.output_dir)
    events = NullDashboardEventSink()
    daemons, toolkit = [], None
    logger.info(
        "Initializing environment: robot reset may occur. No planner will run.",
    )
    try:
        daemons, kwargs = spec.init_runtime(args, output, events, None)
        has_vla = kwargs.get("model") is not None
        toolkit = get_toolkit(
            primitives_kwargs=kwargs, dashboard_events=events, config=config
        )
        toolkit.add_tool(
            "reset",
            {
                "name": "reset",
                "description": "Explicit robot reset",
                "input_schema": {"type": "object", "properties": {}},
            },
            lambda: toolkit._primitives.env.reset(),
        )
        logger.info("Records: %s", output)
        run_console(toolkit, enable_vla=has_vla)
    finally:
        if toolkit is not None:
            toolkit.close()
        for daemon in reversed(daemons):
            daemon.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
