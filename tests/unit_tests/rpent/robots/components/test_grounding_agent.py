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

"""Point-only agent adapters run offline with fake model output."""

import io
from dataclasses import dataclass

import pytest
from PIL import Image
from pydantic_ai.models.test import TestModel

from rpent.robots.components.grounding_agent import GroundingAgent


@pytest.fixture
def image():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("point", [[5, 3], None])
def test_api_point_selection(monkeypatch, image, point):
    import rpent.planner.base as base

    model = TestModel(custom_output_args={"point_xy": point})
    monkeypatch.setattr(base, "build_api_model", lambda *a: model)
    result = GroundingAgent("test:model").ground(image, "cup rim")
    assert result.found == (point is not None)
    assert result.image_size == (20, 10)
    assert model.last_model_request_parameters.function_tools == []


@pytest.mark.parametrize("point", [[25, 3], [-1, 2]])
def test_outside_image_is_rejected(monkeypatch, image, point):
    import rpent.planner.base as base

    monkeypatch.setattr(
        base,
        "build_api_model",
        lambda *a: TestModel(custom_output_args={"point_xy": point}),
    )
    with pytest.raises(ValueError, match="outside"):
        GroundingAgent("test:model").ground(image, "cup rim")


def test_codex_receives_image_and_no_robot_tools(monkeypatch, image):
    import rpent.planner.codex as codex

    @dataclass
    class Config:
        cwd: str = "/unused"
        config_overrides: tuple = ()

    monkeypatch.setattr(codex, "build_probe_config", lambda *a: Config())

    def probe(config, **kwargs):
        assert config.cwd != "/unused"
        assert "mcp_servers={}" in config.config_overrides
        assert "features.shell_tool=false" in config.config_overrides
        assert kwargs["image_bytes"] == image
        assert kwargs["output_schema"]["required"] == ["point_xy"]
        return '{"point_xy": [5, 3]}'

    monkeypatch.setattr(codex, "run_probe_turn", probe)
    assert GroundingAgent("codex:test").ground(image, "cup rim").point_xy == (5, 3)
