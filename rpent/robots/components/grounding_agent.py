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

"""Tool-free visual point selection with the Molmo result contract."""

import io
import tempfile
from dataclasses import replace

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from rpent.robots.components.molmo_client import MolmoResult


class PointSelection(BaseModel):
    """One point in original-image pixel coordinates, or no visible target."""

    model_config = ConfigDict(allow_inf_nan=False)
    point_xy: tuple[float, float] | None = Field(
        description="Pixel (column, row) in the original image; null if not visible."
    )


class GroundingAgent:
    """Locate a named object part without planning or executing robot actions."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        if not model.strip() or (model.startswith("codex:") and not model[6:].strip()):
            raise ValueError("grounding agent requires a nonempty model name")
        self.model = model
        self.base_url = base_url
        self._agent = None

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Select one visible material point or report the target as absent."""
        with Image.open(io.BytesIO(image)) as frame:
            size = frame.size
        prompt = (
            f"Locate {query} in this robot camera image. Preserve the named object "
            "part and requested spatial relation. Select a clearly visible material "
            "point away from silhouettes, occlusion and image borders. Do not "
            "substitute another object or part. Return null if it cannot be identified. "
            f"The image has {size[0]} columns and {size[1]} rows. Return point_xy as "
            "[column, row] in ORIGINAL pixel units, not normalized coordinates. "
            "Only select a point; do not use tools or execute robot actions."
        )
        if self.model.startswith("codex:"):
            selected = self._ground_codex(image, prompt)
        else:
            from pydantic_ai import Agent, BinaryContent
            from pydantic_ai.usage import UsageLimits

            from rpent.planner.base import build_api_model

            if self._agent is None:
                self._agent = Agent(
                    build_api_model(self.model, self.base_url),
                    output_type=PointSelection,
                    retries=0,
                    model_settings={"timeout": 90, "max_tokens": 1024},
                )
            selected = self._agent.run_sync(
                [prompt, BinaryContent(data=image, media_type="image/png")],
                usage_limits=UsageLimits(request_limit=1),
            ).output
        if selected.point_xy is not None and not all(
            0 <= value < bound
            for value, bound in zip(selected.point_xy, size, strict=True)
        ):
            raise ValueError("grounding agent pixel outside source image")
        return MolmoResult(
            found=selected.point_xy is not None,
            point_xy=selected.point_xy,
            image_size=size,
            answer=selected.model_dump_json(),
        )

    def _ground_codex(self, image: bytes, prompt: str) -> PointSelection:
        from rpent.planner.codex import build_probe_config, run_probe_turn

        with tempfile.TemporaryDirectory(prefix="rpent-point-agent-") as cwd:
            config = build_probe_config(self.base_url)
            config = replace(
                config,
                cwd=cwd,
                config_overrides=config.config_overrides
                + (
                    "features.shell_tool=false",
                    "features.unified_exec=false",
                    'web_search="disabled"',
                    "apps._default.enabled=false",
                    "mcp_servers={}",
                ),
            )
            answer = run_probe_turn(
                config,
                prompt=prompt,
                model=self.model.split(":", 1)[1],
                timeout_s=90,
                image_bytes=image,
                output_schema={
                    "type": "object",
                    "properties": {
                        "point_xy": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                {"type": "null"},
                            ]
                        }
                    },
                    "required": ["point_xy"],
                    "additionalProperties": False,
                },
            )
        return PointSelection.model_validate_json(answer)
