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

"""Contracts for the opt-in task1 prompt profile."""

import pytest

from robots.libero.prompt_bundle import system_prompt, user_prompt
from rpent.prompt.utils import format_prompt


def variables(**overrides):
    return {
        "suite": "libero_object_swap", "task": 1, "seed": 0,
        "output_dir": "/tmp/run", "recipe_tag": "object_swap_t1_s0",
        "prompt_profile": "compact", **overrides,
    }


def test_compact_without_experience_has_no_old_memory_or_file_read_workflow():
    values = variables()
    rendered = format_prompt(system_prompt(values), variables=values)
    rendered += format_prompt(user_prompt(values), variables=values)
    assert "cream cheese is a blue-white flat box" not in rendered
    assert "[RPENT_MEMORY]" not in rendered
    assert "view_env_state(step=0)" in rendered
    assert "/tmp/run/object_swap_t1_s0.json" in rendered
    assert "MEMORY.md" not in rendered
    assert "strict_hybrid_guide.md" not in rendered
    assert "{{" not in rendered
    assert len(rendered) < 3000


@pytest.mark.parametrize("overrides", [{"task": 2}, {"suite": "libero_spatial"}, {"mode": "explore"}])
def test_compact_rejects_unrelated_tasks(overrides):
    with pytest.raises(ValueError, match="task 1 evaluation only"):
        system_prompt(variables(**overrides))


def test_default_prompt_is_unchanged():
    values = variables(prompt_profile="default")
    assert "PROVEN LEVERS" in str(system_prompt(values))
    assert "Read the guides" in str(user_prompt(values))


def test_compact_long_without_experience_has_no_old_memory():
    values = variables(suite="libero_10", recipe_tag="10_t1_s0")
    rendered = format_prompt(system_prompt(values), variables=values)
    rendered += format_prompt(user_prompt(values), variables=values)
    assert "move one required object" not in rendered
    assert "[RPENT_MEMORY]" not in rendered
    assert "cream cheese is a blue-white flat box" not in rendered
    assert "MEMORY.md" not in rendered
    assert "{{" not in rendered
    assert len(rendered) < 3000


@pytest.mark.parametrize("suite", ["libero_object_swap", "libero_10"])
def test_compact_accepts_same_demonstration_region_and_nc(tmp_path, suite):
    from types import SimpleNamespace

    from rpent.context.memory import MemorySelector, append_memory, load_memory_blocks

    path = tmp_path / "memory.txt"
    path.write_text("【Identify】\nIdentify the blue package.\n\n【Place】\nLower inside the basket.\n")
    blocks = load_memory_blocks(path)
    values = variables(suite=suite)
    prompt = append_memory(format_prompt(user_prompt(values), variables=values), blocks)
    assert prompt.count("[RPENT_MEMORY]") == 1
    assert "Identify the blue package." in prompt
    assert "Lower inside the basket." in prompt
    request = {"input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    adapter = SimpleNamespace(extract_action=lambda _: [])
    result, stats = MemorySelector(path).process(request, adapter, None)
    assert result is request
    assert stats["selected_blocks"] == ["block_001", "block_002"]
    assert "cream cheese is a blue-white flat box" not in prompt
