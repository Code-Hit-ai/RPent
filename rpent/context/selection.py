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

"""Shared score-mass selection for optional history."""


def select_by_nc(scores: dict[str, float], rho: float) -> set[str]:
    """Select descending score mass; insertion order breaks ties by recency."""
    if not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1]")
    total = sum(scores.values())
    if total <= 0 or rho == 1:
        return set(scores)
    rank = {key: i for i, key in enumerate(scores)}
    selected, retained = set(), 0.0
    for key in sorted(scores, key=lambda key: (-scores[key], -rank[key])):
        selected.add(key)
        retained += scores[key]
        if retained >= rho * total:
            break
    return selected
