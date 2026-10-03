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

"""Causal numeric observations; current counts and explicitly lagged scores."""

from dataclasses import dataclass
from math import log1p

FEATURE_VERSION = "counts-and-lagged-scores-v1"
SCORING_VERSION = "fixed-scale-incremental-v1"


@dataclass(frozen=True)
class Observation:
    """Ordered features, persisted alongside names for auditability."""

    names: tuple[str, ...]
    values: tuple[float, ...]
    version: str = FEATURE_VERSION


def observe(
    records: dict,
    states: dict,
    enabled: set[str],
    previous: dict,
    thresholds: object,
    decision_count: int,
    stage_known: bool,
) -> Observation:
    """Read state without scoring or committing a selection.

    Action protection counts precede image selection. Score summaries are from
    the last completed selection, never the pending response. Raw text, images,
    and robot-specific coordinates do not enter this vector.
    """
    values = {"requests_log": log1p(decision_count), "stage_known": float(stage_known)}
    for modality in ("image", "text", "action"):
        rows = records[modality]
        seen, kept = states[modality]
        eligible = [r for r in rows if r.id not in seen or r.id in kept or r.protected]
        candidates = [
            r for r in eligible if not r.protected and not getattr(r, "initial", False)
        ]
        prefix = modality + "."
        values[prefix + "enabled"] = float(modality in enabled)
        values[prefix + "records_log"] = log1p(len(rows))
        values[prefix + "new_log"] = log1p(sum(r.id not in seen for r in rows))
        values[prefix + "candidates_before_image_log"] = log1p(len(candidates))
        values[prefix + "protected_before_image_fraction"] = sum(
            r.protected for r in rows
        ) / max(1, len(rows))
        values[prefix + "previous_rho"] = getattr(thresholds, modality)
        stats = previous if modality == "image" else previous.get(modality, {})
        old_rows = stats.get("records", [])
        values[prefix + "previous_selection_available"] = float(bool(old_rows))
        values[prefix + "previous_retained_fraction"] = sum(
            r["selected"] for r in old_rows
        ) / max(1, len(old_rows))
        nc = stats.get("nc")
        values[prefix + "previous_nc"] = 0.0 if nc is None else nc
        values[prefix + "previous_nc_missing"] = float(nc is None)
        for field in ("relevance", "novelty", "event", "score"):
            measured = [r[field] for r in old_rows if r.get(field) is not None]
            values[prefix + "previous_" + field + "_mean"] = sum(measured) / max(
                1, len(measured)
            )
            values[prefix + "previous_" + field + "_missing"] = 1 - len(measured) / max(
                1, len(old_rows)
            )
    return Observation(tuple(values), tuple(float(x) for x in values.values()))
