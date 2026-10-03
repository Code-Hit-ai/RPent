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

"""Small CPU threshold policy. Imported only when explicitly enabled."""

import argparse
import hashlib
import io
import math
import os
from functools import cache
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from .controller import Decision, Thresholds
from .observation import FEATURE_VERSION, SCORING_VERSION, Observation, observe


@cache
def feature_names() -> tuple[str, ...]:
    """Obtain the schema from the existing observation builder."""
    modalities = ("image", "text", "action")
    return observe(
        {m: [] for m in modalities},
        {m: (set(), set()) for m in modalities},
        set(),
        {},
        Thresholds(0.2, 0.1, 0.1),
        0,
        False,
    ).names


class ActorCritic(nn.Module):
    """Two shared hidden layers, three Gaussian latent actions and one value."""

    def __init__(self) -> None:
        super().__init__()
        size = len(feature_names())
        self.register_buffer("feature_mean", torch.zeros(size))
        self.register_buffer("feature_scale", torch.ones(size))
        self.body = nn.Sequential(
            nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((3,), math.log(0.2)))

    def forward(self, features: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return a latent Normal and value for a vector or batch."""
        hidden = self.body((features - self.feature_mean) / self.feature_scale)
        distribution = Normal(self.actor(hidden), self.log_std.clamp(-5, 0).exp())
        return distribution, self.critic(hidden).squeeze(-1)


def initialize(thresholds: Thresholds, seed: int = 0) -> ActorCritic:
    """Start at constant configured thresholds, without changing the global RNG."""
    if any(v == 1 for v in (thresholds.image, thresholds.text, thresholds.action)):
        raise ValueError("MLP initialization requires thresholds strictly below 1")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ActorCritic()
        with torch.no_grad():
            nn.init.zeros_(model.actor.weight)
            model.actor.bias.copy_(
                torch.tensor(
                    [
                        math.log(v / (1 - v))
                        for v in (thresholds.image, thresholds.text, thresholds.action)
                    ]
                )
            )
            nn.init.zeros_(model.critic.weight)
            nn.init.zeros_(model.critic.bias)
    return model


def save_checkpoint(model: ActorCritic, path: Path) -> None:
    """Atomically store weights, schema and normalization buffers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        torch.save(
            {
                "format_version": 1,
                "feature_version": FEATURE_VERSION,
                "scoring_version": SCORING_VERSION,
                "feature_names": list(feature_names()),
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class MLPController:
    """Frozen CPU policy; evaluation is deterministic unless sampling is enabled.

    log_prob is the density of the stored latent action z, NOT the density of
    sigmoid(z). PPO must evaluate that same latent action. All three latent
    components are included; disabled modalities are simply ignored by selectors.
    """

    def __init__(self, path: Path, *, sample: bool = False, seed: int = 0) -> None:
        raw = Path(path).read_bytes()
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or (
            checkpoint.get("format_version") != 1
            or checkpoint.get("feature_version") != FEATURE_VERSION
            or checkpoint.get("scoring_version") != SCORING_VERSION
            or checkpoint.get("feature_names") != list(feature_names())
        ):
            raise ValueError("controller checkpoint schema/version mismatch")
        with torch.random.fork_rng(devices=[]):
            self.model = ActorCritic()
        try:
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("invalid controller weights") from error
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError("controller weights must be finite")
        if not (self.model.feature_scale > 0).all():
            raise ValueError("controller feature scales must be positive")
        self.model.eval().requires_grad_(False)
        self.sample = sample
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.version = "mlp-" + hashlib.sha256(raw).hexdigest()

    def decide(self, observation: Observation) -> Decision:
        """Predict thresholds and optionally sample a reproducible latent action."""
        if (
            observation.version != FEATURE_VERSION
            or observation.names != feature_names()
        ):
            raise ValueError("controller observation schema mismatch")
        if len(observation.values) != len(observation.names):
            raise ValueError("controller observation length mismatch")
        values = torch.tensor(observation.values, dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("controller observation must be finite")
        with torch.inference_mode():
            distribution, value = self.model(values)
            latent = distribution.mean
            if self.sample:
                latent = latent + distribution.stddev * torch.randn(
                    3, generator=self.generator
                )
            if not torch.isfinite(latent).all() or not torch.isfinite(value):
                raise ValueError("non-finite controller output")
            # Saturated logits remain valid thresholds. Probability accounting
            # uses the original z, including in this numerical edge case.
            rho = latent.double().sigmoid().clamp(1e-12, 1 - 1e-12).tolist()
            return Decision(
                Thresholds(*rho),
                policy_version=self.version,
                latent_action=tuple(latent.tolist()) if self.sample else None,
                log_prob=float(distribution.log_prob(latent).sum())
                if self.sample
                else None,
                value=float(value),
            )


def main() -> None:
    """Create an untrained checkpoint for integration checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rho-image", type=float, default=0.2)
    parser.add_argument("--rho-text", type=float, default=0.1)
    parser.add_argument("--rho-action", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new checkpoint path")
    try:
        model = initialize(
            Thresholds(args.rho_image, args.rho_text, args.rho_action), args.seed
        )
        save_checkpoint(model, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
