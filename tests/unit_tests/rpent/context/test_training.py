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

"""Controlled sampled trajectories check PPO accounting and update direction."""

import copy
import dataclasses
import json
import math
import subprocess
import sys

import pytest

from rpent.context.observation import SCORING_VERSION
from rpent.context.trajectory import assemble
from tests.unit_tests.rpent.context.test_policy import checkpoint, observation

torch = pytest.importorskip("torch")
from rpent.context.policy import MLPController  # noqa: E402
from rpent.context.training import (  # noqa: E402
    PPOConfig,
    clipped_policy_loss,
    prepare_batch,
    rewards_and_returns,
    update,
)


def episode(policy, episode_id="e", costs=(100, 200), success=True):
    records = []
    for i, cost in enumerate(costs, 1):
        obs = observation()
        decision = {
            **dataclasses.asdict(policy.decide(obs)),
            "id": i,
            "scoring_version": SCORING_VERSION,
            "observation": dataclasses.asdict(obs),
        }
        records.append(
            {
                "request_id": str(i),
                "response_id": f"{episode_id}-{i}",
                "usage": {
                    "input_tokens": cost,
                    "output_tokens": 10,
                    "input_tokens_details": {"cached_tokens": cost // 2},
                },
                "timing": {"status": 200, "latency_s": 1.0},
                "selection": {"decision": decision},
            }
        )
    result = assemble(
        records,
        {"environment_success": success, "agent_error": None},
        episode_id=episode_id,
    )
    # Match actual JSON logs rather than retaining Python tuples.
    return json.loads(json.dumps(result))


def test_hand_calculated_rewards_and_returns():
    config = PPOConfig(token_scale=1000, beta=0.1)
    reward, returns = rewards_and_returns([100, 200], True, config)
    assert reward == pytest.approx([-0.01, 0.98])
    assert returns == pytest.approx([0.97, 0.98])
    reward, returns = rewards_and_returns([100, 200], False, config)
    assert reward == pytest.approx([-0.01, -0.02])
    assert returns == pytest.approx([-0.03, -0.02])


def test_episode_boundaries_and_retry_costs(tmp_path):
    policy = MLPController(checkpoint(tmp_path), sample=True)
    first = episode(policy, "success")
    retry = copy.deepcopy(first["decisions"][0]["attempts"][0])
    retry["request_id"] = "retry"
    retry["response_id"] = "success-retry"
    first["decisions"][0]["attempts"].append(retry)
    second = episode(policy, "failure", success=False)
    batch = prepare_batch([first, second], policy, PPOConfig(1000))
    assert batch.episode_lengths == (2, 2)
    assert batch.rewards.tolist() == pytest.approx([-0.02, 0.98, -0.01, -0.02])
    assert batch.returns.tolist() == pytest.approx([0.96, 0.98, -0.03, -0.02])
    assert len(batch.actions) == 4  # Retry has a cost, not another policy action.


def test_ppo_clipping_stops_only_the_beneficial_excess():
    logs = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)
    loss = clipped_policy_loss(logs, torch.zeros(2), torch.tensor([1.0, -1.0]), 0.2)
    assert float(loss.detach()) == pytest.approx(-0.2)
    loss.backward()
    assert logs.grad.tolist() == [0.0, 0.0]
    harmful = torch.tensor([math.log(1.5)], requires_grad=True)
    loss = clipped_policy_loss(harmful, torch.zeros(1), torch.tensor([-1.0]), 0.2)
    loss.backward()
    assert (
        harmful.grad.item() > 0
    )  # Gradient descent reduces this bad action's probability.


@pytest.mark.parametrize("success", [True, False])
def test_reward_changes_action_probability_in_correct_direction(tmp_path, success):
    policy = MLPController(checkpoint(tmp_path), sample=True, seed=9)
    config = PPOConfig(1000, beta=0.1, epochs=1, value_weight=0)
    trajectory = episode(policy, costs=(100,), success=success)
    batch = prepare_batch([trajectory], policy, config)
    frozen = {k: v.clone() for k, v in policy.model.state_dict().items()}
    model, metrics = update(policy, batch, config)
    with torch.no_grad():
        distribution, _ = model(batch.observations)
        new_log = distribution.log_prob(batch.actions).sum(-1).item()
    assert (
        (new_log > batch.old_log_prob.item())
        if success
        else (new_log < batch.old_log_prob.item())
    )
    assert metrics["updates"] == 1
    assert all(torch.equal(v, policy.model.state_dict()[k]) for k, v in frozen.items())
    assert all(math.isfinite(v) for v in metrics.values())
    assert torch.equal(model.feature_mean, policy.model.feature_mean)
    assert torch.equal(model.feature_scale, policy.model.feature_scale)


def test_critic_learns_terminal_return(tmp_path):
    policy = MLPController(checkpoint(tmp_path), sample=True)
    config = PPOConfig(1000, epochs=4)
    batch = prepare_batch([episode(policy)], policy, config)
    before = float(batch.returns.square().mean())
    _, metrics = update(policy, batch, config)
    assert metrics["value_mse"] < before


@pytest.mark.parametrize(
    "fault",
    [
        "missing_usage",
        "negative_tokens",
        "wrong_policy",
        "wrong_probability",
        "wrong_value",
        "wrong_rho",
        "wrong_features",
        "fixed_policy",
        "unknown_success",
        "duplicate_response",
        "missing_decision",
        "transport",
        "fallback",
    ],
)
def test_invalid_training_data_is_rejected(tmp_path, fault):
    policy = MLPController(checkpoint(tmp_path), sample=True)
    trajectory = episode(policy)
    decision = trajectory["decisions"][0]["decision"]
    attempt = trajectory["decisions"][0]["attempts"][0]
    if fault == "missing_usage":
        attempt["usage"] = None
    elif fault == "negative_tokens":
        attempt["usage"]["input_tokens"] = -1
    elif fault == "wrong_policy":
        decision["policy_version"] = "old-policy"
    elif fault == "wrong_probability":
        decision["log_prob"] += 1
    elif fault == "wrong_value":
        decision["value"] += 1
    elif fault == "wrong_rho":
        decision["thresholds"]["image"] = 0.9
    elif fault == "wrong_features":
        decision["observation"]["names"].reverse()
    elif fault == "fixed_policy":
        decision["latent_action"] = None
    elif fault == "unknown_success":
        trajectory["outcome"]["environment_success"] = None
    elif fault == "duplicate_response":
        trajectory["decisions"][1]["attempts"][0]["response_id"] = attempt[
            "response_id"
        ]
    elif fault == "missing_decision":
        trajectory["decisions"][1]["decision"]["id"] = 3
    elif fault == "transport":
        attempt["timing"]["transport_error"] = True
    else:
        trajectory["issues"] = ["compression_fallback"]
    with pytest.raises(ValueError):
        prepare_batch([trajectory], policy, PPOConfig(1000))


def test_duplicate_episode_is_not_trained_twice(tmp_path):
    policy = MLPController(checkpoint(tmp_path), sample=True)
    trajectory = episode(policy)
    with pytest.raises(ValueError):
        prepare_batch([trajectory, trajectory], policy, PPOConfig(1000))


def test_cli_updates_checkpoint_without_modifying_parent(tmp_path):
    path = checkpoint(tmp_path)
    original = path.read_bytes()
    policy = MLPController(path, sample=True)
    data = tmp_path / "synthetic.json"
    data.write_text(json.dumps(episode(policy)))
    output = tmp_path / "synthetic-trained.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rpent.context.training",
            "--checkpoint",
            str(path),
            "--trajectories",
            str(data),
            "--output",
            str(output),
            "--token-scale",
            "1000",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(completed.stdout)
    assert metrics["samples"] == 2 and metrics["updates"] > 0
    assert path.read_bytes() == original
    trained = MLPController(output)
    assert trained.version != policy.version
    assert (
        trained.decide(observation()).thresholds
        != policy.decide(observation()).thresholds
    )
    # Old trajectories must not be replayed against the newly updated policy.
    with pytest.raises(ValueError):
        prepare_batch([json.loads(data.read_text())], trained, PPOConfig(1000))


def test_kl_stops_further_epochs_and_stale_batch_is_rejected(tmp_path):
    policy = MLPController(checkpoint(tmp_path), sample=True)
    config = PPOConfig(1000, epochs=10, target_kl=1e-12)
    batch = prepare_batch([episode(policy)], policy, config)
    _, metrics = update(policy, batch, config)
    assert metrics["updates"] == 1
    with pytest.raises(ValueError, match="different collection policy"):
        update(policy, dataclasses.replace(batch, policy_version="stale"), config)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), True])
def test_invalid_reward_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        PPOConfig(scale)
