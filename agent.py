"""StoVol-Lite cognitive bandit agent for the MindRL Challenge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Any


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an object."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def clip(value: float, lower: float, upper: float) -> float:
    """Restrict a numeric value to a closed interval."""
    return max(lower, min(value, upper))


class ActionBelief:
    """Trajectory-local beliefs associated with one action."""

    mean: float
    variance: float
    process_variance: float
    observation_variance: float
    error_memory: float = 0.0
    choice_trace: float = 0.0


class Agent:
    """Adaptive stochasticity-volatility cognitive bandit model."""

    def __init__(
        self,
        config: Mapping[str, Any] | Any | None = None,
    ) -> None:
        model = get_field(config or {}, "model", {}) or {}

        self.initial_value = float(
            get_field(model, "initial_value", 50.0)
        )
        self.initial_variance = float(
            get_field(model, "initial_variance", 400.0)
        )
        self.initial_process_variance = float(
            get_field(model, "initial_process_variance", 25.0)
        )
        self.initial_observation_variance = float(
            get_field(model, "initial_observation_variance", 100.0)
        )

        self.process_floor = float(
            get_field(model, "process_floor", 1.0)
        )
        self.process_ceiling = float(
            get_field(model, "process_ceiling", 400.0)
        )
        self.observation_floor = float(
            get_field(model, "observation_floor", 4.0)
        )
        self.observation_ceiling = float(
            get_field(model, "observation_ceiling", 2500.0)
        )
        self.variance_ceiling = float(
            get_field(model, "variance_ceiling", 10000.0)
        )

        self.error_memory_rate = float(
            get_field(model, "error_memory_rate", 0.20)
        )
        self.noise_adaptation_rate = float(
            get_field(model, "noise_adaptation_rate", 0.05)
        )
        self.choice_trace_decay = float(
            get_field(model, "choice_trace_decay", 0.80)
        )

        self.value_sensitivity = float(
            get_field(model, "value_sensitivity", 2.0)
        )
        self.uncertainty_sensitivity = float(
            get_field(model, "uncertainty_sensitivity", 0.15)
        )
        self.choice_sensitivity = float(
            get_field(model, "choice_sensitivity", 0.50)
        )
        self.reward_stay_sensitivity = float(
            get_field(model, "reward_stay_sensitivity", 0.75)
        )
        self.lapse = clip(
            float(get_field(model, "lapse", 0.03)),
            0.0,
            1.0,
        )

        self.actions: list[Any] = []
        self.beliefs: dict[Any, ActionBelief] = {}
        self.last_action: Any = None
        self.last_surprise = 0.0

    def reset(self, context: Mapping[str, Any] | Any) -> None:
        """Clear all trajectory-local state and initialize action beliefs."""
        available_actions = get_field(
            context,
            "available_actions",
            None,
        )
        if available_actions is None:
            raise ValueError("context.available_actions is required")

        self.actions = list(available_actions)
        if not self.actions:
            raise ValueError("available_actions cannot be empty")

        self.beliefs = {
            action: ActionBelief(
                mean=self.initial_value,
                variance=self.initial_variance,
                process_variance=self.initial_process_variance,
                observation_variance=self.initial_observation_variance,
            )
            for action in self.actions
        }

        self.last_action = None
        self.last_surprise = 0.0

    def predict(self, history: Any) -> dict[str, dict[Any, float]]:
        """Return probabilities for the next action."""
        del history  # State is updated through update() after each trial.

        if not self.actions:
            raise RuntimeError("reset(context) must be called before predict()")

        n_actions = len(self.actions)
        value_mean = sum(
            self.beliefs[action].mean for action in self.actions
        ) / n_actions

        uncertainty_mean = sum(
            sqrt(max(self.beliefs[action].variance, 0.0))
            for action in self.actions
        ) / n_actions

        scale = sqrt(
            sum(
                (self.beliefs[action].mean - value_mean) ** 2
                + self.beliefs[action].variance
                for action in self.actions
            )
            / n_actions
        )
        scale = max(scale, 1e-8)

        logits: dict[Any, float] = {}

        for action in self.actions:
            belief = self.beliefs[action]

            value_feature = (belief.mean - value_mean) / scale
            uncertainty_feature = (
                sqrt(max(belief.variance, 0.0)) - uncertainty_mean
            ) / scale

            reward_stay = 0.0
            if action == self.last_action:
                reward_stay = self.last_surprise

            logits[action] = (
                self.value_sensitivity * value_feature
                + self.uncertainty_sensitivity * uncertainty_feature
                + self.choice_sensitivity * belief.choice_trace
                + self.reward_stay_sensitivity * reward_stay
            )

        max_logit = max(logits.values())
        weights = {
            action: exp(logit - max_logit)
            for action, logit in logits.items()
        }
        weight_sum = sum(weights.values())

        probabilities = {
            action: (
                (1.0 - self.lapse) * weights[action] / weight_sum
                + self.lapse / n_actions
            )
            for action in self.actions
        }

        return {"action_probs": probabilities}

    def update(
        self,
        action: Any,
        reward: Any,
        info: Any | None = None,
    ) -> None:
        """Update trajectory-local beliefs after an observed choice."""
        del info

        if action not in self.beliefs:
            return

        for belief in self.beliefs.values():
            belief.variance = min(
                belief.variance + belief.process_variance,
                self.variance_ceiling,
            )
            belief.choice_trace *= self.choice_trace_decay

        selected = self.beliefs[action]
        selected.choice_trace += 1.0
        self.last_action = action

        try:
            observed_reward = float(reward)
        except (TypeError, ValueError):
            self.last_surprise = 0.0
            return

        if not isfinite(observed_reward):
            self.last_surprise = 0.0
            return

        innovation = observed_reward - selected.mean
        predictive_variance = (
            selected.variance + selected.observation_variance
        )
        predictive_variance = max(predictive_variance, 1e-8)

        kalman_gain = selected.variance / predictive_variance
        selected.mean += kalman_gain * innovation
        selected.variance *= 1.0 - kalman_gain

        self.last_surprise = innovation / sqrt(predictive_variance)

        memory_rate = self.error_memory_rate
        selected.error_memory = (
            (1.0 - memory_rate) * selected.error_memory
            + memory_rate * innovation
        )

        persistent_error = selected.error_memory**2
        transient_error = (
            innovation - selected.error_memory
        ) ** 2

        process_target = clip(
            persistent_error,
            self.process_floor,
            self.process_ceiling,
        )
        observation_target = clip(
            transient_error,
            self.observation_floor,
            self.observation_ceiling,
        )

        adaptation_rate = self.noise_adaptation_rate

        selected.process_variance = (
            (1.0 - adaptation_rate) * selected.process_variance
            + adaptation_rate * process_target
        )
        selected.observation_variance = (
            (1.0 - adaptation_rate)
            * selected.observation_variance
            + adaptation_rate * observation_target
        )