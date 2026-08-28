"""Stay-Switch Kalman cognitive agent for the MindRL Challenge."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import exp, isfinite, log, log1p, sqrt
from pathlib import Path
from typing import Any


_EPSILON = 1e-12


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an arbitrary object."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def clip(value: float, lower: float, upper: float) -> float:
    """Restrict a numeric value to a closed interval."""
    return max(lower, min(float(value), upper))


def normalize_probabilities(
    probabilities: Mapping[Any, float],
) -> dict[Any, float]:
    """Return finite, non-negative probabilities that sum to one."""
    cleaned = {
        action: (
            float(probability)
            if isfinite(float(probability)) and float(probability) >= 0.0
            else 0.0
        )
        for action, probability in probabilities.items()
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        uniform = 1.0 / len(cleaned) if cleaned else 0.0
        return {action: uniform for action in cleaned}
    return {
        action: probability / total
        for action, probability in cleaned.items()
    }


def stable_sigmoid(value: float) -> float:
    """Evaluate the logistic function without numerical overflow."""
    if value >= 0.0:
        denominator = 1.0 + exp(-value)
        return 1.0 / denominator
    numerator = exp(value)
    return numerator / (1.0 + numerator)


def stable_softmax(logits: Mapping[Any, float]) -> dict[Any, float]:
    """Convert arbitrary finite logits into a normalized distribution."""
    if not logits:
        return {}
    maximum = max(logits.values())
    weights = {
        action: exp(clip(logit - maximum, -700.0, 0.0))
        for action, logit in logits.items()
    }
    return normalize_probabilities(weights)


class ActionBelief:
    """Trajectory-local Gaussian belief for one action's latent value."""

    __slots__ = ("mean", "variance")

    def __init__(self, mean: float, variance: float) -> None:
        self.mean = float(mean)
        self.variance = float(variance)


class KalmanAgent:
    """Fixed-noise Kalman model with a factored dynamic repeat policy."""

    _FITTED_PARAMETER_NAMES = {
        "initial_value",
        "initial_variance",
        "process_variance",
        "observation_variance",
        "variance_ceiling",
        "value_sensitivity",
        "uncertainty_sensitivity",
        "stay_intercept",
        "stay_value_weight",
        "surprise_linear_weight",
        "surprise_quadratic_weight",
        "run_length_weight",
        "trial_index_weight",
        "lapse",
    }

    def __init__(
        self,
        config: Mapping[str, Any] | Any | None = None,
    ) -> None:
        model = get_field(config or {}, "model", {}) or {}
        fitted = self._load_fitted_parameters(
            get_field(model, "fitted_params_path", None)
        )

        def parameter(name: str, default: float) -> float:
            if name in fitted:
                return float(fitted[name])
            return float(get_field(model, name, default))

        self.initial_value = parameter("initial_value", 50.0)
        self.initial_variance = max(
            parameter("initial_variance", 400.0),
            _EPSILON,
        )
        self.process_variance = max(
            parameter("process_variance", 25.0),
            0.0,
        )
        self.observation_variance = max(
            parameter("observation_variance", 100.0),
            _EPSILON,
        )
        self.variance_ceiling = max(
            parameter("variance_ceiling", 10000.0),
            self.initial_variance,
        )

        self.value_sensitivity = parameter("value_sensitivity", 2.0)
        self.uncertainty_sensitivity = parameter(
            "uncertainty_sensitivity",
            0.15,
        )

        self.stay_intercept = parameter("stay_intercept", 0.0)
        self.stay_value_weight = parameter("stay_value_weight", 1.0)
        self.surprise_linear_weight = parameter(
            "surprise_linear_weight",
            0.0,
        )
        self.surprise_quadratic_weight = parameter(
            "surprise_quadratic_weight",
            0.0,
        )
        self.run_length_weight = parameter("run_length_weight", 0.0)
        self.trial_index_weight = parameter("trial_index_weight", 0.0)
        self.lapse = clip(parameter("lapse", 0.02), 0.0, 1.0)

        self.actions: list[Any] = []
        self.beliefs: dict[Any, ActionBelief] = {}
        self.last_action: Any = None
        self.last_surprise = 0.0
        self.run_length = 0
        self.trial_count = 0

    @classmethod
    def _load_fitted_parameters(
        cls,
        configured_path: Any,
    ) -> dict[str, float]:
        """Load frozen global parameters without fitting during evaluation."""
        if configured_path in (None, ""):
            return {}

        path = Path(str(configured_path))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Fitted parameter file not found: {path}")

        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        raw_parameters = payload.get("parameters", payload)
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("Fitted parameter file must contain a mapping")

        fitted: dict[str, float] = {}
        for name in cls._FITTED_PARAMETER_NAMES:
            if name not in raw_parameters:
                continue
            value = float(raw_parameters[name])
            if not isfinite(value):
                raise ValueError(f"Fitted parameter {name!r} must be finite")
            fitted[name] = value
        return fitted

    def reset(self, context: Mapping[str, Any] | Any) -> None:
        """Clear trajectory-local state and initialize action beliefs."""
        available_actions = get_field(context, "available_actions", None)
        if available_actions is None:
            raise ValueError("context.available_actions is required")

        self.actions = list(available_actions)
        if not self.actions:
            raise ValueError("available_actions cannot be empty")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("available_actions must not contain duplicates")

        self.beliefs = {
            action: ActionBelief(
                mean=self.initial_value,
                variance=self.initial_variance,
            )
            for action in self.actions
        }
        self.last_action = None
        self.last_surprise = 0.0
        self.run_length = 0
        self.trial_count = 0

    def _standardized_features(
        self,
    ) -> tuple[dict[Any, float], dict[Any, float]]:
        """Return scale-stable value and relative-uncertainty features."""
        n_actions = len(self.actions)
        mean_value = sum(
            self.beliefs[action].mean for action in self.actions
        ) / n_actions
        mean_uncertainty = sum(
            sqrt(max(self.beliefs[action].variance, 0.0))
            for action in self.actions
        ) / n_actions

        scale = sqrt(
            sum(
                (self.beliefs[action].mean - mean_value) ** 2
                + self.beliefs[action].variance
                for action in self.actions
            )
            / n_actions
        )
        scale = max(scale, _EPSILON)

        value_features = {
            action: (self.beliefs[action].mean - mean_value) / scale
            for action in self.actions
        }
        uncertainty_features = {
            action: (
                sqrt(max(self.beliefs[action].variance, 0.0))
                - mean_uncertainty
            )
            / scale
            for action in self.actions
        }
        return value_features, uncertainty_features

    def _action_utilities(self) -> dict[Any, float]:
        """Compute value-plus-uncertainty utilities for all actions."""
        value_features, uncertainty_features = self._standardized_features()
        return {
            action: (
                self.value_sensitivity * value_features[action]
                + self.uncertainty_sensitivity
                * uncertainty_features[action]
            )
            for action in self.actions
        }

    def _stay_probability(
        self,
        base_probabilities: Mapping[Any, float],
    ) -> float:
        """Predict whether the participant repeats the previous action."""
        probability = clip(
            base_probabilities[self.last_action],
            1e-9,
            1.0 - 1e-9,
        )
        value_log_odds = log(probability / (1.0 - probability))
        surprise = self.last_surprise

        logit = (
            self.stay_intercept
            + self.stay_value_weight * value_log_odds
            + self.surprise_linear_weight * surprise
            + self.surprise_quadratic_weight * surprise**2
            + self.run_length_weight * log1p(self.run_length)
            + self.trial_index_weight * log1p(self.trial_count)
        )
        return stable_sigmoid(logit)

    def predict(self, history: Any) -> dict[str, dict[Any, float]]:
        """Return next-action probabilities using past information only."""
        del history

        if not self.actions:
            raise RuntimeError("reset(context) must be called before predict()")

        n_actions = len(self.actions)
        if n_actions == 1:
            return {"action_probs": {self.actions[0]: 1.0}}

        utilities = self._action_utilities()
        base_probabilities = stable_softmax(utilities)

        if self.last_action not in self.beliefs:
            probabilities = base_probabilities
        else:
            stay_probability = self._stay_probability(base_probabilities)
            switch_utilities = {
                action: utility
                for action, utility in utilities.items()
                if action != self.last_action
            }
            conditional_switch = stable_softmax(switch_utilities)
            probabilities = {
                action: (
                    stay_probability
                    if action == self.last_action
                    else (1.0 - stay_probability)
                    * conditional_switch[action]
                )
                for action in self.actions
            }

        probabilities = {
            action: (
                (1.0 - self.lapse) * probabilities[action]
                + self.lapse / n_actions
            )
            for action in self.actions
        }
        return {
            "action_probs": normalize_probabilities(probabilities)
        }

    def update(
        self,
        action: Any,
        reward: Any,
        info: Any | None = None,
    ) -> None:
        """Assimilate an observed outcome and propagate beliefs one step."""
        del info

        if action not in self.beliefs:
            return

        if action == self.last_action:
            self.run_length += 1
        else:
            self.run_length = 1
        self.last_action = action
        self.trial_count += 1

        selected = self.beliefs[action]
        try:
            observed_reward = float(reward)
        except (TypeError, ValueError):
            observed_reward = float("nan")

        if isfinite(observed_reward):
            innovation = observed_reward - selected.mean
            predictive_variance = max(
                selected.variance + self.observation_variance,
                _EPSILON,
            )
            kalman_gain = selected.variance / predictive_variance
            selected.mean += kalman_gain * innovation
            selected.variance = max(
                selected.variance * (1.0 - kalman_gain),
                _EPSILON,
            )
            self.last_surprise = innovation / sqrt(predictive_variance)
        else:
            self.last_surprise = 0.0

        # Beliefs stored after update are priors for the next decision.
        for belief in self.beliefs.values():
            belief.variance = min(
                belief.variance + self.process_variance,
                self.variance_ceiling,
            )


HISTORY_PARAMETER_NAMES = (
    "persistence_trace_weight",
    "surprise_trace_weight",
    "frequency_weight",
    "recency_weight",
)


class Agent(KalmanAgent):
    """Kalman policy augmented by causal, trajectory-local history features.

    Zero history coefficients reproduce the original nine-parameter policy.
    All coefficients remain fixed during evaluation; only history state updates.
    """

    _FITTED_PARAMETER_NAMES = (
        KalmanAgent._FITTED_PARAMETER_NAMES | set(HISTORY_PARAMETER_NAMES)
    )

    def __init__(self, config: Mapping[str, Any] | Any | None = None) -> None:
        super().__init__(config)
        model = get_field(config or {}, "model", {}) or {}
        fitted = self._load_fitted_parameters(
            get_field(model, "fitted_params_path", None)
        )
        for name in HISTORY_PARAMETER_NAMES:
            value = float(fitted.get(name, get_field(model, name, 0.0)))
            if not isfinite(value):
                raise ValueError(f"History coefficient {name!r} must be finite")
            setattr(self, name, value)

    def reset(self, context: Mapping[str, Any] | Any) -> None:
        """Clear every history accumulator together with the Kalman beliefs."""
        super().reset(context)
        self.choice_counts = dict.fromkeys(self.actions, 0)
        self.last_choice_times = dict.fromkeys(self.actions, 0)
        self.past_stays = 0
        self.past_switches = 0
        self.surprise_trace = 0.0

    def history_features(
        self,
    ) -> tuple[float, float, dict[Any, float], dict[Any, float]]:
        """Return pre-choice persistence, surprise, frequency and recency."""
        frequency = {a: log1p(self.choice_counts[a]) for a in self.actions}
        recency = {
            a: -log1p(self.trial_count - self.last_choice_times[a])
            for a in self.actions
        }
        mean_frequency = sum(frequency.values()) / len(self.actions)
        mean_recency = sum(recency.values()) / len(self.actions)
        return (
            log((self.past_stays + 2.0) / (self.past_switches + 2.0)),
            self.surprise_trace,
            {a: frequency[a] - mean_frequency for a in self.actions},
            {a: recency[a] - mean_recency for a in self.actions},
        )

    def _action_utilities(self) -> dict[Any, float]:
        utilities = super()._action_utilities()
        _, _, frequency, recency = self.history_features()
        return {
            a: utilities[a]
            + self.frequency_weight * frequency[a]
            + self.recency_weight * recency[a]
            for a in self.actions
        }

    def _stay_probability(self, base_probabilities: Mapping[Any, float]) -> float:
        probability = clip(
            base_probabilities[self.last_action], 1e-9, 1.0 - 1e-9
        )
        persistence, trace, _, _ = self.history_features()
        logit = (
            self.stay_intercept
            + self.stay_value_weight * log(probability / (1.0 - probability))
            + self.surprise_linear_weight * self.last_surprise
            + self.surprise_quadratic_weight * self.last_surprise**2
            + self.run_length_weight * log1p(self.run_length)
            + self.trial_index_weight * log1p(self.trial_count)
            + self.persistence_trace_weight * persistence
            + self.surprise_trace_weight * trace
        )
        return stable_sigmoid(logit)

    def update(self, action: Any, reward: Any, info: Any | None = None) -> None:
        """Update history only after the current action/reward are revealed."""
        if action not in self.beliefs:
            return
        if self.trial_count:
            if action == self.last_action:
                self.past_stays += 1
            else:
                self.past_switches += 1
        super().update(action, reward, info)
        self.choice_counts[action] += 1
        self.last_choice_times[action] = self.trial_count
        self.surprise_trace = 0.8 * self.surprise_trace + 0.2 * self.last_surprise
