"""Behavioral and loader-compatibility tests for the submitted agent."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT / "agent.py"
CONFIG_PATH = ROOT / "config.yaml"


def load_like_official_runner() -> type:
    """Load without pre-registering the module in ``sys.modules``."""
    module_name = "mindrl_submission_agent_test"
    sys.modules.pop(module_name, None)
    specification = importlib.util.spec_from_file_location(
        module_name,
        AGENT_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not create the agent module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.Agent


class AgentTests(unittest.TestCase):
    """Check interface, probability, update, and reset invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG_PATH.open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.agent_class = load_like_official_runner()
        cls.context = {
            "available_actions": [0, 1, 2, 3],
            "num_options": 4,
        }

    def make_agent(self) -> Any:
        return self.agent_class(self.config)

    def assert_distribution(self, probabilities: dict[Any, float]) -> None:
        self.assertEqual(set(probabilities), {0, 1, 2, 3})
        self.assertTrue(
            all(
                math.isfinite(probability) and probability >= 0.0
                for probability in probabilities.values()
            )
        )
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)

    def test_official_style_dynamic_import(self) -> None:
        self.assertTrue(callable(self.agent_class))

    def test_first_prediction_is_uniform(self) -> None:
        agent = self.make_agent()
        agent.reset(self.context)
        probabilities = agent.predict([])["action_probs"]
        self.assert_distribution(probabilities)
        for probability in probabilities.values():
            self.assertAlmostEqual(probability, 0.25, places=12)

    def test_updates_produce_valid_probabilities(self) -> None:
        agent = self.make_agent()
        agent.reset(self.context)
        history: list[dict[str, Any]] = []
        observations = (
            (2, 80.0),
            (2, 75.0),
            (1, 20.0),
            (3, 95.0),
        )
        for action, reward in observations:
            probabilities = agent.predict(history)["action_probs"]
            self.assert_distribution(probabilities)
            trial = {"action": action, "reward": reward, "info": {}}
            agent.update(action, reward, trial["info"])
            history.append(trial)
        self.assert_distribution(agent.predict(history)["action_probs"])

    def test_reset_removes_previous_trajectory_state(self) -> None:
        agent = self.make_agent()
        agent.reset(self.context)
        agent.update(action=2, reward=99.0)
        changed = agent.predict([])["action_probs"]
        self.assertNotEqual(changed[2], 0.25)

        agent.reset(self.context)
        reset_probabilities = agent.predict([])["action_probs"]
        for probability in reset_probabilities.values():
            self.assertAlmostEqual(probability, 0.25, places=12)

    def test_missing_reward_remains_safe(self) -> None:
        agent = self.make_agent()
        agent.reset(self.context)
        agent.update(action=0, reward=None)
        probabilities = agent.predict([])["action_probs"]
        self.assert_distribution(probabilities)

    def test_unavailable_action_is_ignored(self) -> None:
        agent = self.make_agent()
        agent.reset(self.context)
        before = agent.predict([])["action_probs"]
        agent.update(action=99, reward=100.0)
        after = agent.predict([])["action_probs"]
        self.assertEqual(before, after)

    def test_arbitrary_labels_and_action_counts(self) -> None:
        for actions in (["only"], [10, 30], ["A", "B", "C", "D", "E", "F", "G"]):
            agent = self.make_agent()
            agent.reset(SimpleNamespace(available_actions=actions))
            for index in range(12):
                probabilities = agent.predict(None)["action_probs"]
                self.assertEqual(set(probabilities), set(actions))
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)
                self.assertTrue(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities.values()))
                agent.update(actions[index % len(actions)], 50.0 + index)

    def test_prediction_is_read_only_and_ignores_future_history(self) -> None:
        agent = self.make_agent()
        agent.reset(self.context)
        agent.update(2, 61.0)
        before = agent.predict([])
        for _ in range(3):
            self.assertEqual(before, agent.predict([{"action": 0, "reward": 9999.0}]))

    def test_all_state_reset_matches_fresh_agent(self) -> None:
        used, fresh = self.make_agent(), self.make_agent()
        used.reset(self.context)
        for action in [2, 2, 1, 2, 3, 3]:
            used.update(action, 75.0)
        used.reset(self.context)
        fresh.reset(self.context)
        for action in [1, 0, 0, 2]:
            self.assertEqual(used.predict(None), fresh.predict(None))
            used.update(action, 40.0)
            fresh.update(action, 40.0)

    def test_nonfinite_or_unparseable_reward(self) -> None:
        for reward in [float("nan"), float("inf"), -float("inf"), "", "unavailable"]:
            agent = self.make_agent()
            agent.reset(self.context)
            agent.update(0, reward)
            self.assert_distribution(agent.predict(None)["action_probs"])
            self.assertEqual(agent.last_surprise, 0.0)

    def test_parameters_do_not_change_online(self) -> None:
        agent = self.make_agent()
        before = {name: getattr(agent, name) for name in agent._FITTED_PARAMETER_NAMES}
        agent.reset(self.context)
        for i in range(100):
            agent.predict(None)
            agent.update(i % 4, float(i))
        self.assertEqual(before, {name: getattr(agent, name) for name in before})

    def test_context_ids_and_metadata_do_not_change_prediction(self) -> None:
        first, second = self.make_agent(), self.make_agent()
        first.reset(dict(self.context, subject_id="unused_A", metadata={"future": 100}))
        second.reset(dict(self.context, subject_id="unused_B", metadata={"future": 0}))
        for action, reward in [(0, 60.0), (1, 40.0), (1, 70.0)]:
            self.assertEqual(first.predict(None), second.predict(None))
            first.update(action, reward, {"unrevealed": 0})
            second.update(action, reward, {"unrevealed": 100})

    def test_invalid_context_rejected(self) -> None:
        for actions in ([], [0, 0]):
            with self.assertRaises(ValueError):
                self.make_agent().reset({"available_actions": actions})


if __name__ == "__main__":
    unittest.main()
