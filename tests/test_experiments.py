"""Numerical and causal checks for the offline candidate experiment."""

import unittest

from agent import KalmanAgent
from evaluate_public import official_accuracy_credit
from experiments.history_agent import HistoryAgent

try:
    import numpy as np
    from experiments.final_candidates import NAMES, extract, objective, terms
    FIT_AVAILABLE = True
except ImportError:
    FIT_AVAILABLE = False


def toy():
    rng = np.random.default_rng(26)
    return [{"context": {"available_actions": ["a", "b", "c", "d"], "subject_id": str(i)},
             "trials": [{"action": str(rng.choice(["a", "b", "c", "d"])), "reward": float(rng.normal(50, 20))} for _ in range(40)]} for i in range(3)]


@unittest.skipUnless(FIT_AVAILABLE, "Install requirements-fit.txt for offline numerical tests")
class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([1.1, .02, -.7, 1.8, 1.4, -1.1, 1.3, .26, .013, .2, -.15, .1, .3])
        self.config = {"model": dict(zip(NAMES, self.theta)) | {"process_variance": 25.0, "observation_variance": 200.0, "fitted_params_path": None}}
        self.trajectories = toy()
        self.batch = extract(self.trajectories, self.config, {str(i): i for i in range(3)})

    def test_analytic_gradient(self):
        value, gradient = objective(self.theta, self.batch, 1e-3)
        numerical = np.zeros_like(self.theta)
        for i in range(len(self.theta)):
            delta = np.zeros_like(self.theta)
            delta[i] = 1e-6
            numerical[i] = (objective(self.theta + delta, self.batch, 1e-3)[0] - objective(self.theta - delta, self.batch, 1e-3)[0]) / 2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=2e-7)
        self.assertTrue(np.isfinite(value))

    def test_vectorized_online_parity(self):
        expected = terms(self.theta, self.batch)[0]
        collected = []
        agent = HistoryAgent(self.config)
        for trajectory in self.trajectories:
            agent.reset(trajectory["context"])
            for i, trial in enumerate(trajectory["trials"]):
                p = agent.predict(None)["action_probs"]
                self.assertEqual(p, agent.predict([{"future": "must be ignored"}])["action_probs"])
                if i:
                    collected.append(list(p.values()))
                agent.update(trial["action"], trial["reward"])
        np.testing.assert_allclose(collected, expected, atol=1e-12, rtol=1e-10)

    def test_zero_extension_matches_frozen_model(self):
        self.config["model"].update(dict.fromkeys(NAMES[9:], 0.0))
        base, candidate = KalmanAgent(self.config), HistoryAgent(self.config)
        for trajectory in self.trajectories:
            base.reset(trajectory["context"])
            candidate.reset(trajectory["context"])
            for trial in trajectory["trials"]:
                self.assertEqual(base.predict(None), candidate.predict(None))
                base.update(trial["action"], trial["reward"])
                candidate.update(trial["action"], trial["reward"])

    def test_exact_not_near_ties(self):
        self.assertEqual(official_accuracy_credit({0: .5, 1: .5}, 0), .5)
        self.assertEqual(official_accuracy_credit({0: .5 + 1e-13, 1: .5 - 1e-13}, 1), 0.0)

    def test_group_subset(self):
        train, valid = self.batch.subset([0, 1]), self.batch.subset([2])
        self.assertFalse(set(train.subject) & set(valid.subject))
        self.assertEqual(len(train.last) + len(valid.last), len(self.batch.last))
        self.assertEqual(len(train.first_subject) + len(valid.first_subject), 3)

    def test_suffix_cannot_change_prefix_features(self):
        shortened = [{"context": t["context"], "trials": t["trials"][:20]} for t in self.trajectories]
        prefix = extract(shortened, self.config, {str(i): i for i in range(3)})
        for subject in range(3):
            np.testing.assert_array_equal(prefix.utility[prefix.subject == subject], self.batch.utility[self.batch.subject == subject][:19])
            np.testing.assert_array_equal(prefix.gate_static[prefix.subject == subject], self.batch.gate_static[self.batch.subject == subject][:19])


if __name__ == "__main__":
    unittest.main()
