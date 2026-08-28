"""Independent algebra check of Q6; no auditor implementation is imported."""

import math
import random
import unittest

from agent import Agent, normalize_probabilities, stable_softmax


class EquivalenceTests(unittest.TestCase):
    def test_matched_single_softmax(self):
        rng = random.Random(20260827)
        for n_actions in (2, 4, 7):
            agent = Agent({"model": {"fitted_params_path": None, "stay_intercept": -.7, "stay_value_weight": 1.8,
                                     "surprise_linear_weight": 1.4, "surprise_quadratic_weight": -1.1,
                                     "run_length_weight": 1.3, "trial_index_weight": .26, "lapse": .013}})
            agent.reset({"available_actions": [f"option_{i}" for i in range(n_actions)]})
            for _ in range(60):
                agent.update(rng.choice(agent.actions), rng.uniform(20, 80))
                utilities = agent._action_utilities()
                s = agent._stay_probability(stable_softmax(utilities))
                if s == 0.0 or s == 1.0:
                    continue
                # v_last = gate logit + logsumexp(non-last utilities).
                other = [u for a, u in utilities.items() if a != agent.last_action]
                largest = max(other)
                log_sum = largest + math.log(sum(math.exp(u - largest) for u in other))
                matched = dict(utilities)
                matched[agent.last_action] = math.log(s / (1.0 - s)) + log_sum
                softmax = stable_softmax(matched)
                softmax = normalize_probabilities({a: (1.0 - agent.lapse) * p + agent.lapse / n_actions for a, p in softmax.items()})
                expected = agent.predict(None)["action_probs"]
                for action in expected:
                    self.assertAlmostEqual(expected[action], softmax[action], places=12)


if __name__ == "__main__":
    unittest.main()
