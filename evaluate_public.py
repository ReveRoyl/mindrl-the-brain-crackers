"""Evaluate the submitted agent on public MindRL trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from agent import Agent


def official_accuracy_credit(
    probabilities: dict[Any, float],
    action: Any,
) -> float:
    """Match the official metric's fractional credit for tied maxima."""
    maximum = max(float(value) for value in probabilities.values())
    winners = [
        candidate
        for candidate, value in probabilities.items()
        if math.isclose(float(value), maximum, rel_tol=0.0, abs_tol=1e-12)
    ]
    return 1.0 / len(winners) if action in winners else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="data/public_train.jsonl")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    agent = Agent(config)
    total_nll = 0.0
    accuracy_credit = 0.0
    trajectories = 0
    trials = 0
    category_nll = {"first": 0.0, "stay": 0.0, "switch": 0.0}
    category_count = {"first": 0, "stay": 0, "switch": 0}

    with Path(args.data).open(encoding="utf-8") as stream:
        for line in stream:
            trajectory = json.loads(line)
            agent.reset(trajectory["context"])
            history: list[dict[str, Any]] = []
            previous_action: Any = None
            trajectories += 1

            for index, trial in enumerate(trajectory["trials"]):
                probabilities = agent.predict(history)["action_probs"]
                action = trial["action"]
                probability = max(float(probabilities[action]), 1e-12)
                nll = -math.log(probability)

                if index == 0:
                    category = "first"
                elif action == previous_action:
                    category = "stay"
                else:
                    category = "switch"

                total_nll += nll
                accuracy_credit += official_accuracy_credit(
                    probabilities,
                    action,
                )
                category_nll[category] += nll
                category_count[category] += 1
                trials += 1

                agent.update(
                    action=action,
                    reward=trial["reward"],
                    info=trial.get("info"),
                )
                history.append(trial)
                previous_action = action

    print(f"Trajectories: {trajectories}")
    print(f"Trials: {trials}")
    print(f"Mean NLL: {total_nll / trials:.6f}")
    print(f"Official accuracy: {accuracy_credit / trials:.4%}")
    for category, label in (
        ("first", "First trials"),
        ("stay", "Stay trials"),
        ("switch", "Switch trials"),
    ):
        count = category_count[category]
        mean_nll = category_nll[category] / count
        print(f"{label}: {count} (Mean NLL: {mean_nll:.6f})")


if __name__ == "__main__":
    main()
