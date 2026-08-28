"""Evaluate the submitted agent on public MindRL trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
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
        if float(value) == maximum
    ]
    return 1.0 / len(winners) if action in winners else 0.0


def evaluate(agent: Any, data_path: Path) -> dict[str, Any]:
    """Score causally; split switch loss using the final, post-lapse probabilities."""
    total_nll = 0.0
    accuracy_credit = 0.0
    trajectories = 0
    trials = 0
    category_nll = {"first": 0.0, "stay": 0.0, "switch": 0.0}
    category_count = {"first": 0, "stay": 0, "switch": 0}
    switch_gate_loss = 0.0
    switch_target_loss = 0.0

    with data_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            agent.reset(trajectory["context"])
            history: list[dict[str, Any]] = []
            previous_action: Any = None
            trajectories += 1

            for index, trial in enumerate(trajectory["trials"]):
                probabilities = agent.predict(history)["action_probs"]
                if set(probabilities) != set(trajectory["context"]["available_actions"]):
                    raise ValueError("Prediction keys do not match available actions")
                if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probabilities.values()):
                    raise ValueError("Prediction contains an invalid probability")
                if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-10):
                    raise ValueError("Prediction is not normalized")
                action = trial["action"]
                probability = max(float(probabilities[action]), 1e-12)
                nll = -math.log(probability)

                if index == 0:
                    category = "first"
                elif action == previous_action:
                    category = "stay"
                else:
                    category = "switch"
                    switch_mass = sum(p for a, p in probabilities.items() if a != previous_action)
                    switch_gate_loss -= math.log(max(switch_mass, 1e-12))
                    switch_target_loss += nll + math.log(max(switch_mass, 1e-12))

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

    if not trials:
        raise ValueError("No trials to evaluate")
    return {
        "trajectories": trajectories,
        "trials": trials,
        "nll": total_nll / trials,
        "accuracy": accuracy_credit / trials,
        "categories": {
            key: {"trials": category_count[key], "nll": category_nll[key] / category_count[key] if category_count[key] else None}
            for key in category_count
        },
        "switch_gate_nll": switch_gate_loss / category_count["switch"] if category_count["switch"] else None,
        "switch_target_nll": switch_target_loss / category_count["switch"] if category_count["switch"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="data/public_train.jsonl")
    parser.add_argument("--output", type=Path, help="Optional aggregate JSON (no participant records).")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    started = time.perf_counter()
    metrics = evaluate(Agent(config), Path(args.data))
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["python_version"] = platform.python_version()
    print(f"Trajectories: {metrics['trajectories']}")
    print(f"Trials: {metrics['trials']}")
    print(f"Mean NLL: {metrics['nll']:.6f}")
    print(f"Official accuracy: {metrics['accuracy']:.4%}")
    for category, label in (
        ("first", "First trials"),
        ("stay", "Stay trials"),
        ("switch", "Switch trials"),
    ):
        item = metrics["categories"][category]
        formatted = f"{item['nll']:.6f}" if item["nll"] is not None else "n/a"
        print(f"{label}: {item['trials']} (Mean NLL: {formatted})")
    if metrics["switch_gate_nll"] is not None:
        print(f"Switch NLL decomposition: gate={metrics['switch_gate_nll']:.6f}, target={metrics['switch_target_nll']:.6f}")
    if args.output:
        metrics["data_sha256"] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
        metrics["config_sha256"] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
        metrics["agent_sha256"] = hashlib.sha256((Path(__file__).parent / "agent.py").read_bytes()).hexdigest()
        artifact = config.get("model", {}).get("fitted_params_path")
        if artifact:
            metrics["parameters_sha256"] = hashlib.sha256((Path(__file__).parent / artifact).read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
