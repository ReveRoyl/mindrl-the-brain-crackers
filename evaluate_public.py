import json
import math

import yaml

from agent import Agent


with open("config.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

agent = Agent(config)

total_nll = 0.0
correct = 0
n_trials = 0
n_trajectories = 0

with open("data/public_train.jsonl", encoding="utf-8") as file:
    for line in file:
        trajectory = json.loads(line)
        agent.reset(trajectory["context"])

        history = []
        n_trajectories += 1

        for trial in trajectory["trials"]:
            prediction = agent.predict(history)
            probabilities = prediction["action_probs"]

            action = trial["action"]
            probability = max(float(probabilities[action]), 1e-12)

            total_nll -= math.log(probability)
            predicted_action = max(probabilities, key=probabilities.get)
            correct += int(predicted_action == action)
            n_trials += 1

            agent.update(
                action=action,
                reward=trial["reward"],
                info=trial.get("info"),
            )
            history.append(trial)

print(f"Trajectories: {n_trajectories}")
print(f"Trials: {n_trials}")
print(f"Mean NLL: {total_nll / n_trials:.6f}")
print(f"Accuracy: {correct / n_trials:.4%}")