import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATA_PATH = Path("data/raw/public_train.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return records


def main() -> None:
    trajectories = load_jsonl(DATA_PATH)

    all_rewards = []
    all_rts = []
    action_counts = Counter()

    stay_count = 0
    transition_count = 0

    subject_trajectory_counts = Counter()
    block_ids = Counter()

    trajectory_reward_means = []
    trajectory_stay_rates = []

    trial_index_errors = []
    duplicate_trajectory_ids = []
    seen_trajectory_ids = set()

    for trajectory in trajectories:
        context = trajectory["context"]
        trials = trajectory["trials"]

        subject_id = context["subject_id"]
        trajectory_id = context["trajectory_id"]
        block_id = context.get("metadata", {}).get("block_id")

        subject_trajectory_counts[subject_id] += 1

        if block_id is not None:
            block_ids[block_id] += 1

        if trajectory_id in seen_trajectory_ids:
            duplicate_trajectory_ids.append(trajectory_id)
        seen_trajectory_ids.add(trajectory_id)

        expected_indices = list(range(len(trials)))
        observed_indices = [trial["trial_index"] for trial in trials]

        if observed_indices != expected_indices:
            trial_index_errors.append(trajectory_id)

        trajectory_rewards = []
        trajectory_stays = []

        previous_action = None

        for trial in trials:
            action = trial["action"]
            reward = float(trial["reward"])
            rt = trial.get("info", {}).get("rt")

            action_counts[action] += 1
            all_rewards.append(reward)
            trajectory_rewards.append(reward)

            if rt is not None:
                all_rts.append(float(rt))

            if previous_action is not None:
                stayed = action == previous_action
                stay_count += int(stayed)
                transition_count += 1
                trajectory_stays.append(int(stayed))

            previous_action = action

        trajectory_reward_means.append(np.mean(trajectory_rewards))

        if trajectory_stays:
            trajectory_stay_rates.append(np.mean(trajectory_stays))

    rewards = np.asarray(all_rewards, dtype=float)
    rts = np.asarray(all_rts, dtype=float)

    print("=== Data integrity ===")
    print(f"Duplicate trajectory IDs: {len(duplicate_trajectory_ids)}")
    print(f"Trajectories with trial-index errors: {len(trial_index_errors)}")
    print(f"Unique block IDs: {len(block_ids)}")

    print("\n=== Subject structure ===")
    counts = np.asarray(list(subject_trajectory_counts.values()))
    print(f"Trajectories per subject, min: {counts.min()}")
    print(f"Trajectories per subject, max: {counts.max()}")
    print(f"Trajectories per subject, mean: {counts.mean():.2f}")
    print(f"Trajectories per subject, median: {np.median(counts):.2f}")

    print("\n=== Action distribution ===")
    total_actions = sum(action_counts.values())

    for action in sorted(action_counts):
        count = action_counts[action]
        print(
            f"Action {action}: {count} "
            f"({count / total_actions:.2%})"
        )

    print("\n=== Reward distribution ===")
    print(f"Minimum: {rewards.min():.3f}")
    print(f"Maximum: {rewards.max():.3f}")
    print(f"Mean: {rewards.mean():.3f}")
    print(f"Standard deviation: {rewards.std():.3f}")
    print(f"Median: {np.median(rewards):.3f}")
    print(
        "Percentiles 1/5/25/75/95/99:",
        np.percentile(rewards, [1, 5, 25, 75, 95, 99]),
    )

    print("\n=== Reaction-time distribution ===")
    print(f"Minimum non-missing RT: {rts.min():.3f}")
    print(f"Maximum non-missing RT: {rts.max():.3f}")
    print(f"Mean: {rts.mean():.3f}")
    print(f"Median: {np.median(rts):.3f}")
    print(
        "Percentiles 1/5/25/75/95/99:",
        np.percentile(rts, [1, 5, 25, 75, 95, 99]),
    )

    print("\n=== Choice persistence ===")
    overall_stay_rate = stay_count / transition_count
    print(f"Overall stay rate: {overall_stay_rate:.2%}")
    print(
        "Trajectory stay rate mean:",
        f"{np.mean(trajectory_stay_rates):.2%}",
    )
    print(
        "Trajectory stay rate SD:",
        f"{np.std(trajectory_stay_rates):.2%}",
    )

    print("\n=== Between-trajectory reward variation ===")
    print(
        "Mean trajectory reward:",
        f"{np.mean(trajectory_reward_means):.3f}",
    )
    print(
        "SD of trajectory mean reward:",
        f"{np.std(trajectory_reward_means):.3f}",
    )


if __name__ == "__main__":
    main()