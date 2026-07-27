import json
from collections import Counter
from pathlib import Path
from typing import Any


DATA_PATH = Path("data/raw/public_train.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
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

    task_ids = Counter()
    task_families = Counter()
    subject_ids = Counter()
    trajectory_lengths = []
    num_options = Counter()
    missing_rt = 0
    total_trials = 0

    context_key_sets = Counter()
    trial_key_sets = Counter()
    info_key_sets = Counter()

    for trajectory in trajectories:
        context = trajectory["context"]
        trials = trajectory["trials"]

        task_ids[context["task_id"]] += 1
        task_families[context["task_family"]] += 1
        subject_ids[context["subject_id"]] += 1
        num_options[context["num_options"]] += 1
        trajectory_lengths.append(len(trials))

        context_key_sets[tuple(sorted(context.keys()))] += 1

        for trial in trials:
            total_trials += 1
            trial_key_sets[tuple(sorted(trial.keys()))] += 1

            info = trial.get("info", {})
            info_key_sets[tuple(sorted(info.keys()))] += 1

            if info.get("rt") is None:
                missing_rt += 1

    print("=== Dataset overview ===")
    print(f"Number of trajectories: {len(trajectories)}")
    print(f"Number of trials: {total_trials}")
    print(f"Number of unique subjects: {len(subject_ids)}")
    print(f"Number of unique task IDs: {len(task_ids)}")
    print(f"Number of task families: {len(task_families)}")

    print("\n=== Trajectory length ===")
    print(f"Minimum: {min(trajectory_lengths)}")
    print(f"Maximum: {max(trajectory_lengths)}")
    print(
        f"Mean: {sum(trajectory_lengths) / len(trajectory_lengths):.2f}"
    )

    print("\n=== Task families ===")
    for task_family, count in task_families.most_common():
        print(f"{task_family}: {count} trajectories")

    print("\n=== Number of options ===")
    for option_count, count in sorted(num_options.items()):
        print(f"{option_count} options: {count} trajectories")

    print("\n=== Missing RT ===")
    print(f"Missing RT trials: {missing_rt}")
    print(f"Missing RT rate: {missing_rt / total_trials:.2%}")

    print("\n=== Context field patterns ===")
    for keys, count in context_key_sets.items():
        print(f"{count} trajectories: {keys}")

    print("\n=== Trial field patterns ===")
    for keys, count in trial_key_sets.items():
        print(f"{count} trials: {keys}")

    print("\n=== Info field patterns ===")
    for keys, count in info_key_sets.items():
        print(f"{count} trials: {keys}")


if __name__ == "__main__":
    main()