import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(
    "data/raw/public_train_reward_schedules.jsonl"
)
OUTPUT_DIR = Path("analysis/figures")
RANDOM_SEED = 42
N_EXAMPLE_TRAJECTORIES = 5


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
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

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def extract_schedule(record: dict[str, Any]) -> np.ndarray:
    """
    Extract the trial-by-option reward schedule from one record.

    Expected output shape:
        number_of_trials × number_of_options
    """
    possible_schedule_keys = [
        "reward_schedule",
        "reward_schedules",
        "schedule",
        "payoff_schedule",
    ]

    schedule = None

    for key in possible_schedule_keys:
        if key in record:
            schedule = record[key]
            break

    if schedule is None:
        raise KeyError(
            "Could not find a reward schedule field. "
            f"Available top-level keys: {list(record.keys())}"
        )

    # Case 1:
    # [
    #   {"trial_index": 0, "available_rewards": [...]},
    #   ...
    # ]
    if isinstance(schedule, list) and schedule:
        first_item = schedule[0]

        if isinstance(first_item, dict):
            possible_reward_keys = [
                "available_rewards",
                "rewards",
                "option_rewards",
                "payoffs",
            ]

            rows = []

            for trial in schedule:
                reward_values = None

                for reward_key in possible_reward_keys:
                    if reward_key in trial:
                        reward_values = trial[reward_key]
                        break

                if reward_values is None:
                    raise KeyError(
                        "Could not find option rewards in a schedule trial. "
                        f"Available keys: {list(trial.keys())}"
                    )

                rows.append(
                    [float(value) for value in reward_values]
                )

            array = np.asarray(rows, dtype=float)

        # Case 2:
        # [[r_option_0, r_option_1, ...], ...]
        elif isinstance(first_item, list):
            array = np.asarray(schedule, dtype=float)

        else:
            raise TypeError(
                "Unsupported reward schedule structure. "
                f"First schedule item type: {type(first_item)}"
            )

    else:
        raise TypeError(
            "Reward schedule must be a non-empty list."
        )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D schedule, got shape {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Reward schedule contains NaN or infinite values."
        )

    return array


def get_record_id(
    record: dict[str, Any],
    index: int,
) -> str:
    """Return the most informative record identifier available."""
    possible_id_keys = [
        "trajectory_id",
        "block_id",
        "episode_id",
        "schedule_id",
    ]

    for key in possible_id_keys:
        if key in record:
            return str(record[key])

    context = record.get("context", {})

    for key in possible_id_keys:
        if key in context:
            return str(context[key])

    metadata = context.get("metadata", {})

    for key in possible_id_keys:
        if key in metadata:
            return str(metadata[key])

    return f"record_{index:06d}"


def schedule_hash(schedule: np.ndarray) -> str:
    """
    Create a stable hash for exact duplicate detection.

    Two schedules with the same shape and values get the same hash.
    """
    normalized = np.ascontiguousarray(
        schedule.astype(np.float64)
    )

    hash_input = (
        str(normalized.shape).encode("utf-8")
        + normalized.tobytes()
    )

    return hashlib.sha256(hash_input).hexdigest()


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Calculate correlation, returning NaN for constant arrays."""
    if len(x) < 2 or len(y) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def calculate_schedule_statistics(
    schedule: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Calculate temporal statistics for one reward schedule.

    The main statistic is lag-1 autocorrelation:
    correlation between reward at trial t and trial t+1.
    """
    lag1_correlations = []
    shuffled_correlations = []
    mean_absolute_changes = []
    signed_changes = []
    linear_trends = []

    trial_indices = np.arange(schedule.shape[0])

    for option_index in range(schedule.shape[1]):
        option_rewards = schedule[:, option_index]

        lag1 = safe_correlation(
            option_rewards[:-1],
            option_rewards[1:],
        )
        lag1_correlations.append(lag1)

        shuffled_rewards = rng.permutation(option_rewards)

        shuffled_lag1 = safe_correlation(
            shuffled_rewards[:-1],
            shuffled_rewards[1:],
        )
        shuffled_correlations.append(shuffled_lag1)

        changes = np.diff(option_rewards)

        mean_absolute_changes.append(
            float(np.mean(np.abs(changes)))
        )
        signed_changes.extend(changes.tolist())

        if np.std(option_rewards) == 0:
            linear_trends.append(np.nan)
        else:
            slope = np.polyfit(
                trial_indices,
                option_rewards,
                deg=1,
            )[0]
            linear_trends.append(float(slope))

    valid_lag1 = np.asarray(
        lag1_correlations,
        dtype=float,
    )
    valid_shuffled = np.asarray(
        shuffled_correlations,
        dtype=float,
    )
    valid_trends = np.asarray(
        linear_trends,
        dtype=float,
    )

    return {
        "mean_lag1_autocorrelation": float(
            np.nanmean(valid_lag1)
        ),
        "mean_shuffled_lag1_autocorrelation": float(
            np.nanmean(valid_shuffled)
        ),
        "mean_absolute_trial_change": float(
            np.mean(mean_absolute_changes)
        ),
        "mean_signed_trial_change": float(
            np.mean(signed_changes)
        ),
        "mean_absolute_linear_trend": float(
            np.nanmean(np.abs(valid_trends))
        ),
        "within_schedule_reward_sd": float(
            np.std(schedule)
        ),
    }


def compare_same_shape_schedules(
    schedules: list[np.ndarray],
    rng: np.random.Generator,
    max_pairs: int = 5000,
) -> np.ndarray:
    """
    Estimate correlations between schedules from different trajectories.

    Schedules must have identical shape for direct element-wise comparison.
    To keep runtime manageable, randomly sample trajectory pairs.
    """
    groups: dict[tuple[int, int], list[int]] = {}

    for index, schedule in enumerate(schedules):
        groups.setdefault(schedule.shape, []).append(index)

    correlations = []

    for indices in groups.values():
        if len(indices) < 2:
            continue

        number_of_possible_pairs = (
            len(indices) * (len(indices) - 1) // 2
        )
        pairs_to_sample = min(
            max_pairs,
            number_of_possible_pairs,
        )

        sampled_pairs = set()

        while len(sampled_pairs) < pairs_to_sample:
            first, second = rng.choice(
                indices,
                size=2,
                replace=False,
            )

            pair = tuple(sorted((int(first), int(second))))
            sampled_pairs.add(pair)

        for first, second in sampled_pairs:
            schedule_a = schedules[first].ravel()
            schedule_b = schedules[second].ravel()

            correlation = safe_correlation(
                schedule_a,
                schedule_b,
            )

            if np.isfinite(correlation):
                correlations.append(correlation)

    return np.asarray(correlations, dtype=float)


def plot_example_schedules(
    schedules: list[np.ndarray],
    record_ids: list[str],
    output_dir: Path,
) -> None:
    """
    Save one separate figure per example trajectory.

    Each figure contains the four option reward paths.
    """
    number_to_plot = min(
        N_EXAMPLE_TRAJECTORIES,
        len(schedules),
    )

    for index in range(number_to_plot):
        schedule = schedules[index]

        figure, axis = plt.subplots(figsize=(10, 6))

        for option_index in range(schedule.shape[1]):
            axis.plot(
                np.arange(schedule.shape[0]),
                schedule[:, option_index],
                label=f"Option {option_index}",
            )

        axis.set_title(
            f"Reward schedule: {record_ids[index]}"
        )
        axis.set_xlabel("Trial index")
        axis.set_ylabel("Available reward")
        axis.legend()
        axis.grid(alpha=0.25)

        figure.tight_layout()

        output_path = (
            output_dir
            / f"reward_path_{index + 1:02d}.png"
        )

        figure.savefig(output_path, dpi=300)
        plt.close(figure)


def plot_change_distribution(
    schedules: list[np.ndarray],
    output_path: Path,
) -> None:
    """Plot the distribution of trial-to-trial reward changes."""
    all_changes = np.concatenate(
        [
            np.diff(schedule, axis=0).ravel()
            for schedule in schedules
        ]
    )

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.hist(
        all_changes,
        bins=61,
        density=True,
        edgecolor="black",
    )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_title(
        "Distribution of Trial-to-Trial Reward Changes"
    )
    axis.set_xlabel(
        "Reward change: reward(t+1) - reward(t)"
    )
    axis.set_ylabel("Probability density")
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_autocorrelation_comparison(
    observed: np.ndarray,
    shuffled: np.ndarray,
    output_path: Path,
) -> None:
    """
    Compare observed temporal autocorrelation with shuffled schedules.
    """
    figure, axis = plt.subplots(figsize=(9, 6))

    axis.hist(
        observed,
        bins=40,
        density=True,
        alpha=0.6,
        label="Observed schedule",
    )

    axis.hist(
        shuffled,
        bins=40,
        density=True,
        alpha=0.6,
        label="Time-shuffled schedule",
    )

    axis.set_title(
        "Lag-1 Reward Autocorrelation"
    )
    axis.set_xlabel("Lag-1 autocorrelation")
    axis.set_ylabel("Probability density")
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(DATA_PATH)

    schedules = []
    record_ids = []

    for index, record in enumerate(records):
        schedule = extract_schedule(record)

        schedules.append(schedule)
        record_ids.append(
            get_record_id(record, index)
        )

    print("=== Reward schedule structure ===")
    print(f"Number of schedule records: {len(schedules)}")

    shape_counts = Counter(
        schedule.shape for schedule in schedules
    )

    print("Schedule shapes:")

    for shape, count in sorted(shape_counts.items()):
        print(f"  {shape}: {count} trajectories")

    # -------------------------------------------------
    # 1. Are schedules exactly identical?
    # -------------------------------------------------
    hashes = [schedule_hash(s) for s in schedules]
    hash_counts = Counter(hashes)

    unique_schedule_count = len(hash_counts)
    duplicated_record_count = sum(
        count for count in hash_counts.values()
        if count > 1
    )
    largest_duplicate_group = max(hash_counts.values())

    print("\n=== Exact schedule duplication ===")
    print(
        f"Unique exact schedules: "
        f"{unique_schedule_count} / {len(schedules)}"
    )
    print(
        "Records belonging to a duplicated schedule: "
        f"{duplicated_record_count}"
    )
    print(
        "Largest number of trajectories sharing "
        f"one exact schedule: {largest_duplicate_group}"
    )

    # -------------------------------------------------
    # 2. Temporal drift within each trajectory
    # -------------------------------------------------
    statistics = [
        calculate_schedule_statistics(schedule, rng)
        for schedule in schedules
    ]

    observed_lag1 = np.asarray(
        [
            item["mean_lag1_autocorrelation"]
            for item in statistics
        ],
        dtype=float,
    )

    shuffled_lag1 = np.asarray(
        [
            item["mean_shuffled_lag1_autocorrelation"]
            for item in statistics
        ],
        dtype=float,
    )

    absolute_changes = np.asarray(
        [
            item["mean_absolute_trial_change"]
            for item in statistics
        ],
        dtype=float,
    )

    signed_changes = np.asarray(
        [
            item["mean_signed_trial_change"]
            for item in statistics
        ],
        dtype=float,
    )

    print("\n=== Temporal structure within trajectories ===")
    print(
        "Mean observed lag-1 autocorrelation: "
        f"{np.nanmean(observed_lag1):.4f}"
    )
    print(
        "Median observed lag-1 autocorrelation: "
        f"{np.nanmedian(observed_lag1):.4f}"
    )
    print(
        "Mean shuffled lag-1 autocorrelation: "
        f"{np.nanmean(shuffled_lag1):.4f}"
    )
    print(
        "Mean absolute trial-to-trial change: "
        f"{np.mean(absolute_changes):.4f}"
    )
    print(
        "Mean signed trial-to-trial change: "
        f"{np.mean(signed_changes):.4f}"
    )

    # -------------------------------------------------
    # 3. Are schedules similar across trajectories?
    # -------------------------------------------------
    cross_schedule_correlations = (
        compare_same_shape_schedules(
            schedules=schedules,
            rng=rng,
        )
    )

    print("\n=== Similarity across trajectories ===")

    if len(cross_schedule_correlations) > 0:
        print(
            "Mean cross-trajectory schedule correlation: "
            f"{np.mean(cross_schedule_correlations):.4f}"
        )
        print(
            "Median cross-trajectory schedule correlation: "
            f"{np.median(cross_schedule_correlations):.4f}"
        )
        print(
            "Cross-trajectory correlation percentiles "
            "5/25/75/95:",
            np.round(
                np.percentile(
                    cross_schedule_correlations,
                    [5, 25, 75, 95],
                ),
                4,
            ),
        )
    else:
        print(
            "No comparable schedule pairs were available."
        )

    # -------------------------------------------------
    # 4. Save summary
    # -------------------------------------------------
    summary = {
        "random_seed": RANDOM_SEED,
        "number_of_schedule_records": len(schedules),
        "schedule_shape_counts": {
            str(shape): count
            for shape, count in shape_counts.items()
        },
        "unique_exact_schedules": unique_schedule_count,
        "records_in_duplicated_schedules": (
            duplicated_record_count
        ),
        "largest_duplicate_group": largest_duplicate_group,
        "mean_observed_lag1_autocorrelation": float(
            np.nanmean(observed_lag1)
        ),
        "median_observed_lag1_autocorrelation": float(
            np.nanmedian(observed_lag1)
        ),
        "mean_shuffled_lag1_autocorrelation": float(
            np.nanmean(shuffled_lag1)
        ),
        "mean_absolute_trial_change": float(
            np.mean(absolute_changes)
        ),
        "mean_signed_trial_change": float(
            np.mean(signed_changes)
        ),
        "mean_cross_trajectory_correlation": (
            float(np.mean(cross_schedule_correlations))
            if len(cross_schedule_correlations) > 0
            else None
        ),
        "median_cross_trajectory_correlation": (
            float(np.median(cross_schedule_correlations))
            if len(cross_schedule_correlations) > 0
            else None
        ),
    }

    summary_path = (
        OUTPUT_DIR / "reward_dynamics_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------
    # 5. Figures
    # -------------------------------------------------
    plot_example_schedules(
        schedules=schedules,
        record_ids=record_ids,
        output_dir=OUTPUT_DIR,
    )

    plot_change_distribution(
        schedules,
        OUTPUT_DIR
        / "reward_trial_to_trial_changes.png",
    )

    plot_autocorrelation_comparison(
        observed=observed_lag1,
        shuffled=shuffled_lag1,
        output_path=(
            OUTPUT_DIR
            / "reward_lag1_autocorrelation.png"
        ),
    )

    print("\n=== Saved outputs ===")
    print(summary_path)
    print(
        OUTPUT_DIR
        / "reward_trial_to_trial_changes.png"
    )
    print(
        OUTPUT_DIR
        / "reward_lag1_autocorrelation.png"
    )
    print(
        f"{N_EXAMPLE_TRAJECTORIES} example "
        "reward-path figures"
    )


if __name__ == "__main__":
    main()