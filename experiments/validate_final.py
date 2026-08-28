"""Public-only end-to-end parity and preserved-reference checks.

No per-trial predictions or subject identifiers are written to the report.
"""

import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from agent import KalmanAgent
from experiments.final_candidates import NAMES, extract, terms, write_json
from fit_public import config_for_feature_extraction, load_trajectories


ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    reference_config = yaml.safe_load((ROOT / "config_baseline.yaml").read_text(encoding="utf-8"))
    artifact_path = ROOT / config["model"]["fitted_params_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    baseline_path = ROOT / "artifacts/fitted_params.json"
    preserved_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert preserved_hash == "7a374bdc31549e6a70b18e8062712e34c7f377c87d3498d7e1f14376c2d32a01"
    data_path = ROOT / "data/public_train.jsonl"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert data_hash == artifact["data_sha256"]
    cv = json.loads((ROOT / "artifacts/final_development_cv.json").read_text(encoding="utf-8"))
    assert artifact["selected_candidate"] == cv["selected"]
    assert cv["pooled"][cv["selected"]]["qualifies"]
    assert len(cv["folds"]) == 5
    attempts = [a for fold in cv["folds"] for candidate in fold["candidates"].values() for a in candidate.get("attempts", [])]

    # Match the official dynamic-loader condition: do not register in sys.modules.
    specification = importlib.util.spec_from_file_location("final_validation_agent", ROOT / "agent.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    runtime = module.Agent(config)
    compatible_reference = module.Agent(reference_config)
    original_policy = KalmanAgent(reference_config)
    trajectories = load_trajectories(data_path)
    subjects = sorted({t["context"]["subject_id"] for t in trajectories})
    params = artifact["parameters"]
    feature_config = config_for_feature_extraction(config, params["process_variance"], params["observation_variance"])
    feature_config["model"].update(params)
    batch = extract(trajectories, feature_config, {s: i for i, s in enumerate(subjects)})
    expected = terms(np.array([params[name] for name in NAMES]), batch)[0]
    max_runtime_error = max_reference_error = 0.0
    later_index = trial_count = 0
    for trajectory_index, trajectory in enumerate(trajectories):
        for agent in (runtime, compatible_reference, original_policy):
            agent.reset(trajectory["context"])
        for index, trial in enumerate(trajectory["trials"]):
            probabilities = runtime.predict(None)["action_probs"]
            if index:
                target = expected[later_index]
                later_index += 1
            else:
                target = np.full(len(runtime.actions), 1.0 / len(runtime.actions))
            max_runtime_error = max(max_runtime_error, float(np.max(np.abs(np.array(list(probabilities.values())) - target))))
            old = original_policy.predict(None)["action_probs"]
            compatible = compatible_reference.predict(None)["action_probs"]
            max_reference_error = max(max_reference_error, max(abs(old[a] - compatible[a]) for a in old))
            for agent in (runtime, compatible_reference, original_policy):
                agent.update(trial["action"], trial.get("reward"), trial.get("info"))
            trial_count += 1
        if (trajectory_index + 1) % 500 == 0:
            print(f"Verified {trajectory_index + 1}/{len(trajectories)} trajectories", flush=True)
    assert later_index == len(expected)
    assert trial_count == 320080
    assert max_runtime_error < 1e-10
    assert max_reference_error == 0.0

    sequential = json.loads((ROOT / "artifacts/final_public_evaluation.json").read_text(encoding="utf-8"))
    for key in ("nll", "accuracy"):
        assert abs(sequential[key] - artifact["final_public_fit_metrics"][key]) < 1e-10
    assert abs(sequential["switch_gate_nll"] + sequential["switch_target_nll"] - sequential["categories"]["switch"]["nll"]) < 1e-10
    assert hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest() == sequential["agent_sha256"]
    assert hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest() == sequential["config_sha256"]
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == sequential["parameters_sha256"]
    card = (ROOT / "interpretation_card.md").read_text(encoding="utf-8")
    for name, value in params.items():
        assert f"| `{name}` | {value:.6f} |" in card
    metadata = yaml.safe_load((ROOT / "submission.yaml").read_text(encoding="utf-8"))
    assert metadata["team"]["team_id"] == "the_brain_crackers"
    assert metadata["runtime_profile"]["requires_external_api"] is False
    assert "commit_hash" not in metadata and "commit_hash" not in metadata["submission"]
    for field in ("agent_path", "config_path", "requirements_path", "interpretation_card_path"):
        assert (ROOT / metadata["submission"][field]).is_file()

    report = {
        "status": "pass", "verified_on": "2026-08-27", "python": platform.python_version(),
        "trajectories": len(trajectories), "trials": trial_count,
        "max_runtime_vectorized_probability_error": max_runtime_error,
        "max_preserved_reference_probability_error": max_reference_error,
        "preserved_parameters_sha256": preserved_hash, "data_sha256": data_hash,
        "cv_fitting_attempts": len(attempts),
        "cv_successful_attempts": sum(a["success"] for a in attempts),
        "final_fitting_attempts": len(artifact["fit_attempts"]),
        "final_successful_attempts": sum(a["success"] for a in artifact["fit_attempts"]),
        "card_parameter_values_match": True, "metadata_paths_exist": True,
        "sequential_metrics_match_fit": True, "official_style_dynamic_import": "pass",
        "elapsed_seconds": time.perf_counter() - started,
        "limits": ["Public data only; no official CI or hidden evaluation executed.", "This parity check does not test psychological identifiability."]
    }
    write_json(ROOT / "artifacts/final_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
