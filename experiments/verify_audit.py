"""Recheck Q7 arithmetic from an authorized checkout of the pinned audit evidence.

Does not run simulations, optimize the target, or read any other team's model.
python -m experiments.verify_audit --audit-root PATH_TO_SUBMISSION_WL
"""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root / "audits/the_brain_crackers/testing-parameter-recovery/outputs"
    with (root / "model_fits/fit_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "generating_parameters.csv").open(encoding="utf-8", newline="") as stream:
        generating = {r["parameter_set_id"]: r for r in csv.DictReader(stream)}
    if len(rows) != 100 or len({r["parameter_set_id"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique fits")
    report = json.loads(Path("artifacts/audit_evidence_verification.json").read_text(encoding="utf-8"))
    for item in report["recovery"]:
        name = item["parameter"]
        x = [float(r["true_" + name]) for r in rows]
        y = [float(r["recovered_" + name]) for r in rows]
        for row, value in zip(rows, x):
            assert value == float(generating[row["parameter_set_id"]][name])
        mx, my = sum(x) / len(x), sum(y) / len(y)
        correlation = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx)**2 for a in x) * sum((b - my)**2 for b in y))
        mse = sum((a - b)**2 for a, b in zip(x, y)) / len(x)
        assert abs(correlation - item["pearson_r"]) < 1e-12
        assert abs(mse - item["mse"]) < 1e-12
        print(f"{name}: r={correlation:.10f}, MSE={mse:.10f}")
    print("Matches saved arithmetic verification; simulations/fits were not rerun.")


if __name__ == "__main__":
    main()
