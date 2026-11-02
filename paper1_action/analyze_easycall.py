"""Three-seed and speaker-cluster summary of the secondary EasyCall task."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper1_action/results"
SEEDS = (11, 13, 17)


def mean_sd(rows, path):
    values = [row for row in rows]
    for key in path:
        values = [row[key] for row in values]
    return {"mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "seeds": values}


def main():
    data = {}
    preds = {}
    speaker = y = None
    for kind in ("ann", "snn"):
        data[kind] = [json.loads((RESULTS / f"easycall_audio_conv_{kind}_{seed}.json").read_text())
                      for seed in SEEDS]
        preds[kind] = []
        for seed in SEEDS:
            with np.load(RESULTS / f"easycall_audio_conv_{kind}_{seed}_predictions.npz") as z:
                if y is not None and not np.array_equal(y, z["y"]):
                    raise ValueError("Mismatched EasyCall labels")
                if speaker is not None and not np.array_equal(speaker, z["speaker"]):
                    raise ValueError("Mismatched EasyCall speakers")
                y, speaker = z["y"], z["speaker"]
                preds[kind].append(z["scores"].argmax(axis=1))
        preds[kind] = np.stack(preds[kind])
    output = {}
    for kind in ("ann", "snn"):
        rows = data[kind]
        output[kind] = {
            "accuracy": mean_sd(rows, ("test", "accuracy")),
            "macro_f1": mean_sd(rows, ("test", "macro_f1")),
            "dysarthric_accuracy": mean_sd(rows, ("by_speaker_group", "dysarthric", "accuracy")),
            "control_accuracy": mean_sd(rows, ("by_speaker_group", "control", "accuracy")),
            "false_execution_on_noncommands": mean_sd(rows, ("test", "false_execution_on_noncommands")),
            "correct_execution_on_commands": mean_sd(rows, ("test", "correct_execution_on_commands")),
            "wrong_execution_on_commands": mean_sd(rows, ("test", "wrong_execution_on_commands")),
            "emergency_false_execution_count": [r["test"]["emergency_false_execution_count"] for r in rows],
        }
    unique_speakers = np.unique(speaker)
    groups = [np.flatnonzero(speaker == s) for s in unique_speakers]
    delta = (preds["snn"] == y).astype(np.int8) - (preds["ann"] == y).astype(np.int8)
    per_item = delta.mean(axis=0)
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(5000):
        chosen = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in chosen])
        boot.append(per_item[indices].mean())
    output["paired_accuracy_difference_snn_minus_ann"] = {
        "mean": float(per_item.mean()),
        "speaker_cluster_bootstrap_95ci": [float(v) for v in np.percentile(boot, (2.5, 97.5))],
        "test_speakers": int(len(groups)),
    }
    (RESULTS / "easycall_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
