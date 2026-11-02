"""Auditable class and upstream-error analysis for the SLURP decision task."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_asr_text import edit_distance
from run_text import tokenize


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
RESULTS = ROOT / "paper1_action/results"


def main():
    manifest = json.loads((CACHE / "slurp_audio_manifest.json").read_text())
    actions = manifest["actions"]
    asr = {r["slurp_id"]: r for r in map(json.loads,
              (CACHE / "slurp_asr_tiny_en.jsonl").read_text().splitlines())}
    with np.load(RESULTS / "slurp_asr_snn_11_predictions.npz") as z:
        scores, y, ids = z["scores"], z["y"], z["slurp_id"]
    with np.load(RESULTS / "slurp_oracle_snn_11_predictions.npz") as z:
        oracle_scores, oracle_y = z["scores"], z["y"]
    if not np.array_equal(y, oracle_y):
        raise ValueError("Oracle and ASR labels are not aligned")
    pred, oracle_pred = scores.argmax(axis=1), oracle_scores.argmax(axis=1)
    wer = []
    for identifier in ids:
        row = asr[int(identifier)]
        ref, hyp = tokenize(row["reference"]), tokenize(row["hypothesis"])
        wer.append(edit_distance(ref, hyp) / max(1, len(ref)))
    wer = np.asarray(wer)
    correct = pred == y
    counts = np.bincount(y, minlength=len(actions))
    recalls = np.bincount(y[correct], minlength=len(actions)) / np.maximum(counts, 1)
    wrong = Counter((actions[int(a)], actions[int(b)]) for a, b in zip(y, pred) if a != b)
    output = {
        "seed": 11,
        "mean_utterance_wer_when_action_correct": float(wer[correct].mean()),
        "mean_utterance_wer_when_action_wrong": float(wer[~correct].mean()),
        "oracle_correct_asr_wrong_count": int(((oracle_pred == y) & (pred != y)).sum()),
        "oracle_wrong_asr_correct_count": int(((oracle_pred != y) & (pred == y)).sum()),
        "lowest_recall_actions_at_least_20_test_examples": [
            {"action": actions[i], "n": int(counts[i]), "recall": float(recalls[i])}
            for i in sorted((j for j in range(len(actions)) if counts[j] >= 20),
                            key=lambda j: recalls[j])[:8]],
        "most_frequent_confusions": [
            {"true": a, "predicted": b, "count": n}
            for (a, b), n in wrong.most_common(8)],
    }
    (RESULTS / "slurp_error_analysis.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
