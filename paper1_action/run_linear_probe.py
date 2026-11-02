"""Strong nonrecurrent control for the frozen SLURP speech representation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from run_slurp_audio import split_arrays
from run_text import calibration_temperature, metrics, selective_threshold, softmax


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper1_action/results"


def summarize(x):
    return np.concatenate((x.mean(axis=1), x.max(axis=1)), axis=1)


def main():
    splits, actions, encoder_seconds = split_arrays("whisper")
    train_x, train_y = summarize(splits["train"][0]), splits["train"][1]
    valid_x, valid_y = summarize(splits["valid"][0]), splits["valid"][1]
    test_x, test_y, test_rows = splits["test"]
    test_x = summarize(test_x)
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=2000, random_state=11)
        model.fit(train_x, train_y)
        scores = model.decision_function(valid_x)
        temperature = calibration_temperature(scores, valid_y)
        nll = metrics(scores, valid_y, temperature, 1.01)["nll"]
        if best is None or nll < best[0]:
            best = (nll, c, model, temperature)
    _, c, model, temperature = best
    valid_scores = model.decision_function(valid_x)
    threshold = selective_threshold(softmax(valid_scores / temperature), valid_y)
    test_scores = model.decision_function(test_x)
    result = {
        "benchmark": "slurp", "input": "frozen_whisper_audio", "kind": "linear_probe",
        "C": c, "optimization_iterations": int(np.max(model.n_iter_)),
        "n_parameters_decision_model": int(model.coef_.size + model.intercept_.size),
        "mean_encoder_seconds_during_cache_generation_unisolated": encoder_seconds,
        "temperature": temperature, "threshold_10pct_dev_error": threshold,
        "valid": metrics(valid_scores, valid_y, temperature, threshold),
        "test": metrics(test_scores, test_y, temperature, threshold),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slurp_audio_whisper_linear_probe.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / "slurp_audio_whisper_linear_probe_predictions.npz",
                        scores=test_scores, y=test_y,
                        slurp_id=np.array([r["slurp_id"] for r in test_rows]))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
