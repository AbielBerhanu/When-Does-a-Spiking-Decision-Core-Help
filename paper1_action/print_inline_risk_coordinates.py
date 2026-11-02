"""Print sampled risk-coverage coordinates for the standalone LaTeX preview."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_text import softmax


ROOT = Path(__file__).resolve().parents[1] / "paper1_action/results"
PATHS = ("slurp_audio_whisper_plain", "slurp_asr", "slurp_oracle")
KINDS = ("ann", "leaky_ann", "snn", "gru")
SEEDS = (11, 13, 17, 19, 23)
GRID = np.r_[1, 2, 5, np.arange(10, 101, 5)]


def curve(scores: np.ndarray, y: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = softmax(scores / temperature)
    order = np.argsort(-probabilities.max(axis=1))
    mistakes = (probabilities.argmax(axis=1) != y).astype(float)[order]
    risk = np.cumsum(mistakes) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    return np.interp(GRID / 100, coverage, risk) * 100


for path in PATHS:
    print("%", path)
    for kind in KINDS:
        samples = []
        for seed in SEEDS:
            result = json.loads((ROOT / f"{path}_{kind}_{seed}.json").read_text())
            with np.load(ROOT / f"{path}_{kind}_{seed}_predictions.npz") as data:
                samples.append(curve(data["scores"], data["y"], result["temperature"]))
        mean = np.mean(samples, axis=0)
        print("%", kind)
        print(" ".join(f"({int(x)},{value:.2f})" for x, value in zip(GRID, mean)))
