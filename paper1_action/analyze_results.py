"""Summarize completed action-decision experiments and paired uncertainty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper1_action/results"
SEEDS = (11, 13, 17, 19, 23)
KINDS = ("ann", "leaky_ann", "snn", "gru")


def summary(name):
    rows = [json.loads((RESULTS / f"{name}_{kind}_{seed}.json").read_text())
            for kind in KINDS for seed in SEEDS]
    grouped = {}
    for kind in KINDS:
        part = [r for r in rows if r["kind"] == kind]
        grouped[kind] = {}
        for key in ("accuracy", "macro_f1", "nll", "brier", "ece_15",
                    "selective_coverage", "selective_error"):
            values = [r["test"][key] for r in part]
            numeric = [v for v in values if v is not None]
            grouped[kind][key] = {
                "mean": float(np.mean(numeric)) if numeric else None,
                "sample_sd": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else None,
                "seeds": values}
        grouped[kind]["parameters"] = part[0].get("n_parameters", part[0].get("n_parameters_decision_model"))
        grouped[kind]["mean_hidden_spike_rate"] = (
            float(np.mean([r["mean_hidden_spike_rate"] for r in part])) if kind == "snn" else None)
    scores = {}
    y = None
    for kind in KINDS:
        preds = []
        for seed in SEEDS:
            with np.load(RESULTS / f"{name}_{kind}_{seed}_predictions.npz") as z:
                if y is not None and not np.array_equal(y, z["y"]):
                    raise ValueError(f"Unpaired test labels in {name}")
                y = z["y"]
                preds.append(z["scores"].argmax(axis=1))
        scores[kind] = np.stack(preds)
    # Resample utterances as the independent unit, keeping seeds and both
    # architectures paired on every sampled item. This interval conditions on
    # the five trained seeds; their sample SD is reported separately above.
    rng = np.random.default_rng(2026)
    for comparator in ("ann", "leaky_ann", "gru"):
        delta = (scores["snn"] == y).astype(np.int8) - (scores[comparator] == y).astype(np.int8)
        per_item_delta = delta.mean(axis=0)
        bootstrap = np.array([per_item_delta[rng.integers(len(y), size=len(y))].mean()
                              for _ in range(5000)])
        hierarchical_rng = np.random.default_rng(542)
        hierarchical = np.empty(5000)
        for i in range(5000):
            seed_ids = hierarchical_rng.integers(len(SEEDS), size=len(SEEDS))
            item_ids = hierarchical_rng.integers(len(y), size=len(y))
            hierarchical[i] = delta[np.ix_(seed_ids, item_ids)].mean()
        grouped[f"paired_accuracy_difference_snn_minus_{comparator}"] = {
            "mean": float(per_item_delta.mean()),
            "bootstrap_95ci": [float(v) for v in np.percentile(bootstrap, (2.5, 97.5))],
            "hierarchical_95ci": [float(v) for v in np.percentile(hierarchical, (2.5, 97.5))],
            "unit": f"{'Speech-MASSIVE' if name.startswith('speech_massive') else 'SLURP'} utterance; five fixed seeds averaged",
        }
    return grouped


def main():
    output = {}
    for name in ("slurp_oracle", "slurp_asr", "slurp_asr_trained",
                 "slurp_audio_whisper_plain",
                 "speech_massive_fr_audio", "speech_massive_fr_asr"):
        if all((RESULTS / f"{name}_{kind}_{seed}.json").exists()
               and (RESULTS / f"{name}_{kind}_{seed}_predictions.npz").exists()
               for kind in KINDS for seed in SEEDS):
            output[name] = summary(name)
    (RESULTS / "action_decision_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
