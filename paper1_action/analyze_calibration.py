"""Supplementary calibration and selective-risk summaries from held-out logits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_text import ece, softmax


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper1_action/results"
SEEDS = (11, 13, 17, 19, 23)
PATHS = ("slurp_audio_whisper_plain", "slurp_asr", "slurp_oracle",
         "speech_massive_fr_audio")
KINDS = ("ann", "leaky_ann", "snn", "gru")


def risk_coverage_area(probs, y):
    order = np.argsort(-probs.max(axis=1))
    error = (probs.argmax(axis=1) != y).astype(float)[order]
    risk = np.cumsum(error) / np.arange(1, len(y) + 1)
    return float(risk.mean())


def reliability(probs, y, bins=15):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    count = np.zeros(bins, dtype=int)
    mean_confidence = np.full(bins, np.nan)
    mean_accuracy = np.full(bins, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lo) & (confidence < hi if i < bins - 1 else confidence <= hi)
        count[i] = mask.sum()
        if count[i]:
            mean_confidence[i] = confidence[mask].mean()
            mean_accuracy[i] = correct[mask].mean()
    return count, mean_confidence, mean_accuracy


def main():
    output = {}
    diagnostics = {"before": [], "after": [], "y": []}
    for path in PATHS:
        output[path] = {}
        for kind in KINDS:
            pre, post, aurc = [], [], []
            for seed in SEEDS:
                row = json.loads((RESULTS / f"{path}_{kind}_{seed}.json").read_text())
                with np.load(RESULTS / f"{path}_{kind}_{seed}_predictions.npz") as z:
                    y = z["y"]
                    before = softmax(z["scores"])
                    after = softmax(z["scores"] / row["temperature"])
                pre.append(ece(before, y))
                post.append(ece(after, y))
                aurc.append(risk_coverage_area(after, y))
                if path == "slurp_asr" and kind == "snn":
                    diagnostics["before"].append(before)
                    diagnostics["after"].append(after)
                    diagnostics["y"].append(y)
            output[path][kind] = {"ece_before_mean": float(np.mean(pre)),
                                  "ece_after_mean": float(np.mean(post)),
                                  "aurc_mean": float(np.mean(aurc)),
                                  "aurc_sample_sd": float(np.std(aurc, ddof=1))}
    bow = json.loads((RESULTS / "slurp_asr_bow.json").read_text())
    with np.load(RESULTS / "slurp_asr_bow_predictions.npz") as z:
        bow_y, bow_scores = z["y"], z["scores"]
    output["slurp_asr"]["bow"] = {
        "ece_before_mean": ece(softmax(bow_scores), bow_y),
        "ece_after_mean": ece(softmax(bow_scores / bow["temperature"]), bow_y),
        "aurc_mean": risk_coverage_area(softmax(bow_scores / bow["temperature"]), bow_y),
        "aurc_sample_sd": None}
    (RESULTS / "calibration_risk_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)

    y = np.concatenate(diagnostics["y"])
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.25), sharex=True, sharey=True)
    for ax, (key, title, color) in zip(axes, (("before", "Before calibration", "#C66A28"),
                                               ("after", "After calibration", "#2468A6"))):
        probs = np.concatenate(diagnostics[key])
        count, conf, acc = reliability(probs, y)
        ax.set_facecolor("#FAFBFC")
        ax.plot([0, 1], [0, 1], color="#7A8793", linestyle="--", linewidth=1)
        mask = count > 0
        ax.scatter(conf[mask], acc[mask], s=15 + 110 * count[mask] / count.max(),
                   color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=8)
        ax.grid(color="#E0E7ED", linewidth=0.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, .25, .5, .75, 1),
               yticks=(0, .25, .5, .75, 1))
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Observed accuracy", fontsize=8)
    fig.supxlabel("Predicted confidence", y=0.02, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(ROOT / "paper1_action/reliability_asr_snn.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper1_action/reliability_asr_snn.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
