"""Five-seed test risk--coverage curves for calibrated bounded decisions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_text import softmax


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper1_action/results"
SEEDS = (11, 13, 17, 19, 23)
PATHS = (
    ("slurp_audio_whisper_plain", "Direct acoustic state"),
    ("slurp_asr", "Recognized words"),
    ("slurp_oracle", "Oracle transcript"),
)
STYLES = {
    "ann": ("#C66A28", "Tanh RNN", "--"),
    "leaky_ann": ("#7957A8", "Leaky tanh", ":"),
    "snn": ("#2468A6", "Spiking LIF", "-"),
    "gru": ("#16826E", "GRU", "-."),
}


def curve(scores, y, temperature, grid):
    probs = softmax(scores / temperature)
    confidence = probs.max(axis=1)
    error = (probs.argmax(axis=1) != y).astype(np.float32)
    order = np.argsort(-confidence)
    risk = np.cumsum(error[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    return np.interp(grid, coverage, risk)


def main():
    grid = np.linspace(0.01, 1.0, 150)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.42), sharey=True)
    for ax, (prefix, title) in zip(axes, PATHS):
        ax.set_facecolor("#FAFBFC")
        for kind, (color, label, linestyle) in STYLES.items():
            curves = []
            for seed in SEEDS:
                result = json.loads((RESULTS / f"{prefix}_{kind}_{seed}.json").read_text())
                with np.load(RESULTS / f"{prefix}_{kind}_{seed}_predictions.npz") as z:
                    curves.append(curve(z["scores"], z["y"], result["temperature"], grid))
            curves = np.stack(curves)
            ax.plot(grid * 100, curves.mean(axis=0) * 100, label=label,
                    color=color, linewidth=1.65, linestyle=linestyle)
            ax.fill_between(grid * 100, curves.min(axis=0) * 100,
                            curves.max(axis=0) * 100, color=color, alpha=0.11,
                            linewidth=0)
        if prefix in ("slurp_asr", "slurp_oracle"):
            bow = json.loads((RESULTS / f"{prefix}_bow.json").read_text())
            with np.load(RESULTS / f"{prefix}_bow_predictions.npz") as z:
                bow_curve = curve(z["scores"], z["y"], bow["temperature"], grid)
            ax.plot(grid * 100, bow_curve * 100, label="TF-IDF",
                    color="#B23C74", linewidth=1.8, linestyle=(0, (5, 2)))
        ax.axhline(10, color="#707B87", linestyle=(0, (2, 2)), linewidth=0.8)
        ax.set(xlim=(0, 100), ylim=(0, 65), title=title,
               xticks=(0, 50, 100), yticks=(0, 20, 40, 60))
        ax.grid(True, color="#DDE4EB", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#9AA7B5")
    axes[0].set_ylabel("Error among executed actions (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.015), columnspacing=2.2,
               handlelength=2.8, fontsize=7.5)
    fig.supxlabel("Executed decisions (%)", y=0.045, fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.27, top=0.79, wspace=0.13)
    out = ROOT / "paper1_action"
    fig.savefig(out / "risk_coverage.pdf", bbox_inches="tight")
    fig.savefig(out / "risk_coverage.png", dpi=300, bbox_inches="tight")
    print(out / "risk_coverage.pdf")


if __name__ == "__main__":
    main()
