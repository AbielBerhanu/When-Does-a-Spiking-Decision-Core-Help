"""Publication figure for five-seed SLURP action accuracy by input path."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper1_action/results/action_decision_summary.json"
OUT = ROOT / "paper1_action/accuracy_comparison.png"

PATHS = ("slurp_audio_whisper_plain", "slurp_asr", "slurp_oracle")
LABELS = ("Frozen audio\nstates", "Recognized\nwords", "Oracle\ntranscript")
MODELS = (
    ("ann", "Tanh RNN", "#C66A28", "///"),
    ("leaky_ann", "Leaky tanh", "#7957A8", ".."),
    ("snn", "Spiking LIF", "#2468A6", ""),
    ("gru", "GRU", "#16826E", "\\\\"),
)


def main() -> None:
    summary = json.loads(RESULTS.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(7.15, 2.42))
    ax.set_facecolor("#FAFBFC")
    ax.axvspan(1.5, 2.5, color="#EEF2F6", alpha=0.75, zorder=0)
    x = np.arange(len(PATHS))
    width = 0.19
    for index, (kind, label, color, hatch) in enumerate(MODELS):
        centers = x + (index - 1.5) * width
        means = np.array([summary[path][kind]["accuracy"]["mean"] * 100
                          for path in PATHS])
        sd = np.array([summary[path][kind]["accuracy"]["sample_sd"] * 100
                       for path in PATHS])
        bars = ax.bar(centers, means, width=width * 0.91, color=color,
                      edgecolor="#34495A", linewidth=0.45, hatch=hatch,
                      label=label, zorder=3)
        ax.errorbar(centers, means, yerr=sd, fmt="none", ecolor="#25394B",
                    elinewidth=0.8, capsize=2.1, capthick=0.8, zorder=4)
        for bar, value, err in zip(bars, means, sd):
            ax.text(bar.get_x() + bar.get_width()/2, value + err + 1.05,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=7.2,
                    color="#273C4C")
    ax.set(xlim=(-0.52, 2.52), ylim=(0, 88), ylabel="Action accuracy (%)",
           xticks=x, xticklabels=LABELS, yticks=(0, 20, 40, 60, 80))
    ax.grid(axis="y", color="#DDE4EB", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.axvline(1.5, color="#AFBAC6", linewidth=0.7, linestyle=(0, (3, 3)))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#9AA7B5")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=4,
              frameon=False, fontsize=8, columnspacing=1.8,
              handlelength=1.5)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.84)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
