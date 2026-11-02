"""Build a source-and-results release without raw corpora or model checkpoints."""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper1_action"
PACKAGE = ROOT / "AIxDke2026_stage"
ACTION = PACKAGE / "paper1_action"


def copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    if PACKAGE.exists():
        raise FileExistsError(f"Release directory already exists: {PACKAGE}")
    ACTION.mkdir(parents=True)
    for source in SOURCE.glob("*.py"):
        copy(source, ACTION / source.name)
    for extension in ("*.md", "*.txt"):
        for source in SOURCE.glob(extension):
            copy(source, ACTION / source.name)
    for name in ("paper.tex", "paper.pdf", "requirements.txt",
                 "accuracy_comparison.pdf", "accuracy_comparison.png",
                 "risk_coverage.pdf", "risk_coverage.png",
                 "reliability_asr_snn.pdf", "reliability_asr_snn.png"):
        source = SOURCE / name
        if source.exists():
            copy(source, ACTION / name)
    for source in (SOURCE / "tables").glob("*.csv"):
        copy(source, ACTION / "tables" / source.name)
    summaries = {"action_decision_summary.json", "calibration_risk_summary.json",
                 "decision_cost.json", "easycall_summary.json",
                 "slurp_pipeline_cpu_timing.json", "slurp_asr_bow.json",
                 "slurp_oracle_bow.json", "speech_massive_fr_asr_bow.json"}
    prefixes = ("slurp_oracle_", "slurp_asr_", "slurp_asr_trained_",
                "slurp_audio_whisper_plain_", "slurp_audio_whisper_linear_probe",
                "speech_massive_fr_audio_",
                "speech_massive_fr_asr_", "easycall_audio_conv_")
    for source in (SOURCE / "results").iterdir():
        if source.suffix not in (".json", ".npz"):
            continue
        if source.name not in summaries and not source.name.startswith(prefixes):
            continue
        copy(source, ACTION / "results" / source.name)
    (PACKAGE / ".gitignore").write_text(
        ".venv/\ndata/\npaper1_action/cache/\n"
        "paper1_action/results/*.pt\npaper1_action/results/*.joblib\n"
        "__pycache__/\n*.aux\n*.log\n*.synctex.gz\n.DS_Store\n")
    (PACKAGE / "README.md").write_text(
        "# When Does a Spiking Decision Core Help? A Controlled Study of Spoken Action Selection\n\n"
        "This repository contains the six-page IEEE conference manuscript, "
        "reproducible simulation code, held-out prediction scores, and tables. "
        "The manuscript is [`paper1_action/paper.pdf`](paper1_action/paper.pdf); "
        "its LaTeX source is [`paper1_action/paper.tex`](paper1_action/paper.tex).\n\n"
        "## Main finding\n\n"
        "The spiking core is competitive for recognized words, but its small "
        "French lead over leak-matched tanh is inconclusive across seeds. "
        "A GRU is stronger on acoustic states; TF--IDF logistic regression "
        "is stronger on recognized words and on SLURP selective risk and "
        "simulated action cost. No hardware energy or clinical claim is made.\n\n"
        "## Contents\n\n"
        "- `paper1_action/*.py`: preparation, training, evaluation, analysis, "
        "validation, plotting, and table-export code.\n"
        "- `paper1_action/tables/*.csv`: machine-readable result tables.\n"
        "- `paper1_action/results/*.json`: per-run metrics and summaries.\n"
        "- `paper1_action/results/*.npz`: held-out logits, labels, and "
        "paired item IDs for independent analysis; no waveform files.\n"
        "- `paper1_action/fresh_benchmark_protocol.md`: the fixed French "
        "extension protocol.\n"
        "- `paper1_action/reproduce.md`: dataset sources, commands, and "
        "interpretation limits.\n\n"
        "## Rebuild\n\n"
        "From the repository root, install "
        "`paper1_action/requirements.txt` in a Python environment and follow "
        "`paper1_action/reproduce.md`. Raw SLURP, Speech-MASSIVE, and EasyCall "
        "recordings and trained checkpoints are intentionally omitted. "
        "The included predictions and summaries support result audits without "
        "redistributing those corpora. To build the PDF, run "
        "`tectonic -C -o paper1_action paper1_action/paper.tex` "
        "with Tectonic installed.\n")
    print(PACKAGE)


if __name__ == "__main__":
    main()
