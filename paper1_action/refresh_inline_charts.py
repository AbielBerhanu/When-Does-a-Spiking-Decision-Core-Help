"""Refresh standalone-editor TikZ fallbacks from saved five-seed results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_text import softmax


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper1_action/paper.tex"
RESULTS = ROOT / "paper1_action/results"
SEEDS = (11, 13, 17, 19, 23)
KINDS = ("ann", "leaky_ann", "snn", "gru")
PATHS = (("slurp_audio_whisper_plain", "Audio"),
         ("slurp_asr", "ASR"), ("slurp_oracle", "Oracle"))
GRID = np.r_[1, 2, 5, np.arange(10, 101, 5)]
STYLE = {
    "ann": ("modelorange", "dashed", "Tanh RNN"),
    "leaky_ann": ("modelpurple", "dotted", "Leaky tanh"),
    "snn": ("modelblue", "solid", "Spiking LIF"),
    "gru": ("modelgreen", "dashdotted", "GRU"),
}


def replace_tikz(source: str, anchor: str, replacement: str) -> str:
    anchor_index = source.index(anchor)
    start = source.index(r"\begin{tikzpicture}", anchor_index)
    end = source.index(r"\end{tikzpicture}", start) + len(r"\end{tikzpicture}")
    return source[:start] + replacement + source[end:]


def accuracy_tikz(summary: dict) -> str:
    lines = [r"\begin{tikzpicture}",
             r"\begin{axis}[ybar,bar width=9pt,width=0.94\textwidth,height=5.2cm,",
             r"  ymin=0,ymax=88,ytick={0,20,40,60,80},ylabel={Action accuracy (\%)},",
             r"  symbolic x coords={Audio,ASR,Oracle},xtick=data,",
             r"  xticklabels={Frozen audio states,Recognized words,Oracle transcript},",
             r"  enlarge x limits=0.2,axis background/.style={fill=gray!3},",
             r"  ymajorgrids,grid style={gray!20},",
             r"  tick label style={font=\scriptsize},label style={font=\scriptsize},",
             r"  legend style={font=\scriptsize,draw=none,fill=none,at={(0.01,0.99)},anchor=north west,legend columns=4}]" ]
    for kind in KINDS:
        color = STYLE[kind][0]
        coords = []
        for path, short in PATHS:
            mean = 100 * summary[path][kind]["accuracy"]["mean"]
            sd = 100 * summary[path][kind]["accuracy"]["sample_sd"]
            coords.append(f"({short},{mean:.2f}) +- (0,{sd:.2f})")
        lines.append(f"\\addplot+[fill={color},draw={color}!75!black,error bars/.cd,y dir=both,y explicit]")
        lines.append("  coordinates {" + " ".join(coords) + "};")
    lines.extend([r"\legend{Tanh RNN,Leaky tanh,Spiking LIF,GRU}",
                  r"\end{axis}", r"\end{tikzpicture}"])
    return "\n".join(lines)


def risk_curve(path: str, kind: str) -> np.ndarray:
    samples = []
    stems = ([f"{path}_bow"] if kind == "bow" else
             [f"{path}_{kind}_{seed}" for seed in SEEDS])
    for stem in stems:
        row = json.loads((RESULTS / f"{stem}.json").read_text())
        with np.load(RESULTS / f"{stem}_predictions.npz") as z:
            probs = softmax(z["scores"] / row["temperature"])
            order = np.argsort(-probs.max(axis=1))
            error = (probs.argmax(axis=1) != z["y"]).astype(float)[order]
            risk = np.cumsum(error) / np.arange(1, len(error) + 1)
            coverage = np.arange(1, len(error) + 1) / len(error)
            samples.append(np.interp(GRID / 100, coverage, risk) * 100)
    return np.mean(samples, axis=0)


def risk_tikz() -> str:
    lines = [r"\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=3 by 1,horizontal sep=0.6cm},",
             r"  width=0.29\textwidth,height=4.4cm,xmin=0,xmax=100,ymin=0,ymax=65,",
             r"  xtick={0,50,100},ytick={0,20,40,60},xlabel={Executed (\%)},",
             r"  grid=major,grid style={gray!18},axis background/.style={fill=gray!3},",
             r"  tick label style={font=\tiny},label style={font=\scriptsize},",
             r"  title style={font=\scriptsize}]" ]
    titles = ("Direct acoustic state", "Recognized words", "Oracle transcript")
    for index, ((path, _), title) in enumerate(zip(PATHS, titles)):
        options = f"title={{{title}}}"
        if index == 0:
            options += (r",ylabel={Action error (\%)},"
                        r"legend style={font=\tiny,draw=none,fill=none,at={(0.03,0.97)},anchor=north west}")
        lines.append(f"\\nextgroupplot[{options}]")
        for kind in KINDS:
            color, linestyle, _ = STYLE[kind]
            coords = " ".join(f"({int(x)},{y:.2f})" for x, y in zip(GRID, risk_curve(path, kind)))
            lines.append(f"\\addplot[{color},very thick,{linestyle}] coordinates {{{coords}}};")
        if index == 0:
            lines.append(r"\addplot[modelrose,very thick,dashed] coordinates {};")
            lines.append(r"\legend{Tanh RNN,Leaky tanh,Spiking LIF,GRU,TF--IDF}")
        else:
            coords = " ".join(f"({int(x)},{y:.2f})" for x, y in zip(GRID, risk_curve(path, "bow")))
            lines.append(f"\\addplot[modelrose,very thick,dashed] coordinates {{{coords}}};")
        lines.append(r"\addplot[gray,densely dotted] coordinates {(0,10) (100,10)};")
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}"])
    return "\n".join(lines)


def main() -> None:
    summary = json.loads((RESULTS / "action_decision_summary.json").read_text())
    source = PAPER.read_text()
    source = replace_tikz(source, r"\IfFileExists{accuracy_comparison.png}", accuracy_tikz(summary))
    source = replace_tikz(source, r"\IfFileExists{risk_coverage.pdf}", risk_tikz())
    PAPER.write_text(source)
    print("Updated editor fallbacks in", PAPER)


if __name__ == "__main__":
    main()
