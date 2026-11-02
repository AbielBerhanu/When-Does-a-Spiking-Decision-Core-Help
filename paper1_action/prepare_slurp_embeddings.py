"""Frozen Whisper encoder states for the SLURP audio decision path.

The encoder is upstream of the decision core. Its parameter count and runtime
must be included in full-pipeline comparisons, never attributed to the SNN.
"""
from __future__ import annotations

import io
import json
import math
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/slurp_real.tar.gz"
MANIFEST = ROOT / "paper1_action/cache/slurp_audio_manifest.json"
OUT = ROOT / "paper1_action/cache/slurp_whisper_tiny_en.npz"
STEPS, DIM = 32, 384


def pool(states):
    if len(states) >= STEPS:
        edges = np.linspace(0, len(states), STEPS + 1).astype(int)
        return np.stack([states[edges[i]:edges[i + 1]].mean(axis=0) for i in range(STEPS)])
    old, new = np.linspace(0, 1, len(states)), np.linspace(0, 1, STEPS)
    return np.stack([np.interp(new, old, states[:, j]) for j in range(DIM)], axis=1)


def main():
    manifest = json.loads(MANIFEST.read_text())
    rows, actions = manifest["rows"], manifest["actions"]
    wanted = defaultdict(list)
    for i, row in enumerate(rows):
        wanted[row["audio_file"]].append(i)
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                         download_root=str(ROOT / "data/asr_models"), local_files_only=True,
                         cpu_threads=4)
    x = np.empty((len(rows), STEPS, DIM), dtype=np.float32)
    durations = np.empty(len(rows), dtype=np.float32)
    elapsed = np.empty(len(rows), dtype=np.float32)
    seen = set()
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in wanted:
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(name)
            wave, sr = sf.read(io.BytesIO(stream.read()), dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            duration = len(wave) / sr
            if sr != 16000:
                common = math.gcd(sr, 16000)
                wave = resample_poly(wave, 16000 // common, sr // common).astype(np.float32)
            started = time.perf_counter()
            feature = model.feature_extractor(wave)
            encoded = np.asarray(model.encode(feature))[0]
            pooled = pool(encoded).astype(np.float32)
            seconds = time.perf_counter() - started
            for i in wanted[name]:
                x[i], durations[i], elapsed[i] = pooled, duration, seconds
            seen.add(name)
            if len(seen) % 2000 == 0:
                print(f"encoder {len(seen)}/{len(wanted)}", flush=True)
    if len(seen) != len(wanted):
        raise ValueError(f"Missing {len(wanted) - len(seen)} recordings")
    train = np.array([r["split"] == "train" for r in rows])
    mean = x[train].mean(axis=(0, 1))
    std = x[train].std(axis=(0, 1)).clip(min=0.05)
    x = ((x - mean) / std).clip(-5, 5).astype(np.float32)
    y = np.array([actions.index(r["action"]) for r in rows], dtype=np.int64)
    np.savez_compressed(OUT, x=x, y=y, mean=mean, std=std,
                        duration_s=durations, encoder_seconds=elapsed)
    print("prepared",len(rows),"encoder mean seconds",float(elapsed.mean()),flush=True)


if __name__ == "__main__":
    main()
