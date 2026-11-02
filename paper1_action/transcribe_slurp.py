"""Offline ASR hypotheses for the SLURP development and test utterances.

Uses Faster Whisper tiny.en (int8 CPU), beam size one, fixed language English.
Resumable JSONL output. ASR is a separate upstream model from the decision core.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper1_action/cache/slurp_audio_manifest.json"
AUDIO = ROOT / "data/slurp_selected"
ARCHIVE = ROOT / "data/slurp_real.tar.gz"
OUT = ROOT / "paper1_action/cache/slurp_asr_tiny_en.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    args = parser.parse_args()
    rows = [r for r in json.loads(MANIFEST.read_text())["rows"]
            if (r["split"] == "train") == (args.split == "train")]
    out = (ROOT / "paper1_action/cache/slurp_asr_train_tiny_en.jsonl"
           if args.split == "train" else OUT)
    done = set()
    if out.exists():
        done = {json.loads(line)["slurp_id"] for line in out.read_text().splitlines()}
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                         download_root=str(ROOT / "data/asr_models"), local_files_only=True,
                         cpu_threads=4)
    def transcribe(row, wave, sr, stream):
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            if sr != 16000:
                wave = resample_poly(wave, 16000, sr).astype(np.float32)
            started = time.perf_counter()
            segments, _ = model.transcribe(wave, language="en", beam_size=1,
                                           condition_on_previous_text=False,
                                           vad_filter=False)
            hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
            result = {"slurp_id": row["slurp_id"], "split": row["split"],
                      "reference": row["sentence"], "hypothesis": hypothesis,
                      "asr_seconds": time.perf_counter() - started}
            stream.write(json.dumps(result) + "\n")

    with out.open("a") as stream:
        if args.split == "train":
            wanted = defaultdict(list)
            for row in rows:
                if row["slurp_id"] not in done:
                    wanted[row["audio_file"]].append(row)
            completed = len(done)
            with tarfile.open(ARCHIVE, mode="r|gz") as archive:
                for member in archive:
                    batch = wanted.get(Path(member.name).name)
                    if not batch:
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(member.name)
                    wave, sr = sf.read(io.BytesIO(extracted.read()), dtype="float32")
                    for row in batch:
                        transcribe(row, wave, sr, stream)
                        completed += 1
                    if completed % 100 < len(batch):
                        stream.flush()
                        print(f"ASR {completed}/{len(rows)}", flush=True)
            if completed != len(rows):
                raise ValueError(f"Missing {len(rows) - completed} training recordings")
        else:
            for i, row in enumerate(rows):
                if row["slurp_id"] in done:
                    continue
                wave, sr = sf.read(AUDIO / row["audio_file"], dtype="float32")
                transcribe(row, wave, sr, stream)
                if (i + 1) % 100 == 0:
                    stream.flush()
                    print(f"ASR {i + 1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
