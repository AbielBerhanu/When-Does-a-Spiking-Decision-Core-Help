"""Resumable French ASR for the fixed development and held-out recordings."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from prepare_speech_massive import files


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
OUT = CACHE / "speech_massive_fr_asr_tiny.jsonl"


def main():
    manifest = json.loads((CACHE / "speech_massive_fr_manifest.json").read_text())
    with np.load(CACHE / "speech_massive_fr_whisper_tiny.npz") as z:
        split = z["split"]
    rows = manifest["rows"]
    if len(rows) != len(split):
        raise ValueError("Manifest/split mismatch")
    wanted = {r["id"]: int(code) for r, code in zip(rows, split) if code in (1, 2)}
    if len(wanted) != 3180:
        raise ValueError(f"Expected 3,180 fixed evaluation recordings, got {len(wanted)}")
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in OUT.read_text().splitlines()}
        if not done <= wanted.keys():
            raise ValueError("Existing ASR output contains an unknown identifier")
    model = WhisperModel("tiny", device="cpu", compute_type="int8",
                         download_root=str(ROOT / "data/asr_models"), local_files_only=True,
                         cpu_threads=4)
    completed = len(done)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as stream:
        for source_split in ("train", "validation"):
            for path in files(source_split):
                parquet = pq.ParquetFile(path)
                for group in range(parquet.num_row_groups):
                    meta = parquet.read_row_group(group, columns=["id", "utt", "is_validated"]).to_pylist()
                    selected = [(j, row) for j, row in enumerate(meta)
                                if row["is_validated"] and row["id"] in wanted and row["id"] not in done]
                    if not selected:
                        continue
                    audio = parquet.read_row_group(group, columns=["audio"])["audio"].to_pylist()
                    for j, row in selected:
                        wave, sr = sf.read(io.BytesIO(audio[j]["bytes"]), dtype="float32")
                        if wave.ndim == 2:
                            wave = wave.mean(axis=1)
                        if sr != 16000:
                            common = math.gcd(sr, 16000)
                            wave = resample_poly(wave, 16000 // common, sr // common).astype(np.float32)
                        segments, _ = model.transcribe(wave, language="fr", beam_size=1,
                                                       condition_on_previous_text=False,
                                                       vad_filter=False)
                        hypothesis = " ".join(s.text.strip() for s in segments).strip()
                        stream.write(json.dumps({"id": row["id"], "split_code": wanted[row["id"]],
                                                 "reference": row["utt"],
                                                 "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
                        done.add(row["id"])
                        completed += 1
                        if completed % 100 == 0:
                            stream.flush()
                            print(f"French ASR {completed}/{len(wanted)}", flush=True)
    if done != wanted.keys():
        raise ValueError(f"Missing {len(wanted) - len(done)} French recordings")
    print(f"French ASR complete: {completed}/{len(wanted)}", flush=True)


if __name__ == "__main__":
    main()
