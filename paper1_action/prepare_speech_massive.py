"""Frozen multilingual Whisper features for prospective French replication."""
from __future__ import annotations

import glob
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

from prepare_slurp_embeddings import pool


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/speech_massive/main/fr-FR"
OUT = ROOT / "paper1_action/cache/speech_massive_fr_whisper_tiny.npz"
MANIFEST = ROOT / "paper1_action/cache/speech_massive_fr_manifest.json"


def files(split):
    paths = [Path(p) for p in sorted(glob.glob(str(SOURCE / f"{split}-*.parquet")))]
    if not paths:
        raise FileNotFoundError(split)
    return paths


def main():
    expected = {"train": 9, "validation": 2}
    entries = []
    for split in ("train", "validation"):
        paths = files(split)
        if len(paths) != expected[split]:
            raise ValueError(f"Expected {expected[split]} {split} shards, found {len(paths)}")
        for path in paths:
            pf = pq.ParquetFile(path)
            meta = pf.read(columns=["id", "intent_idx", "intent_str", "is_validated", "speaker_id"])
            for row in meta.to_pylist():
                entries.append({"id": row["id"], "intent_idx": int(row["intent_idx"]),
                                "intent_str": row["intent_str"],
                                "is_validated": bool(row["is_validated"]),
                                "speaker_id": row["speaker_id"], "source_split": split,
                                "source_file": path.name})
    kept = [i for i, r in enumerate(entries) if r["is_validated"]]
    ids = [entries[i]["id"] for i in kept]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Speech-MASSIVE identifiers")
    train_indices = [j for j, i in enumerate(kept) if entries[i]["source_split"] == "train"]
    train_labels = [entries[kept[j]]["intent_idx"] for j in train_indices]
    fit, dev = train_test_split(train_indices, test_size=0.1, random_state=20261101,
                                stratify=train_labels)
    split_code = np.full(len(kept), 2, dtype=np.uint8)  # 2: untouched official validation
    split_code[fit] = 0
    split_code[dev] = 1
    label = np.array([entries[i]["intent_idx"] for i in kept], dtype=np.int64)
    if sorted(set(label[split_code == 0])) != list(range(60)):
        raise ValueError("Training split does not contain all 60 intents")
    model = WhisperModel("tiny", device="cpu", compute_type="int8",
                         download_root=str(ROOT / "data/asr_models"), local_files_only=True,
                         cpu_threads=4)
    x = np.empty((len(kept), 32, 384), dtype=np.float32)
    duration = np.empty(len(kept), dtype=np.float32)
    original_to_kept = {original: j for j, original in enumerate(kept)}
    original = 0
    processed = 0
    for split in ("train", "validation"):
        for path in files(split):
            pf = pq.ParquetFile(path)
            for group in range(pf.num_row_groups):
                audio = pf.read_row_group(group, columns=["audio"])["audio"].to_pylist()
                for item in audio:
                    j = original_to_kept.get(original)
                    original += 1
                    if j is None:
                        continue
                    wave, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32")
                    if wave.ndim == 2:
                        wave = wave.mean(axis=1)
                    duration[j] = len(wave) / sr
                    if sr != 16000:
                        common = math.gcd(sr, 16000)
                        wave = resample_poly(wave, 16000 // common, sr // common).astype(np.float32)
                    states = np.asarray(model.encode(model.feature_extractor(wave)))[0]
                    x[j] = pool(states).astype(np.float32)
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"French encoder {processed}/{len(kept)}", flush=True)
    if processed != len(kept) or original != len(entries):
        raise ValueError(f"Processed {processed}/{len(kept)} kept recordings")
    mean = x[split_code == 0].mean(axis=(0, 1))
    std = x[split_code == 0].std(axis=(0, 1)).clip(min=0.05)
    x = ((x - mean) / std).clip(-5, 5).astype(np.float32)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, x=x, y=label, split=split_code,
                        mean=mean, std=std, duration_s=duration)
    MANIFEST.write_text(json.dumps({"source": "FBK-MT/Speech-MASSIVE fr-FR",
                                    "split_seed": 20261101,
                                    "split_codes": {"train": 0, "development": 1,
                                                    "heldout_official_validation": 2},
                                    "rows": [entries[i] for i in kept]}, indent=2))
    print(json.dumps({"included": len(kept), "excluded": len(entries) - len(kept),
                      "train": int((split_code == 0).sum()),
                      "development": int((split_code == 1).sum()),
                      "heldout": int((split_code == 2).sum())}), flush=True)


if __name__ == "__main__":
    main()
