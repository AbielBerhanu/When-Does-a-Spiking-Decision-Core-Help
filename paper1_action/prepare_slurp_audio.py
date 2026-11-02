"""Prepare one real SLURP recording per utterance for paired action decisions."""
from __future__ import annotations

import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_easycall import N_FRAMES, N_MELS, acoustic_features_array


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/slurp_real.tar.gz"
ANNOTATIONS = ROOT / "data/slurp_repo/dataset/slurp"
OUT = ROOT / "paper1_action/cache"
SELECTED_AUDIO = ROOT / "data/slurp_selected"


def metadata():
    rows = []
    for split, filename in (("train", "train.jsonl"), ("valid", "devel.jsonl"), ("test", "test.jsonl")):
        for line in (ANNOTATIONS / filename).read_text().splitlines():
            record = json.loads(line)
            if not record["recordings"]:
                raise ValueError(f"No audio for {record['slurp_id']}")
            rows.append({"split": split, "slurp_id": record["slurp_id"],
                         "sentence": record["sentence"],
                         "audio_file": record["recordings"][0]["file"],
                         "action": f"{record['scenario']}_{record['action']}"})
    return rows


def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    rows = metadata()
    wanted = defaultdict(list)
    for index, row in enumerate(rows):
        wanted[row["audio_file"]].append(index)
    actions = sorted({r["action"] for r in rows if r["split"] == "train"})
    if len(actions) != 60 or {r["action"] for r in rows} != set(actions):
        raise ValueError("Unexpected SLURP action catalog")
    action_to_id = {a: i for i, a in enumerate(actions)}
    x = np.empty((len(rows), N_FRAMES, N_MELS), dtype=np.float32)
    seen = set()
    SELECTED_AUDIO.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in wanted:
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing stream: {name}")
            raw = stream.read()
            if any(rows[i]["split"] != "train" for i in wanted[name]):
                (SELECTED_AUDIO / name).write_bytes(raw)
            wave, sr = sf.read(io.BytesIO(raw), dtype="float32")
            feat, duration = acoustic_features_array(wave, sr)
            for i in wanted[name]:
                x[i] = feat
                rows[i]["duration_s"] = duration
            seen.add(name)
            if len(seen) % 2000 == 0:
                print(f"selected recordings {len(seen)}/{len(wanted)}", flush=True)
    missing = set(wanted) - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} referenced recordings; examples: {sorted(missing)[:10]}")
    train = np.array([r["split"] == "train" for r in rows])
    mean = x[train].mean(axis=(0, 1))
    std = x[train].std(axis=(0, 1)).clip(min=0.05)
    x = ((x - mean) / std).clip(-5, 5).astype(np.float32)
    y = np.array([action_to_id[r["action"]] for r in rows], dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "slurp_audio.npz", x=x, y=y, mean=mean, std=std)
    (OUT / "slurp_audio_manifest.json").write_text(json.dumps({"actions": actions, "rows": rows}, indent=2))
    print("prepared",len(rows),Counter(r["split"] for r in rows),"unique audio",len(seen),flush=True)


if __name__ == "__main__":
    main()
