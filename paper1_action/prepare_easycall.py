"""Build a speaker-disjoint, action-labeled EasyCall audio benchmark.

The 30 no-action prompts follow Table 4.2 of the EasyCall authors' thesis.
The archive spells that table's ``Salve`` entry as ``Salva``. No speech content
or held-out prediction is used to define the mapping.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/easycall_orig/EasyCall"
OUT = ROOT / "paper1_action/cache"
N_FRAMES, N_MELS, SR = 48, 40, 16000

NO_ACTION = set("""
rubrica|preferiti|sezione|vivavoce|chiamata|richiama|tastiera|terminare|apri|chiudi|
muto|zelo|nave|mali|tra|scesi|bue|sotto|sopra|salva|muovi|raggiungi|cella|top|
cancella contatto|vai alla pagina principale|fai una telefonata|chiudi applicazione|
nuovo contatto|chiama ultimo numero
""".replace("\n", "").split("|"))

COMMAND_TO_ACTION = {
    "aggiungi": "add",
    "aggiungi ai preferiti": "add_favorite",
    "apri rubrica": "open_contacts",
    "attiva vivavoce": "speaker_on",
    "cancella": "delete",
    "cancella tutto": "delete_all",
    "chiama": "call",
    "chiama emergenza": "emergency_call",
    "chiudi rubrica": "close_contacts",
    "cinque": "digit_5",
    "deseleziona": "deselect",
    "disattiva vivavoce": "speaker_off",
    "due": "digit_2",
    "esci da rubrica": "close_contacts",
    "indietro": "back",
    "no": "no",
    "nove": "digit_9",
    "otto": "digit_8",
    "quattro": "digit_4",
    "rimuovi": "remove",
    "sali": "go_up",
    "scendi": "go_down",
    "scorri verso il basso": "scroll_down",
    "scorri verso l alto": "scroll_up",
    "sei": "digit_6",
    "seleziona": "select",
    "sette": "digit_7",
    "si": "yes",
    "stop": "stop",
    "termina chiamata": "end_call",
    "tre": "digit_3",
    "uno": "digit_1",
    "vai alla tastiera": "open_keypad",
    "vai nei preferiti": "open_favorites",
    "vai nel registro chiamate": "open_call_history",
    "vai nella rubrica": "open_contacts",
    "zero": "digit_0",
}


def prompt_from_path(path: Path) -> str:
    stem = path.stem.split("_", 2)[-1].replace("_", " ").replace("’", " ")
    return " ".join(re.sub(r"\s+[123]$", "", stem.lower()).split())


def make_manifest() -> list[dict]:
    files = sorted(SOURCE.rglob("*.wav"))
    if len(files) != 21386:
        raise ValueError(f"Expected 21,386 original recordings; found {len(files)}")
    prompts = {prompt_from_path(p) for p in files}
    if len(prompts) != 67 or prompts != NO_ACTION | COMMAND_TO_ACTION.keys():
        raise ValueError(f"Prompt ontology mismatch: {sorted(prompts ^ (NO_ACTION | COMMAND_TO_ACTION.keys()))}")
    speakers = sorted({p.relative_to(SOURCE).parts[0] for p in files})
    if len(speakers) != 55:
        raise ValueError(f"Expected 55 speakers; found {len(speakers)}")
    rng = np.random.default_rng(2026)
    split_by_speaker = {}
    for group in ("dysarthric", "control"):
        group_speakers = [s for s in speakers if ("c" not in s) == (group == "dysarthric")]
        rng.shuffle(group_speakers)
        n_test = round(0.2 * len(group_speakers))
        n_valid = round(0.2 * len(group_speakers))
        for i, speaker in enumerate(group_speakers):
            split_by_speaker[speaker] = ("test" if i < n_test else
                                         "valid" if i < n_test + n_valid else "train")
    rows = []
    for path in files:
        speaker = path.relative_to(SOURCE).parts[0]
        prompt = prompt_from_path(path)
        rows.append({"path": str(path.relative_to(ROOT)), "speaker": speaker,
                     "speaker_group": "control" if "c" in speaker else "dysarthric",
                     "split": split_by_speaker[speaker], "prompt": prompt,
                     "action": "no_action" if prompt in NO_ACTION else COMMAND_TO_ACTION[prompt]})
    return rows


def mel_filterbank():
    n_fft = 512
    hz = np.arange(n_fft // 2 + 1) * SR / n_fft
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    points = inv(np.linspace(mel(70), mel(7600), N_MELS + 2))
    bank = np.zeros((n_fft // 2 + 1, N_MELS), np.float32)
    for j in range(N_MELS):
        a, b, c = points[j:j + 3]
        bank[:, j] = np.maximum(0, np.minimum((hz - a) / (b - a), (c - hz) / (c - b)))
        bank[:, j] /= max(bank[:, j].sum(), 1e-9)
    return bank


FB = mel_filterbank()
WINDOW = np.hamming(400).astype(np.float32)


def acoustic_features(path: Path) -> tuple[np.ndarray, float]:
    wave, sr = sf.read(path, dtype="float32")
    return acoustic_features_array(wave, sr)


def acoustic_features_array(wave: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    duration = len(wave) / sr
    if sr != SR:
        common = math.gcd(sr, SR)
        wave = resample_poly(wave, SR // common, sr // common).astype(np.float32)
    wave = np.pad(wave, (0, max(0, 400 - len(wave))))
    frames = np.lib.stride_tricks.sliding_window_view(wave, 400)[::160]
    power = np.abs(np.fft.rfft(frames * WINDOW, n=512, axis=1)) ** 2
    mel = np.log1p(10.0 * (power @ FB)).astype(np.float32)
    if len(mel) >= N_FRAMES:
        edges = np.linspace(0, len(mel), N_FRAMES + 1).astype(int)
        pooled = np.stack([mel[edges[i]:edges[i + 1]].mean(axis=0) for i in range(N_FRAMES)])
    else:
        old, new = np.linspace(0, 1, len(mel)), np.linspace(0, 1, N_FRAMES)
        pooled = np.stack([np.interp(new, old, mel[:, j]) for j in range(N_MELS)], axis=1)
    return pooled.astype(np.float32), duration


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_manifest()
    x = np.empty((len(rows), N_FRAMES, N_MELS), dtype=np.float32)
    for i, row in enumerate(rows):
        x[i], row["duration_s"] = acoustic_features(ROOT / row["path"])
        if (i + 1) % 2000 == 0:
            print(f"features {i + 1}/{len(rows)}", flush=True)
    train = np.array([r["split"] == "train" for r in rows])
    mean = x[train].mean(axis=(0, 1))
    std = x[train].std(axis=(0, 1)).clip(min=0.05)
    x = ((x - mean) / std).clip(-5, 5).astype(np.float32)
    actions = sorted({r["action"] for r in rows})
    y = np.array([actions.index(r["action"]) for r in rows], dtype=np.int64)
    np.savez_compressed(OUT / "easycall_audio.npz", x=x, y=y, mean=mean, std=std)
    (OUT / "easycall_manifest.json").write_text(json.dumps({"actions": actions, "rows": rows}, indent=2))
    print("count",len(rows),"speakers",Counter(r["split"] for r in rows),
          "actions",len(actions),"no_action",sum(r["action"] == "no_action" for r in rows),flush=True)


if __name__ == "__main__":
    main()
