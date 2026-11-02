"""Isolated CPU timing of the two SLURP speech-to-action paths.

Run only after all training/feature-generation jobs finish, so shared CPU load
does not distort timings. Waveforms are already loaded before timing. The
reported time therefore excludes file I/O and the duration of the utterance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from model import DecisionNet, GRUDecisionNet
from prepare_slurp_embeddings import pool
from run_text import MAX_TOKENS, tokenize


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
OUT = ROOT / "paper1_action/results"
AUDIO = ROOT / "data/slurp_selected"


def stats(values):
    return {"p50_ms": float(np.median(values) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "mean_ms": float(np.mean(values) * 1000)}


def timed_call(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def main():
    torch.set_num_threads(1)
    manifest = json.loads((CACHE / "slurp_audio_manifest.json").read_text())
    test = [r for r in manifest["rows"] if r["split"] == "test"]
    chosen = [test[i] for i in np.linspace(0, len(test) - 1, 100, dtype=int)]
    audio_model = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                               download_root=str(ROOT / "data/asr_models"),
                               local_files_only=True, cpu_threads=4)
    with np.load(CACHE / "slurp_whisper_tiny_en.npz") as z:
        mean, std = z["mean"], z["std"]
    data = {}
    for kind in ("ann", "snn", "gru"):
        audio_ckpt = torch.load(OUT / f"slurp_audio_whisper_plain_{kind}_11.pt",
                                map_location="cpu", weights_only=False)
        text_ckpt = torch.load(OUT / f"slurp_oracle_{kind}_11.pt",
                               map_location="cpu", weights_only=False)
        audio_net = (GRUDecisionNet(n_actions=len(audio_ckpt["actions"]), input_dim=384)
                     if kind == "gru" else
                     DecisionNet(kind=kind, n_actions=len(audio_ckpt["actions"]), input_dim=384)).eval()
        audio_net.load_state_dict(audio_ckpt["state_dict"])
        text_net = (GRUDecisionNet(n_actions=len(text_ckpt["actions"]),
                                  vocab_size=len(text_ckpt["vocab"]) + 2)
                    if kind == "gru" else
                    DecisionNet(kind=kind, n_actions=len(text_ckpt["actions"]),
                                vocab_size=len(text_ckpt["vocab"]) + 2)).eval()
        text_net.load_state_dict(text_ckpt["state_dict"])
        data[kind] = {"audio_net": audio_net, "text_net": text_net,
                      "vocab": text_ckpt["vocab"],
                      "audio_decision": [], "text_decision": [],
                      "audio_upstream": [], "asr_upstream": [],
                      "audio_total": [], "asr_total": []}

    durations = []
    for index, row in enumerate(chosen):
        wave, sr = sf.read(AUDIO / row["audio_file"], dtype="float32")
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        durations.append(len(wave) / sr)
        resample_t = 0.0
        if sr != 16000:
            wave, resample_t = timed_call(
                lambda: resample_poly(wave, 16000, sr).astype(np.float32))
        audio_features, upstream_t = timed_call(
            lambda: pool(np.asarray(audio_model.encode(audio_model.feature_extractor(wave)))[0]))
        audio_features, normalize_t = timed_call(
            lambda: np.clip((audio_features - mean) / std, -5, 5).astype(np.float32))
        upstream_t += normalize_t + resample_t
        audio_tensor = torch.from_numpy(audio_features[None])
        (segments, _info), asr_t = timed_call(lambda: audio_model.transcribe(
            wave, language="en", beam_size=1, condition_on_previous_text=False,
            vad_filter=False))
        # Transcription is lazy: consume the generator inside its measured interval.
        hypothesis, decode_t = timed_call(
            lambda: " ".join(segment.text.strip() for segment in segments).strip())
        asr_t += decode_t + resample_t
        for kind, item in data.items():
            token_start = time.perf_counter()
            text_tokens = np.zeros((1, MAX_TOKENS), dtype=np.int64)
            for j, token in enumerate(tokenize(hypothesis)[:MAX_TOKENS]):
                text_tokens[0, j] = item["vocab"].get(token, 1)
            token_t = time.perf_counter() - token_start
            text_tensor = torch.from_numpy(text_tokens)
            with torch.no_grad():
                # Warm up model operations once on each sample shape.
                item["audio_net"](audio_tensor)
                item["text_net"](text_tensor)
                _, audio_t = timed_call(lambda: item["audio_net"](audio_tensor))
                _, text_t = timed_call(lambda: item["text_net"](text_tensor))
            item["audio_decision"].append(audio_t)
            item["text_decision"].append(text_t)
            item["audio_upstream"].append(upstream_t)
            item["asr_upstream"].append(asr_t + token_t)
            item["audio_total"].append(upstream_t + audio_t)
            item["asr_total"].append(asr_t + token_t + text_t)
        if (index + 1) % 20 == 0:
            print(f"profiled {index + 1}/{len(chosen)}", flush=True)

    result = {"device": "host CPU", "sample_count": len(chosen),
              "measurement": "single utterance, loaded waveform to action logits; no file I/O or speech observation time",
              "audio_duration_seconds": {"p50": float(np.median(durations)),
                                         "p95": float(np.percentile(durations, 95)),
                                         "mean": float(np.mean(durations))}}
    for kind, item in data.items():
        result[kind] = {key: stats(item[key]) for key in (
            "audio_decision", "text_decision", "audio_upstream",
            "asr_upstream", "audio_total", "asr_total")}
    (OUT / "slurp_pipeline_cpu_timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
