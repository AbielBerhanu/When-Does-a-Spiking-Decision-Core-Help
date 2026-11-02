"""Direct-audio action selection on paired official SLURP utterances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_easycall import predict, train
from run_text import calibration_temperature, metrics, selective_threshold, softmax


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
OUT = ROOT / "paper1_action/results"


def split_arrays(representation):
    source = "slurp_whisper_tiny_en.npz" if representation == "whisper" else "slurp_audio.npz"
    with np.load(CACHE / source) as z:
        x, y = z["x"], z["y"]
        encoder_seconds = float(z["encoder_seconds"].mean()) if "encoder_seconds" in z else None
    manifest = json.loads((CACHE / "slurp_audio_manifest.json").read_text())
    rows, actions = manifest["rows"], manifest["actions"]
    if len(rows) != len(y):
        raise ValueError("Feature/manifest length mismatch")
    splits = {}
    for split in ("train", "valid", "test"):
        indices = np.array([i for i, row in enumerate(rows) if row["split"] == split])
        splits[split] = x[indices], y[indices], [rows[i] for i in indices]
    id_sets = [{r["slurp_id"] for r in splits[s][2]} for s in ("train", "valid", "test")]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("SLURP utterance leakage across splits")
    return splits, actions, encoder_seconds


def run(kind, seed, splits, actions, epochs, batch, front, representation, encoder_seconds):
    model, history, best_epoch = train(kind, seed, splits, len(actions), epochs, batch,
                                       front=front, balanced=False,
                                       input_dim=384 if representation == "whisper" else 40)
    valid_scores, _ = predict(model, splits["valid"][0])
    y_valid = splits["valid"][1]
    temperature = calibration_temperature(valid_scores, y_valid)
    threshold = selective_threshold(softmax(valid_scores / temperature), y_valid)
    test_scores, test_rate = predict(model, splits["test"][0])
    test_y, test_rows = splits["test"][1:]
    result = {"benchmark": "slurp", "input": "audio", "representation": representation,
              "kind": kind, "front": front,
              "mean_encoder_seconds_during_cache_generation_unisolated": encoder_seconds,
              "seed": seed, "n_parameters": model.n_parameters, "best_epoch": best_epoch,
              "temperature": temperature, "threshold_10pct_dev_error": threshold,
              "mean_hidden_spike_rate": test_rate,
              "valid": metrics(valid_scores, y_valid, temperature, threshold),
              "test": metrics(test_scores, test_y, temperature, threshold),
              "history": history}
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"slurp_audio_{representation}_{front}_{kind}_{seed}"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"{name}_predictions.npz",
                        scores=test_scores, y=test_y,
                        slurp_id=np.array([r["slurp_id"] for r in test_rows]))
    torch.save({"state_dict": model.state_dict(), "actions": actions},
               OUT / f"{name}.pt")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["snn", "ann", "leaky_ann", "gru"], required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--front", choices=["plain", "conv"], default="conv")
    p.add_argument("--representation", choices=["logmel", "whisper"], default="logmel")
    args = p.parse_args()
    if args.representation == "whisper" and args.front != "plain":
        p.error("Whisper encoder states require --front plain")
    if args.kind == "gru" and args.front != "plain":
        p.error("The GRU control uses --front plain")
    splits, actions, encoder_seconds = split_arrays(args.representation)
    run(args.kind, args.seed, splits, actions, args.epochs, args.batch, args.front,
        args.representation, encoder_seconds)


if __name__ == "__main__":
    main()
