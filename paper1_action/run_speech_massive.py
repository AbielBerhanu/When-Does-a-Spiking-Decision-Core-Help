"""Precommitted five-seed French Speech-MASSIVE acoustic replication."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_easycall import predict, train
from run_text import calibration_temperature, metrics, selective_threshold, softmax


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache/speech_massive_fr_whisper_tiny.npz"
OUT = ROOT / "paper1_action/results"
KINDS = ("ann", "leaky_ann", "snn", "gru")
SEEDS = (11, 13, 17, 19, 23)


def main():
    with np.load(CACHE) as z:
        x, y, split = z["x"], z["y"], z["split"]
    manifest = json.loads((CACHE.parent / "speech_massive_fr_manifest.json").read_text())
    rows = manifest["rows"]
    if len(x) != len(y) or len(x) != len(rows):
        raise ValueError("Feature/label/manifest mismatch")
    if len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("Duplicate utterance identifiers")
    splits = {}
    for name, code in (("train", 0), ("valid", 1), ("test", 2)):
        idx = np.flatnonzero(split == code)
        splits[name] = (x[idx], y[idx], [rows[i] for i in idx])
    print({name: len(value[1]) for name, value in splits.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fitted = []
    # Finish every fixed training run before consulting the untouched official
    # validation partition. The internal development partition selects epochs.
    for kind in KINDS:
        for seed in SEEDS:
            name = f"speech_massive_fr_audio_{kind}_{seed}"
            path = OUT / f"{name}.pt"
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("complete_training"):
                    fitted.append((kind, seed, path))
                    print("Reusing completed", name, flush=True)
                    continue
            model, history, best_epoch = train(kind, seed, splits, 60, 24, 128,
                                                front="plain", balanced=False,
                                                input_dim=384)
            torch.save({"state_dict": model.state_dict(), "kind": kind,
                        "seed": seed, "history": history, "best_epoch": best_epoch,
                        "complete_training": True}, path)
            fitted.append((kind, seed, path))
            print("Fitted", name, flush=True)
    for kind, seed, path in fitted:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        from model import DecisionNet, GRUDecisionNet
        model = (GRUDecisionNet(n_actions=60, input_dim=384) if kind == "gru"
                 else DecisionNet(kind=kind, n_actions=60, input_dim=384))
        model.load_state_dict(payload["state_dict"])
        valid_scores, _ = predict(model, splits["valid"][0])
        valid_y = splits["valid"][1]
        temperature = calibration_temperature(valid_scores, valid_y)
        threshold = selective_threshold(softmax(valid_scores / temperature), valid_y)
        test_scores, test_rate = predict(model, splits["test"][0])
        test_y = splits["test"][1]
        result = {"benchmark": "speech_massive_fr", "official_evaluation_split": "validation",
                  "kind": kind, "seed": seed, "input": "frozen_multilingual_whisper_tiny_audio",
                  "n_parameters": model.n_parameters,
                  "best_epoch": payload["best_epoch"],
                  "temperature": temperature, "threshold_10pct_dev_error": threshold,
                  "mean_hidden_spike_rate": test_rate,
                  "valid": metrics(valid_scores, valid_y, temperature, threshold),
                  "test": metrics(test_scores, test_y, temperature, threshold),
                  "history": payload["history"]}
        name = f"speech_massive_fr_audio_{kind}_{seed}"
        (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / f"{name}_predictions.npz",
                            scores=test_scores, y=test_y,
                            item_id=np.array([r["id"] for r in splits["test"][2]]))
        print(name, "accuracy", result["test"]["accuracy"],
              "macro F1", result["test"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
