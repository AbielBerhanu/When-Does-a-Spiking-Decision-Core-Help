"""Evaluate the SLURP text decision core on real ASR hypotheses."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from model import DecisionNet, GRUDecisionNet
from run_text import (MAX_TOKENS, OUT, calibration_temperature, metrics,
                      predict, selective_threshold, softmax, tokenize)


ROOT = Path(__file__).resolve().parents[1]
ASR = ROOT / "paper1_action/cache/slurp_asr_tiny_en.jsonl"
MANIFEST = ROOT / "paper1_action/cache/slurp_audio_manifest.json"


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def word_error_rate(rows):
    edits, words = 0, 0
    for row in rows:
        ref = tokenize(row["reference"])
        hyp = tokenize(row["hypothesis"])
        edits += edit_distance(ref, hyp)
        words += len(ref)
    return edits / words


def encode(rows, vocab):
    x = np.zeros((len(rows), MAX_TOKENS), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, token in enumerate(tokenize(row["hypothesis"])[:MAX_TOKENS]):
            x[i, j] = vocab.get(token, 1)
    return x


def run(kind, seed):
    manifest = json.loads(MANIFEST.read_text())
    by_id = {r["slurp_id"]: r for r in manifest["rows"]}
    actions = manifest["actions"]
    asr_rows = [json.loads(line) for line in ASR.read_text().splitlines()]
    if len(asr_rows) != 5007 or len({r["slurp_id"] for r in asr_rows}) != 5007:
        raise ValueError("Expected one ASR hypothesis per development/test utterance")
    checkpoint = torch.load(OUT / f"slurp_oracle_{kind}_{seed}.pt", map_location="cpu",
                            weights_only=False)
    if checkpoint["actions"] != actions:
        raise ValueError("Action catalog mismatch")
    vocab = checkpoint["vocab"]
    model = (GRUDecisionNet(n_actions=len(actions), vocab_size=len(vocab) + 2) if kind == "gru"
             else DecisionNet(kind=kind, n_actions=len(actions), vocab_size=len(vocab) + 2))
    model.load_state_dict(checkpoint["state_dict"])
    torch.set_num_threads(4)
    splits = {}
    for split in ("valid", "test"):
        part = [r for r in asr_rows if r["split"] == split]
        x = encode(part, vocab)
        y = np.array([actions.index(by_id[r["slurp_id"]]["action"]) for r in part])
        scores, spike_rate = predict(model, x)
        splits[split] = part, y, scores, spike_rate
    valid_rows, valid_y, valid_scores, _ = splits["valid"]
    temperature = calibration_temperature(valid_scores, valid_y)
    threshold = selective_threshold(softmax(valid_scores / temperature), valid_y)
    test_rows, test_y, test_scores, test_rate = splits["test"]
    result = {"benchmark": "slurp", "input": "asr_hypothesis", "asr": "faster_whisper_tiny.en_int8",
              "kind": kind, "seed": seed, "n_parameters_decision_model": model.n_parameters,
              "temperature": temperature, "threshold_10pct_dev_error": threshold,
              "mean_hidden_spike_rate": test_rate,
              "valid_wer": word_error_rate(valid_rows), "test_wer": word_error_rate(test_rows),
              "mean_asr_seconds_valid": float(np.mean([r["asr_seconds"] for r in valid_rows])),
              "mean_asr_seconds_test": float(np.mean([r["asr_seconds"] for r in test_rows])),
              "valid": metrics(valid_scores, valid_y, temperature, threshold),
              "test": metrics(test_scores, test_y, temperature, threshold)}
    (OUT / f"slurp_asr_{kind}_{seed}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"slurp_asr_{kind}_{seed}_predictions.npz",
                        scores=test_scores, y=test_y,
                        slurp_id=np.array([r["slurp_id"] for r in test_rows]))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["snn", "ann", "leaky_ann", "gru"], required=True)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    run(args.kind, args.seed)


if __name__ == "__main__":
    main()
