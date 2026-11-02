"""Train the text decision core on actual SLURP training-audio ASR output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_text import (OUT, calibration_temperature, metrics, predict, prepare,
                      read_rows, selective_threshold, softmax, train)


ROOT = Path(__file__).resolve().parents[1]


def load_asr(path: Path, expected: int) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {r["slurp_id"]: r for r in rows}
    if len(rows) != expected or len(by_id) != expected:
        raise ValueError(f"Expected {expected} unique ASR results in {path}, found {len(rows)}")
    return by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("ann", "leaky_ann", "snn", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    rows = read_rows()
    train_asr = load_asr(ROOT / "paper1_action/cache/slurp_asr_train_tiny_en.jsonl", 11514)
    eval_asr = load_asr(ROOT / "paper1_action/cache/slurp_asr_tiny_en.jsonl", 5007)
    all_asr = train_asr | eval_asr
    asr_rows = {}
    for split, part in rows.items():
        asr_rows[split] = [dict(r, sentence=all_asr[r["slurp_id"]]["hypothesis"])
                           for r in part]
    arrays, vocab, actions = prepare(asr_rows)
    model, history, best_epoch = train(args.kind, args.seed, arrays, len(vocab) + 2,
                                       len(actions), 18, 128)
    valid_scores, _ = predict(model, arrays["valid"][0])
    valid_y = arrays["valid"][1]
    temperature = calibration_temperature(valid_scores, valid_y)
    threshold = selective_threshold(softmax(valid_scores / temperature), valid_y)
    test_scores, activity = predict(model, arrays["test"][0])
    test_y = arrays["test"][1]
    result = {"benchmark": "slurp", "input": "asr_hypothesis",
              "training_input": "asr_hypothesis", "kind": args.kind, "seed": args.seed,
              "n_parameters": model.n_parameters, "best_epoch": best_epoch,
              "temperature": temperature, "threshold_10pct_dev_error": threshold,
              "mean_hidden_spike_rate": activity,
              "valid": metrics(valid_scores, valid_y, temperature, threshold),
              "test": metrics(test_scores, test_y, temperature, threshold),
              "history": history}
    OUT.mkdir(exist_ok=True)
    name = f"slurp_asr_trained_{args.kind}_{args.seed}"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"{name}_predictions.npz", scores=test_scores, y=test_y)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "actions": actions},
               OUT / f"{name}.pt")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
