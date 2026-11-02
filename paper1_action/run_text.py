"""Oracle-transcript action selection on official SLURP splits.

This experiment does not evaluate a real ASR pipeline. Do not describe its text
results as recognized-word or deployed-pipeline results in a paper.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DecisionNet, GRUDecisionNet


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/slurp_repo/dataset/slurp"
OUT = ROOT / "paper1_action/results"
MAX_TOKENS = 40
VOCAB_LIMIT = 5000
TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def tokenize(s: str) -> list[str]:
    return TOKEN_PATTERN.findall(s.lower())


def read_rows() -> dict[str, list[dict]]:
    rows = {}
    for split, filename in (("train", "train.jsonl"), ("valid", "devel.jsonl"), ("test", "test.jsonl")):
        rows[split] = [json.loads(line) for line in (SOURCE / filename).read_text().splitlines()]
    return rows


def prepare(rows: dict[str, list[dict]]):
    train = rows["train"]
    actions = sorted({f"{r['scenario']}_{r['action']}" for r in train})
    action_to_id = {s: i for i, s in enumerate(actions)}
    count = Counter(token for row in train for token in tokenize(row["sentence"]))
    vocab = {token: i + 2 for i, (token, _) in enumerate(count.most_common(VOCAB_LIMIT - 2))}
    arrays = {}
    for split, part in rows.items():
        x = np.zeros((len(part), MAX_TOKENS), dtype=np.int64)
        y = np.zeros(len(part), dtype=np.int64)
        for i, row in enumerate(part):
            for j, token in enumerate(tokenize(row["sentence"])[:MAX_TOKENS]):
                x[i, j] = vocab.get(token, 1)
            y[i] = action_to_id[f"{row['scenario']}_{row['action']}"]
        arrays[split] = x, y
    return arrays, vocab, actions


def predict(model, x, batch=256):
    model.eval()
    scores, activity = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logit, rate = model(torch.from_numpy(x[start:start + batch]), return_activity=True)
            scores.append(logit.numpy())
            if rate is not None:
                activity.append(rate.numpy())
    return np.concatenate(scores), (float(np.concatenate(activity).mean()) if activity else None)


def softmax(logits):
    scaled = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_at_temperature(log_t, scores, y):
    p = softmax(scores / np.exp(log_t))
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def calibration_temperature(scores, y):
    result = minimize_scalar(nll_at_temperature, bounds=(-3, 3), method="bounded", args=(scores, y))
    return float(np.exp(result.x))


def ece(probs, y, bins=15):
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    return float(sum((mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
                     for lo, hi in zip(edges[:-1], edges[1:])
                     if (mask := (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)).any()))


def selective_threshold(probs, y, max_risk=0.10):
    conf = probs.max(axis=1)
    errors = (probs.argmax(axis=1) != y).astype(np.int64)
    order = np.argsort(-conf)
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(y) + 1)
    valid = np.flatnonzero(cumulative_risk <= max_risk)
    if len(valid) == 0:
        return 1.01
    return float(conf[order[valid[-1]]])


def metrics(scores, y, temperature, threshold):
    probs = softmax(scores / temperature)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    onehot = np.eye(probs.shape[1])[y]
    return {
        "n": int(len(y)), "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "nll": float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "ece_15": ece(probs, y),
        "selective_coverage": float(accepted.mean()),
        "selective_error": float((pred[accepted] != y[accepted]).mean()) if accepted.any() else None,
    }


def train(kind, seed, arrays, vocab_size, n_actions, epochs, batch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = (GRUDecisionNet(n_actions=n_actions, vocab_size=vocab_size) if kind == "gru"
             else DecisionNet(kind=kind, n_actions=n_actions, vocab_size=vocab_size))
    x_train, y_train = arrays["train"]
    x_valid, y_valid = arrays["valid"]
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    best_state, best_loss, best_epoch, patience = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        for x, y in loader:
            optim.zero_grad()
            logits, rate = model(x, return_activity=True)
            loss = nn.functional.cross_entropy(logits, y)
            if rate is not None:
                loss = loss + 0.002 * rate.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.detach().item() * len(y)
        valid_scores, valid_rate = predict(model, x_valid)
        valid_nll = nll_at_temperature(0.0, valid_scores, y_valid)
        row = {"epoch": epoch, "train_loss": total / len(y_train), "valid_nll": valid_nll,
               "valid_accuracy": float((valid_scores.argmax(axis=1) == y_valid).mean()),
               "valid_spike_rate": valid_rate, "seconds": time.perf_counter() - t0}
        history.append(row)
        print(kind, seed, row, flush=True)
        if valid_nll < best_loss - 0.001:
            best_loss, best_epoch, patience = valid_nll, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def run(kind, seed, arrays, vocab, actions, epochs, batch):
    model, history, best_epoch = train(kind, seed, arrays, len(vocab) + 2, len(actions), epochs, batch)
    valid_scores, _ = predict(model, arrays["valid"][0])
    temperature = calibration_temperature(valid_scores, arrays["valid"][1])
    threshold = selective_threshold(softmax(valid_scores / temperature), arrays["valid"][1])
    test_scores, spike_rate = predict(model, arrays["test"][0])
    result = {"benchmark": "slurp", "input": "oracle_transcript", "kind": kind, "seed": seed,
              "n_parameters": model.n_parameters, "best_epoch": best_epoch,
              "temperature": temperature, "threshold_10pct_dev_error": threshold,
              "mean_hidden_spike_rate": spike_rate,
              "valid": metrics(valid_scores, arrays["valid"][1], temperature, threshold),
              "test": metrics(test_scores, arrays["test"][1], temperature, threshold),
              "history": history}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"slurp_oracle_{kind}_{seed}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"slurp_oracle_{kind}_{seed}_predictions.npz",
                        scores=test_scores, y=arrays["test"][1])
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "actions": actions},
               OUT / f"slurp_oracle_{kind}_{seed}.pt")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["snn", "ann", "leaky_ann", "gru"], required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    arrays, vocab, actions = prepare(read_rows())
    run(a.kind, a.seed, arrays, vocab, actions, a.epochs, a.batch)


if __name__ == "__main__":
    main()
