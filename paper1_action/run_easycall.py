"""Speaker-disjoint EasyCall phone-action decision study from recorded audio."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ConvAudioDecisionNet, DecisionNet, GRUDecisionNet
from run_text import calibration_temperature, metrics, softmax


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
OUT = ROOT / "paper1_action/results"


def predict(model, x, batch=256):
    model.eval()
    scores, activity = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits, rate = model(torch.from_numpy(x[start:start + batch]), return_activity=True)
            scores.append(logits.numpy())
            if rate is not None:
                activity.append(rate.numpy())
    return np.concatenate(scores), (float(np.concatenate(activity).mean()) if activity else None)


def split_arrays():
    with np.load(CACHE / "easycall_audio.npz") as z:
        x, y = z["x"], z["y"]
    manifest = json.loads((CACHE / "easycall_manifest.json").read_text())
    rows, actions = manifest["rows"], manifest["actions"]
    if len(rows) != len(y):
        raise ValueError("Feature/manifest length mismatch")
    splits = {}
    for split in ("train", "valid", "test"):
        indices = np.array([i for i, row in enumerate(rows) if row["split"] == split])
        splits[split] = x[indices], y[indices], [rows[i] for i in indices]
    sets = [{r["speaker"] for r in splits[s][2]} for s in ("train", "valid", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Speaker leakage across splits")
    return splits, actions


def false_execution_threshold(probs, y, no_action_id, maximum=0.01):
    """Lowest development threshold with <=1% false action execution on no-actions."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    negatives = y == no_action_id
    if not negatives.any():
        raise ValueError("No noncommands in development set")
    for threshold in np.r_[0.0, np.unique(conf[negatives & (pred != no_action_id)])]:
        false_exec = ((pred[negatives] != no_action_id) & (conf[negatives] >= threshold)).mean()
        if false_exec <= maximum:
            return float(threshold)
    return 1.01


def assistive_metrics(scores, y, actions, temperature, execution_threshold):
    probs = softmax(scores / temperature)
    no_action_id = actions.index("no_action")
    raw_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    execute = (raw_pred != no_action_id) & (conf >= execution_threshold)
    noncommand = y == no_action_id
    command = ~noncommand
    return {
        **metrics(scores, y, temperature, execution_threshold),
        "n_commands": int(command.sum()), "n_noncommands": int(noncommand.sum()),
        "false_execution_on_noncommands": float(execute[noncommand].mean()),
        "correct_execution_on_commands": float((execute[command] & (raw_pred[command] == y[command])).mean()),
        "wrong_execution_on_commands": float((execute[command] & (raw_pred[command] != y[command])).mean()),
        "deferred_or_no_action_on_commands": float((~execute[command]).mean()),
        "emergency_false_execution_count": int((execute & (raw_pred == actions.index("emergency_call")) &
                                                (y != actions.index("emergency_call"))).sum()),
    }


def train(kind, seed, splits, n_actions, epochs, batch, front, balanced=True, input_dim=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = (ConvAudioDecisionNet(kind=kind, n_actions=n_actions) if front == "conv"
             else GRUDecisionNet(n_actions=n_actions, input_dim=input_dim) if kind == "gru"
             else DecisionNet(kind=kind, n_actions=n_actions, input_dim=input_dim))
    x_train, y_train, _ = splits["train"]
    x_valid, y_valid, _ = splits["valid"]
    counts = np.bincount(y_train, minlength=n_actions)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1)) if balanced else np.ones(n_actions)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights.astype(np.float32)))
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.001)
    best_state, best_nll, best_epoch, patience = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            logits, rate = model(x, return_activity=True)
            loss = criterion(logits, y)
            if rate is not None:
                loss = loss + 0.002 * rate.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach().item() * len(y)
        valid_scores, valid_rate = predict(model, x_valid)
        probs = softmax(valid_scores)
        valid_nll = float(-np.log(np.maximum(probs[np.arange(len(y_valid)), y_valid], 1e-12)).mean())
        row = {"epoch": epoch, "train_loss": total_loss / len(y_train),
               "valid_nll": valid_nll,
               "valid_accuracy": float((valid_scores.argmax(axis=1) == y_valid).mean()),
               "valid_spike_rate": valid_rate, "seconds": time.perf_counter() - start}
        history.append(row)
        print(kind, seed, row, flush=True)
        if valid_nll < best_nll - 0.001:
            best_nll, best_epoch, patience = valid_nll, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def run(kind, seed, splits, actions, epochs, batch, front):
    model, history, best_epoch = train(kind, seed, splits, len(actions), epochs, batch, front)
    valid_scores, _ = predict(model, splits["valid"][0])
    y_valid = splits["valid"][1]
    temperature = calibration_temperature(valid_scores, y_valid)
    threshold = false_execution_threshold(softmax(valid_scores / temperature), y_valid,
                                          actions.index("no_action"))
    test_scores, test_rate = predict(model, splits["test"][0])
    test_y, test_rows = splits["test"][1:]
    groups = {group: np.array([r["speaker_group"] == group for r in test_rows])
              for group in ("dysarthric", "control")}
    result = {
        "benchmark": "easycall", "input": "audio", "kind": kind, "front": front, "seed": seed,
        "n_parameters": model.n_parameters, "best_epoch": best_epoch,
        "temperature": temperature, "execution_threshold_1pct_dev_false_action": threshold,
        "mean_hidden_spike_rate": test_rate,
        "valid": assistive_metrics(valid_scores, y_valid, actions, temperature, threshold),
        "test": assistive_metrics(test_scores, test_y, actions, temperature, threshold),
        "by_speaker_group": {
            group: assistive_metrics(test_scores[mask], test_y[mask], actions, temperature, threshold)
            for group, mask in groups.items()
        },
        "history": history,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"easycall_audio_{front}_{kind}_{seed}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"easycall_audio_{front}_{kind}_{seed}_predictions.npz",
                        scores=test_scores, y=test_y,
                        speaker=np.array([r["speaker"] for r in test_rows]))
    torch.save({"state_dict": model.state_dict(), "actions": actions},
               OUT / f"easycall_audio_{front}_{kind}_{seed}.pt")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["snn", "ann"], required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--front", choices=["plain", "conv"], default="conv")
    args = p.parse_args()
    run(args.kind, args.seed, *split_arrays(), args.epochs, args.batch, args.front)


if __name__ == "__main__":
    main()
