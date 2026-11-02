"""Validation-selected action/deferral cost simulation on SLURP.

Wrong action costs 1; deferral costs one of the stated hypothetical fractions.
This is an offline decision-policy analysis, not measured user utility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluate_asr_text import encode
from model import DecisionNet, GRUDecisionNet
from run_slurp_audio import split_arrays
from run_easycall import predict as predict_audio
from run_text import predict as predict_text, softmax


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
RESULTS = ROOT / "paper1_action/results"
COSTS = (0.1, 0.2, 0.5)
SEEDS = (11, 13, 17, 19, 23)
KINDS = ("ann", "leaky_ann", "snn", "gru")


def threshold(probs, y, defer_cost):
    conf = probs.max(axis=1)
    wrong = (probs.argmax(axis=1) != y).astype(float)
    order = np.argsort(-conf)
    sorted_conf = conf[order]
    cumulative_wrong = np.r_[0, np.cumsum(wrong[order])]
    n = len(y)
    losses = (cumulative_wrong + defer_cost * (n - np.arange(n + 1))) / n
    k = int(np.argmin(losses))
    if k == 0:
        return 1.01
    if k == n:
        return 0.0
    return float((sorted_conf[k - 1] + sorted_conf[k]) / 2)


def evaluate(probs, y, cut, defer_cost):
    accept = probs.max(axis=1) >= cut
    incorrect = probs.argmax(axis=1) != y
    return {"coverage": float(accept.mean()),
            "wrong_action_rate": float((accept & incorrect).mean()),
            "defer_rate": float((~accept).mean()),
            "mean_cost": float(((accept & incorrect) + defer_cost * (~accept)).mean())}


def model_for(kind, checkpoint, representation):
    actions = checkpoint["actions"]
    if representation == "audio":
        params = {"n_actions": len(actions), "input_dim": 384}
    else:
        params = {"n_actions": len(actions), "vocab_size": len(checkpoint["vocab"]) + 2}
    model = GRUDecisionNet(**params) if kind == "gru" else DecisionNet(kind=kind, **params)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main():
    torch.set_num_threads(4)
    audio_splits, _actions, _seconds = split_arrays("whisper")
    asr_rows = [json.loads(s) for s in (CACHE / "slurp_asr_tiny_en.jsonl").read_text().splitlines()]
    asr_valid = [r for r in asr_rows if r["split"] == "valid"]
    manifest = json.loads((CACHE / "slurp_audio_manifest.json").read_text())
    action_id = {a: i for i, a in enumerate(manifest["actions"])}
    id_to_action = {r["slurp_id"]: action_id[r["action"]] for r in manifest["rows"]}
    asr_valid_y = np.array([id_to_action[r["slurp_id"]] for r in asr_valid])
    rows = []
    for path in ("audio", "asr"):
        prefix = "slurp_audio_whisper_plain" if path == "audio" else "slurp_asr"
        ckpt_prefix = prefix if path == "audio" else "slurp_oracle"
        for kind in KINDS:
            for seed in SEEDS:
                result = json.loads((RESULTS / f"{prefix}_{kind}_{seed}.json").read_text())
                ckpt = torch.load(RESULTS / f"{ckpt_prefix}_{kind}_{seed}.pt",
                                  map_location="cpu", weights_only=False)
                model = model_for(kind, ckpt, path)
                if path == "audio":
                    valid_x, valid_y, _ = audio_splits["valid"]
                    valid_scores, _ = predict_audio(model, valid_x)
                else:
                    valid_x = encode(asr_valid, ckpt["vocab"])
                    valid_y = asr_valid_y
                    valid_scores, _ = predict_text(model, valid_x)
                with np.load(RESULTS / f"{prefix}_{kind}_{seed}_predictions.npz") as z:
                    test_scores, test_y = z["scores"], z["y"]
                temp = result["temperature"]
                valid_probs, test_probs = softmax(valid_scores / temp), softmax(test_scores / temp)
                for defer_cost in COSTS:
                    cut = threshold(valid_probs, valid_y, defer_cost)
                    rows.append({"path": path, "kind": kind, "seed": seed,
                                 "defer_cost": defer_cost, "threshold": cut,
                                 "development": evaluate(valid_probs, valid_y, cut, defer_cost),
                                 "test": evaluate(test_probs, test_y, cut, defer_cost)})
                print(path, kind, seed, flush=True)
    grouped = {}
    for path in ("audio", "asr"):
        grouped[path] = {}
        for defer_cost in COSTS:
            grouped[path][str(defer_cost)] = {}
            for kind in KINDS:
                part = [r for r in rows if r["path"] == path and r["kind"] == kind
                        and r["defer_cost"] == defer_cost]
                grouped[path][str(defer_cost)][kind] = {
                    field: float(np.mean([r["test"][field] for r in part]))
                    for field in ("coverage", "wrong_action_rate", "defer_rate", "mean_cost")}
    bow = json.loads((RESULTS / "slurp_asr_bow.json").read_text())
    with np.load(RESULTS / "slurp_asr_bow_development.npz") as z:
        bow_valid_scores, bow_valid_y = z["scores"], z["y"]
    with np.load(RESULTS / "slurp_asr_bow_predictions.npz") as z:
        bow_test_scores, bow_test_y = z["scores"], z["y"]
    bow_valid_probs = softmax(bow_valid_scores / bow["temperature"])
    bow_test_probs = softmax(bow_test_scores / bow["temperature"])
    bow_single_fit = {}
    for defer_cost in COSTS:
        cut = threshold(bow_valid_probs, bow_valid_y, defer_cost)
        bow_single_fit[str(defer_cost)] = {
            "threshold": cut,
            "development": evaluate(bow_valid_probs, bow_valid_y, cut, defer_cost),
            "test": evaluate(bow_test_probs, bow_test_y, cut, defer_cost)}
    output = {"cost_definition": "wrong action=1, deferral=hypothetical fraction; correct action=0",
              "per_seed": rows, "five_seed_means": grouped,
              "bow_asr_single_fit": bow_single_fit}
    (RESULTS / "decision_cost.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
