"""Evaluate the fixed French reference-trained text cores on real ASR words."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from evaluate_asr_text import word_error_rate
from model import DecisionNet, GRUDecisionNet
from run_text import (MAX_TOKENS, calibration_temperature, metrics, predict,
                      selective_threshold, softmax, tokenize)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
OUT = ROOT / "paper1_action/results"
KINDS = ("ann", "leaky_ann", "snn", "gru")
SEEDS = (11, 13, 17, 19, 23)


def encode(rows, vocab):
    x = np.zeros((len(rows), MAX_TOKENS), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, token in enumerate(tokenize(row["hypothesis"])[:MAX_TOKENS]):
            x[i, j] = vocab.get(token, 1)
    return x


def main():
    manifest = json.loads((CACHE / "speech_massive_fr_manifest.json").read_text())
    with np.load(CACHE / "speech_massive_fr_whisper_tiny.npz") as z:
        y, split = z["y"], z["split"]
    rows = [json.loads(s) for s in (CACHE / "speech_massive_fr_asr_tiny.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    if len(rows) != 3180 or len(by_id) != 3180:
        raise ValueError("French ASR hypotheses incomplete or duplicated")
    parts = {}
    for name, code, expected in (("valid", 1, 1149), ("test", 2, 2031)):
        idx = np.flatnonzero(split == code)
        ids = [manifest["rows"][i]["id"] for i in idx]
        if len(ids) != expected or any(by_id[item]["split_code"] != code for item in ids):
            raise ValueError(f"French {name} partition mismatch")
        parts[name] = ([by_id[item] for item in ids], y[idx], ids)
    expected_ids = set(parts["valid"][2]) | set(parts["test"][2])
    if set(by_id) != expected_ids:
        raise ValueError("French ASR output contains unexpected IDs")
    if not all((OUT / f"speech_massive_fr_text_{kind}_{seed}.pt").exists()
               for kind in KINDS for seed in SEEDS):
        raise ValueError("Evaluate only after all 20 French text cores are trained")
    prompt_wer = {name: word_error_rate(part[0]) for name, part in parts.items()}
    torch.set_num_threads(4)
    for kind in KINDS:
        for seed in SEEDS:
            checkpoint = torch.load(OUT / f"speech_massive_fr_text_{kind}_{seed}.pt",
                                    map_location="cpu", weights_only=False)
            if not checkpoint.get("complete_training"):
                raise ValueError("Incomplete French text checkpoint")
            vocab = checkpoint["vocab"]
            model = (GRUDecisionNet(n_actions=60, vocab_size=len(vocab) + 2)
                     if kind == "gru" else
                     DecisionNet(kind=kind, n_actions=60, vocab_size=len(vocab) + 2))
            model.load_state_dict(checkpoint["state_dict"])
            valid_x = encode(parts["valid"][0], vocab)
            test_x = encode(parts["test"][0], vocab)
            valid_scores, _ = predict(model, valid_x)
            valid_y = parts["valid"][1]
            temperature = calibration_temperature(valid_scores, valid_y)
            cut = selective_threshold(softmax(valid_scores / temperature), valid_y)
            test_scores, rate = predict(model, test_x)
            test_y = parts["test"][1]
            result = {"benchmark": "speech_massive_fr", "input": "asr_hypothesis",
                      "training_input": "reference_prompt", "official_evaluation_split": "validation",
                      "kind": kind, "seed": seed, "n_parameters": model.n_parameters,
                      "best_epoch": checkpoint["best_epoch"], "temperature": temperature,
                      "threshold_10pct_dev_error": cut, "mean_hidden_spike_rate": rate,
                      "prompt_wer": prompt_wer,
                      "valid": metrics(valid_scores, valid_y, temperature, cut),
                      "test": metrics(test_scores, test_y, temperature, cut)}
            stem = f"speech_massive_fr_asr_{kind}_{seed}"
            (OUT / f"{stem}.json").write_text(json.dumps(result, indent=2))
            np.savez_compressed(OUT / f"{stem}_predictions.npz",
                                scores=test_scores, y=test_y, item_id=np.array(parts["test"][2]))
            np.savez_compressed(OUT / f"{stem}_development.npz",
                                scores=valid_scores, y=valid_y, item_id=np.array(parts["valid"][2]))
            print(stem, "accuracy", result["test"]["accuracy"], flush=True)
    baseline = joblib.load(OUT / "speech_massive_fr_text_bow.joblib")
    vectorizer, classifier = baseline["vectorizer"], baseline["classifier"]
    valid_scores = classifier.decision_function(
        vectorizer.transform([r["hypothesis"] for r in parts["valid"][0]]))
    valid_y = parts["valid"][1]
    temperature = calibration_temperature(valid_scores, valid_y)
    cut = selective_threshold(softmax(valid_scores / temperature), valid_y)
    test_scores = classifier.decision_function(
        vectorizer.transform([r["hypothesis"] for r in parts["test"][0]]))
    test_y = parts["test"][1]
    bow = {"benchmark": "speech_massive_fr", "input": "asr_hypothesis",
           "training_input": "reference_prompt", "model": "tfidf_bigram_logistic",
           "features": len(vectorizer.vocabulary_), "temperature": temperature,
           "threshold_10pct_dev_error": cut, "prompt_wer": prompt_wer,
           "valid": metrics(valid_scores, valid_y, temperature, cut),
           "test": metrics(test_scores, test_y, temperature, cut)}
    (OUT / "speech_massive_fr_asr_bow.json").write_text(json.dumps(bow, indent=2))
    np.savez_compressed(OUT / "speech_massive_fr_asr_bow_predictions.npz",
                        scores=test_scores, y=test_y, item_id=np.array(parts["test"][2]))
    np.savez_compressed(OUT / "speech_massive_fr_asr_bow_development.npz",
                        scores=valid_scores, y=valid_y, item_id=np.array(parts["valid"][2]))
    print("French TF-IDF accuracy", bow["test"]["accuracy"], "prompt WER", prompt_wer, flush=True)


if __name__ == "__main__":
    main()
