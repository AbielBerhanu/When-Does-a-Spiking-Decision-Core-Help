"""Nonrecurrent TF-IDF logistic control for SLURP action selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from run_text import (OUT, calibration_temperature, metrics, read_rows,
                      selective_threshold, softmax, tokenize)


ROOT = Path(__file__).resolve().parents[1]


def evaluate(name, model, vectorizer, texts, labels, dev_texts, dev_labels):
    dev_scores = model.decision_function(vectorizer.transform(dev_texts))
    temperature = calibration_temperature(dev_scores, dev_labels)
    threshold = selective_threshold(softmax(dev_scores / temperature), dev_labels)
    scores = model.decision_function(vectorizer.transform(texts))
    result = {"model": "tfidf_bigram_logistic", "input": name,
              "features": int(len(vectorizer.vocabulary_)),
              "temperature": temperature, "threshold_10pct_dev_error": threshold,
              "valid": metrics(dev_scores, dev_labels, temperature, threshold),
              "test": metrics(scores, labels, temperature, threshold)}
    (OUT / f"slurp_{name}_bow.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"slurp_{name}_bow_predictions.npz", scores=scores, y=labels)
    np.savez_compressed(OUT / f"slurp_{name}_bow_development.npz",
                        scores=dev_scores, y=dev_labels)
    print(json.dumps(result, indent=2), flush=True)


def main():
    rows = read_rows()
    actions = sorted({f"{r['scenario']}_{r['action']}" for r in rows["train"]})
    by_id = {r["slurp_id"]: r for split in rows.values() for r in split}
    label = lambda r: actions.index(f"{r['scenario']}_{r['action']}")
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                 lowercase=False, ngram_range=(1, 2),
                                 max_features=5000, sublinear_tf=True)
    x_train = vectorizer.fit_transform([r["sentence"] for r in rows["train"]])
    y_train = np.array([label(r) for r in rows["train"]])
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    model.fit(x_train, y_train)
    if not np.array_equal(model.classes_, np.arange(len(actions))):
        raise ValueError("Training classes do not match action catalog")
    OUT.mkdir(exist_ok=True)
    evaluate("oracle", model, vectorizer,
             [r["sentence"] for r in rows["test"]],
             np.array([label(r) for r in rows["test"]]),
             [r["sentence"] for r in rows["valid"]],
             np.array([label(r) for r in rows["valid"]]))
    asr = [json.loads(line) for line in
           (ROOT / "paper1_action/cache/slurp_asr_tiny_en.jsonl").read_text().splitlines()]
    grouped = {split: [r for r in asr if r["split"] == split]
               for split in ("valid", "test")}
    evaluate("asr", model, vectorizer,
             [r["hypothesis"] for r in grouped["test"]],
             np.array([label(by_id[r["slurp_id"]]) for r in grouped["test"]]),
             [r["hypothesis"] for r in grouped["valid"]],
             np.array([label(by_id[r["slurp_id"]]) for r in grouped["valid"]]))


if __name__ == "__main__":
    main()
