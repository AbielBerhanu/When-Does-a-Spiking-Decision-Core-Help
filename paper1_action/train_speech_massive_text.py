"""Five-seed French text cores, selected only with internal prompt development data."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from prepare_speech_massive import files
from run_text import MAX_TOKENS, VOCAB_LIMIT, tokenize, train


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
OUT = ROOT / "paper1_action/results"
KINDS = ("ann", "leaky_ann", "snn", "gru")
SEEDS = (11, 13, 17, 19, 23)


def load():
    manifest = json.loads((CACHE / "speech_massive_fr_manifest.json").read_text())
    with np.load(CACHE / "speech_massive_fr_whisper_tiny.npz") as z:
        y, split = z["y"], z["split"]
    if len(y) != len(manifest["rows"]):
        raise ValueError("French manifest/label mismatch")
    by_id = {}
    for source_split in ("train", "validation"):
        for path in files(source_split):
            parquet = pq.ParquetFile(path)
            for row in parquet.read(columns=["id", "utt", "intent_idx"]).to_pylist():
                by_id[row["id"]] = row
    ids = [r["id"] for r in manifest["rows"]]
    if len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
        raise ValueError("Duplicate/missing French utterance")
    if any(int(by_id[item]["intent_idx"]) != int(label) for item, label in zip(ids, y)):
        raise ValueError("French prompt labels disagree with acoustic cache")
    texts = [by_id[item]["utt"] for item in ids]
    return texts, y, split, ids


def arrays_for(texts, y, split):
    train_texts = [texts[i] for i in np.flatnonzero(split == 0)]
    count = Counter(token for text in train_texts for token in tokenize(text))
    vocab = {token: i + 2 for i, (token, _) in enumerate(count.most_common(VOCAB_LIMIT - 2))}
    arrays = {}
    for name, code in (("train", 0), ("valid", 1), ("test", 2)):
        idx = np.flatnonzero(split == code)
        x = np.zeros((len(idx), MAX_TOKENS), dtype=np.int64)
        for j, i in enumerate(idx):
            for k, token in enumerate(tokenize(texts[i])[:MAX_TOKENS]):
                x[j, k] = vocab.get(token, 1)
        arrays[name] = x, y[idx]
    return arrays, vocab


def main():
    texts, y, split, ids = load()
    if tuple(int((split == code).sum()) for code in (0, 1, 2)) != (10333, 1149, 2031):
        raise ValueError("French partition counts changed")
    arrays, vocab = arrays_for(texts, y, split)
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        for seed in SEEDS:
            name = f"speech_massive_fr_text_{kind}_{seed}"
            path = OUT / f"{name}.pt"
            if path.exists() and torch.load(path, map_location="cpu", weights_only=False).get("complete_training"):
                print("Reusing completed", name, flush=True)
                continue
            model, history, best_epoch = train(kind, seed, arrays, len(vocab) + 2, 60, 18, 128)
            torch.save({"state_dict": model.state_dict(), "kind": kind,
                        "seed": seed, "vocab": vocab, "best_epoch": best_epoch,
                        "history": history, "complete_training": True}, path)
            print("Fitted", name, "epoch", best_epoch, flush=True)
    bow_path = OUT / "speech_massive_fr_text_bow.joblib"
    if not bow_path.exists():
        fit_idx = np.flatnonzero(split == 0)
        vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                     lowercase=False, ngram_range=(1, 2),
                                     max_features=5000, sublinear_tf=True)
        x_fit = vectorizer.fit_transform([texts[i] for i in fit_idx])
        classifier = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        classifier.fit(x_fit, y[fit_idx])
        if not np.array_equal(classifier.classes_, np.arange(60)):
            raise ValueError("French TF-IDF classes disagree with ontology")
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, bow_path)
        print("Fitted French TF-IDF", len(vectorizer.vocabulary_), flush=True)


if __name__ == "__main__":
    main()
