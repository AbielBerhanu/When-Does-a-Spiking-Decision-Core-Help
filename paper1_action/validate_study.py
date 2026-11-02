"""Recompute key results and audit split/label integrity before submission."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_text import metrics


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper1_action/cache"
RESULTS = ROOT / "paper1_action/results"
SEEDS = (11, 13, 17, 19, 23)


def audit_manifest(name, key):
    data = json.loads((CACHE / f"{name}_manifest.json").read_text())
    rows = data["rows"]
    splits = {s: {r[key] for r in rows if r["split"] == s}
              for s in ("train", "valid", "test")}
    assert all(not (splits[a] & splits[b]) for a, b in
               (("train", "valid"), ("train", "test"), ("valid", "test")))
    assert all(r["action"] in data["actions"] for r in rows)
    return len(rows), len(data["actions"])


def main():
    slurp = audit_manifest("slurp_audio", "slurp_id")
    easycall = audit_manifest("easycall", "speaker")
    assert slurp == (16521, 60)
    assert easycall == (21386, 36)
    slurp_manifest = json.loads((CACHE / "slurp_audio_manifest.json").read_text())
    audio_by_split = {s: {r["audio_file"] for r in slurp_manifest["rows"] if r["split"] == s}
                      for s in ("train", "valid", "test")}
    assert all(not (audio_by_split[a] & audio_by_split[b]) for a, b in
               (("train", "valid"), ("train", "test"), ("valid", "test")))
    checked = 0
    for prefix in ("slurp_oracle", "slurp_asr", "slurp_asr_trained",
                   "slurp_audio_whisper_plain", "speech_massive_fr_audio",
                   "speech_massive_fr_asr"):
        expected_n = 2031 if prefix.startswith("speech_massive") else 2974
        for kind in ("ann", "leaky_ann", "snn", "gru"):
            for seed in SEEDS:
                stem = f"{prefix}_{kind}_{seed}"
                row = json.loads((RESULTS / f"{stem}.json").read_text())
                with np.load(RESULTS / f"{stem}_predictions.npz") as z:
                    scores, y = z["scores"], z["y"]
                assert len(y) == expected_n and scores.shape == (expected_n, 60)
                assert np.isfinite(scores).all()
                threshold = row["threshold_10pct_dev_error"]
                actual = metrics(scores, y, row["temperature"], threshold)
                for measure in ("accuracy", "macro_f1", "nll", "brier",
                                "ece_15", "selective_coverage", "selective_error"):
                    assert abs(actual[measure] - row["test"][measure]) < 1e-6, (stem, measure)
                checked += 1
        ann = json.loads((RESULTS / f"{prefix}_ann_11.json").read_text())
        snn = json.loads((RESULTS / f"{prefix}_snn_11.json").read_text())
        assert ann.get("n_parameters", ann.get("n_parameters_decision_model")) == snn.get(
            "n_parameters", snn.get("n_parameters_decision_model"))
    with np.load(CACHE / "speech_massive_fr_whisper_tiny.npz") as z:
        assert len(z["y"]) == 13513
        assert tuple(int((z["split"] == code).sum()) for code in (0, 1, 2)) == (
            10333, 1149, 2031)
        assert np.isfinite(z["x"]).all()
    fresh_manifest = json.loads((CACHE / "speech_massive_fr_manifest.json").read_text())
    assert len({r["id"] for r in fresh_manifest["rows"]}) == 13513
    french_asr = [json.loads(s) for s in
                  (CACHE / "speech_massive_fr_asr_tiny.jsonl").read_text().splitlines()]
    french_ids = {r["id"] for r in french_asr}
    assert len(french_asr) == len(french_ids) == 3180
    with np.load(CACHE / "speech_massive_fr_whisper_tiny.npz") as z:
        expected_french_ids = {r["id"] for r, code in
                               zip(fresh_manifest["rows"], z["split"])
                               if code in (1, 2)}
    assert french_ids == expected_french_ids
    asr = [json.loads(s) for s in (CACHE / "slurp_asr_tiny_en.jsonl").read_text().splitlines()]
    assert len(asr) == 5007 and len({r["slurp_id"] for r in asr}) == 5007
    train_asr = [json.loads(s) for s in (CACHE / "slurp_asr_train_tiny_en.jsonl").read_text().splitlines()]
    assert len(train_asr) == 11514 and len({r["slurp_id"] for r in train_asr}) == 11514
    assert not {r["slurp_id"] for r in train_asr} & {r["slurp_id"] for r in asr}
    print(json.dumps({"slurp_samples_actions": slurp,
                      "easycall_samples_actions": easycall,
                      "verified_recurrent_result_files": checked,
                      "unique_french_asr_hypotheses": len(french_asr),
                      "unique_asr_hypotheses": len(asr),
                      "unique_train_asr_hypotheses": len(train_asr)}, indent=2))


if __name__ == "__main__":
    main()
