"""Extend the fixed SLURP recurrent comparisons from three to five seeds."""
from __future__ import annotations

from run_slurp_audio import run as audio_run, split_arrays
from run_text import prepare, read_rows, run as oracle_run
from evaluate_asr_text import run as asr_run


def main():
    seeds = (19, 23)
    kinds = ("ann", "leaky_ann", "snn", "gru")
    audio_splits, actions, encoder_seconds = split_arrays("whisper")
    text_arrays, vocab, text_actions = prepare(read_rows())
    if actions != text_actions:
        raise ValueError("Action catalogs differ")
    for seed in seeds:
        for kind in kinds:
            audio_run(kind, seed, audio_splits, actions, 24, 128, "plain", "whisper",
                      encoder_seconds)
            oracle_run(kind, seed, text_arrays, vocab, text_actions, 18, 128)
            asr_run(kind, seed)


if __name__ == "__main__":
    main()
