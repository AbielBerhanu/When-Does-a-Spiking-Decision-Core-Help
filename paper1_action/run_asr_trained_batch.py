"""Run the five-seed ASR-trained text sensitivity comparison."""
from __future__ import annotations

import sys

from run_asr_trained_text import main


if __name__ == "__main__":
    for kind in ("ann", "leaky_ann", "snn", "gru"):
        for seed in (11, 13, 17, 19, 23):
            sys.argv = [sys.argv[0], "--kind", kind, "--seed", str(seed)]
            main()
