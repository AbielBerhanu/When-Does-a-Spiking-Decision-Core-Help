"""Export paired held-out scores from saved oracle-transcript checkpoints."""
from __future__ import annotations

import numpy as np
import torch

from model import DecisionNet, GRUDecisionNet
from run_text import OUT, predict, prepare, read_rows


def main():
    arrays, vocab, actions = prepare(read_rows())
    for kind in ("ann", "snn", "gru"):
        for seed in (11, 13, 17):
            name = f"slurp_oracle_{kind}_{seed}"
            ckpt = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=False)
            model = (GRUDecisionNet(n_actions=len(actions), vocab_size=len(vocab) + 2)
                     if kind == "gru" else
                     DecisionNet(kind=kind, n_actions=len(actions), vocab_size=len(vocab) + 2))
            model.load_state_dict(ckpt["state_dict"])
            torch.set_num_threads(4)
            scores, _ = predict(model, arrays["test"][0])
            np.savez_compressed(OUT / f"{name}_predictions.npz",
                                scores=scores, y=arrays["test"][1])
            print(name, flush=True)


if __name__ == "__main__":
    main()
