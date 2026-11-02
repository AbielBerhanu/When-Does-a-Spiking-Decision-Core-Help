"""Matched recurrent action-decision models for offline speech benchmarks.

The SNN uses a leaky integrate-and-fire hidden layer and a surrogate gradient.
Input embedding/log-mel projection and the output probability head are analog;
spike activity refers to the hidden decision core only.
"""
from __future__ import annotations

import torch
from torch import nn


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (voltage,) = ctx.saved_tensors
        return grad_output / (1 + 5 * voltage.abs()).square()


class DecisionNet(nn.Module):
    def __init__(
        self,
        *,
        kind: str,
        n_actions: int,
        input_dim: int = 40,
        vocab_size: int | None = None,
        hidden: int = 64,
        embedding_dim: int = 48,
    ) -> None:
        super().__init__()
        if kind not in {"snn", "ann", "leaky_ann"}:
            raise ValueError(kind)
        self.kind = kind
        self.n_actions = n_actions
        self.hidden = hidden
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) if vocab_size else None
        self.input = nn.Linear(embedding_dim if vocab_size else input_dim, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)
        self.readout = nn.Linear(2 * hidden, n_actions)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.25 if kind in {"snn", "leaky_ann"} else 0.8)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None,
                return_activity: bool = False):
        if self.embedding is not None:
            lengths = (x != 0).sum(dim=1).clamp_min(1) if lengths is None else lengths
            x = self.embedding(x)
        batch, steps, _ = x.shape
        if lengths is None:
            lengths = torch.full((batch,), steps, dtype=torch.long, device=x.device)
        h = x.new_zeros(batch, self.hidden)
        voltage = x.new_zeros(batch, self.hidden)
        state_sum = x.new_zeros(batch, self.hidden)
        spike_sum = x.new_zeros(batch)
        for t in range(steps):
            active = (t < lengths).unsqueeze(1)
            drive = self.input(x[:, t]) + self.recurrent(h)
            if self.kind == "snn":
                next_voltage = 0.9 * voltage + drive
                next_h = Spike.apply(next_voltage - 1.0)
                next_voltage = next_voltage - next_h.detach()
                voltage = torch.where(active, next_voltage, voltage)
                h = torch.where(active, next_h, h)
                spike_sum = spike_sum + (h * active).sum(dim=1)
            elif self.kind == "leaky_ann":
                next_voltage = 0.9 * voltage + drive
                voltage = torch.where(active, next_voltage, voltage)
                h = torch.where(active, torch.tanh(voltage), h)
            else:
                h = torch.where(active, torch.tanh(drive), h)
            state_sum = state_sum + h * active
        mean_state = state_sum / lengths.unsqueeze(1)
        logits = self.readout(torch.cat((mean_state, h), dim=1))
        if return_activity:
            rate = spike_sum / (lengths * self.hidden) if self.kind == "snn" else None
            return logits, rate
        return logits

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConvAudioDecisionNet(nn.Module):
    """Shared acoustic front end with either a spiking or nonspiking decision core."""

    def __init__(self, *, kind: str, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.kind = kind
        self.frontend = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.core = DecisionNet(kind=kind, n_actions=n_actions, input_dim=64, hidden=hidden)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None,
                return_activity: bool = False):
        features = self.frontend(x.transpose(1, 2)).transpose(1, 2)
        return self.core(features, lengths, return_activity)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GRUDecisionNet(nn.Module):
    """Capacity-matched gated recurrent control with the same temporal readout."""

    def __init__(self, *, n_actions: int, input_dim: int = 40,
                 vocab_size: int | None = None, hidden: int = 47,
                 embedding_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) if vocab_size else None
        self.input = nn.Linear(embedding_dim if vocab_size else input_dim, hidden)
        self.recurrent = nn.GRUCell(hidden, hidden)
        self.readout = nn.Linear(2 * hidden, n_actions)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None,
                return_activity: bool = False):
        if self.embedding is not None:
            lengths = (x != 0).sum(dim=1).clamp_min(1) if lengths is None else lengths
            x = self.embedding(x)
        batch, steps, _ = x.shape
        if lengths is None:
            lengths = torch.full((batch,), steps, dtype=torch.long, device=x.device)
        h = x.new_zeros(batch, self.hidden)
        state_sum = x.new_zeros(batch, self.hidden)
        for t in range(steps):
            active = (t < lengths).unsqueeze(1)
            next_h = self.recurrent(self.input(x[:, t]), h)
            h = torch.where(active, next_h, h)
            state_sum = state_sum + h * active
        mean_state = state_sum / lengths.unsqueeze(1)
        logits = self.readout(torch.cat((mean_state, h), dim=1))
        return (logits, None) if return_activity else logits

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
