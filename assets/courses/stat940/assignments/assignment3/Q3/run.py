"""Supplied fixed harness. Run from its parent directory: python -m Q3.run."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .core import forward_noising, noise_loss, sample_chain


seeds = (11, 22)
m = 100
updates = 1500
batch_size = 128


def make_schedule(dtype=torch.float32):
    """Return one-indexed (alpha, abar), each of shape (m+1,)."""
    beta = torch.linspace(0.0001, 0.15, m, dtype=dtype)
    alpha = torch.cat((torch.ones(1, dtype=dtype), 1 - beta))
    abar = torch.cumprod(alpha, dim=0)
    return alpha, abar


def centers():
    """Return the fixed component centers with shape (2,8)."""
    angles = 2 * np.pi * np.arange(8) / 8
    return 2 * np.vstack((np.cos(angles), np.sin(angles)))


def make_data(n, seed):
    """Draw labels first, then noise of shape (2,n), using NumPy default_rng."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 8, size=n)
    X = centers()[:, labels] + 0.08 * rng.standard_normal((2, n))
    return torch.tensor(X, dtype=torch.float32)


class ColumnLinear(nn.Module):
    """Transpose at the nn.Linear boundary; return features in columns."""

    def __init__(self, d, f):
        super().__init__()
        self.affine = nn.Linear(d, f)

    def forward(self, H):
        return self.affine(H.T).T


class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ColumnLinear(4, 64)
        self.layer2 = ColumnLinear(64, 64)
        self.layer3 = ColumnLinear(64, 2)

    def forward(self, Xt, t):
        phase = math.pi * t.to(dtype=Xt.dtype) / m
        H = torch.cat((Xt, phase.sin().unsqueeze(0), phase.cos().unsqueeze(0)), dim=0)
        H = torch.relu(self.layer1(H))
        H = torch.relu(self.layer2(H))
        return self.layer3(H)


def make_generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def make_evaluation(X0, abar):
    """One generator draws fixed held-out timesteps, then held-out noise."""
    generator = make_generator(4000)
    t = torch.randint(1, m + 1, (X0.shape[1],), generator=generator)
    E = torch.randn(X0.shape, generator=generator, dtype=X0.dtype)
    return forward_noising(X0, t, E, abar=abar), t, E


def train(seed, X0, abar):
    torch.manual_seed(seed)
    model = NoisePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    batch_generator = make_generator(seed + 1000)
    timestep_generator = make_generator(seed + 2000)
    noise_generator = make_generator(seed + 3000)
    history = []
    model.train()
    for update in range(1, updates + 1):
        indices = torch.randint(X0.shape[1], (batch_size,), generator=batch_generator)
        t = torch.randint(1, m + 1, (batch_size,), generator=timestep_generator)
        E = torch.randn((2, batch_size), generator=noise_generator, dtype=X0.dtype)
        Xt = forward_noising(X0[:, indices], t, E, abar=abar)
        optimizer.zero_grad(set_to_none=True)
        loss = noise_loss(model(Xt, t), E)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite training loss at seed {seed}, update {update}.")
        history.append({"seed": seed, "update": update, "training_noise_loss": value})
    return model, history


def sample_pair(model, seed, alpha, abar, n=1024):
    """Supply matched initial columns/noise to the student-written sampler."""
    initial = torch.randn((2, n), generator=make_generator(seed + 5000))
    reverse_generator = make_generator(seed + 6000)
    Z_steps = torch.zeros((m + 1, 2, n), dtype=initial.dtype)
    # Preserve the prescribed descending draw order; no noise is used at t=1.
    for t in range(m, 1, -1):
        Z_steps[t] = torch.randn(initial.shape, generator=reverse_generator,
                                dtype=initial.dtype)
    samples = {}
    for sampler, add_noise in (("stochastic", True), ("zero_noise", False)):
        X = sample_chain(model, initial.clone(), Z_steps.clone(), alpha=alpha,
                         abar=abar, add_noise=add_noise)
        if X.shape != (2, n) or not torch.isfinite(X).all():
            raise RuntimeError(f"Invalid samples for seed {seed}, sampler {sampler}.")
        samples[sampler] = X.numpy()
    return samples


def sample_metrics(X):
    """Return radial error, occupied sectors and counts for data of shape (2,n)."""
    X = np.asarray(X, dtype=np.float64)
    radial_error = float(np.mean(np.abs(np.linalg.norm(X, axis=0) - 2)))
    angles = np.arctan2(X[1], X[0])
    center_angles = 2 * np.pi * np.arange(8) / 8
    delta = angles[None, :] - center_angles[:, None]
    distances = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
    # Roundoff-sized ties choose the first (smallest) center index.
    nearest = distances <= distances.min(axis=0, keepdims=True) + 8 * np.finfo(float).eps
    sectors = np.argmax(nearest, axis=0)
    counts = np.bincount(sectors, minlength=8)
    occupied_modes = int(np.count_nonzero(counts >= 0.01 * X.shape[1]))
    return radial_error, occupied_modes, counts


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_samples(samples, output):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    limit = max(2.5, max(float(np.abs(X).max()) for X in samples.values()) + 0.15)
    C = centers()
    for row, seed in enumerate(seeds):
        for column, sampler in enumerate(("stochastic", "zero_noise")):
            ax = axes[row, column]
            X = samples[(seed, sampler)]
            ax.scatter(X[0], X[1], s=5, alpha=0.45, color="#225f93", rasterized=True)
            ax.scatter(C[0], C[1], marker="x", s=55, linewidths=1.5, color="#b63b2f", label="Centers")
            ax.set(title=f"Seed {seed}: {sampler.replace('_', ' ')}", xlim=(-limit, limit), ylim=(-limit, limit))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            if row == 1:
                ax.set_xlabel(r"$x_1$")
            if column == 0:
                ax.set_ylabel(r"$x_2$")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "samples.png", dpi=180)
    fig.savefig(output / "samples.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    started = time.perf_counter()
    alpha, abar = make_schedule()
    Xtrain = make_data(2048, 2026)
    Xheldout = make_data(512, 2027)
    data_radial_error = sample_metrics(Xheldout.numpy())[0]
    print(json.dumps({"held_out_data_radial_error": data_radial_error}), flush=True)
    Xt_eval, t_eval, E_eval = make_evaluation(Xheldout, abar)
    rows, training_rows, mode_rows, all_samples = [], [], [], {}
    for seed in seeds:
        model, history = train(seed, Xtrain, abar)
        training_rows.extend(history)
        model.eval()
        with torch.no_grad():
            heldout = float(noise_loss(model(Xt_eval, t_eval), E_eval))
        for sampler, X in sample_pair(model, seed, alpha, abar).items():
            radial, occupied, counts = sample_metrics(X)
            row = {"seed": seed, "sampler": sampler, "held_out_noise_loss": heldout,
                   "radial_error": radial, "occupied_modes": occupied,
                   "held_out_data_radial_error": data_radial_error}
            rows.append(row)
            mode_rows.append({"seed": seed, "sampler": sampler, **{f"sector_{j}": int(counts[j]) for j in range(8)}})
            all_samples[(seed, sampler)] = X
            print(json.dumps(row), flush=True)
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "training.csv", training_rows)
    write_csv(args.output / "mode_counts.csv", mode_rows)
    np.savez_compressed(args.output / "samples.npz", **{f"seed_{seed}_{sampler}": X for (seed, sampler), X in all_samples.items()})
    plot_samples(all_samples, args.output)
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "matplotlib": matplotlib.__version__, "torch_threads": torch.get_num_threads(),
                "dtype": "float32", "seeds": list(seeds), "updates": updates, "batch_size": batch_size,
                "held_out_data_radial_error": data_radial_error,
                "elapsed_seconds": time.perf_counter() - started}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
