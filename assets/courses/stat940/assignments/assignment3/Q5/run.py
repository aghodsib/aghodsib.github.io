"""Supplied fixed harness. Run from its parent directory: python -m Q5.run."""

import argparse
import copy
import csv
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import GraphLayer, normalized_adjacency


seeds = (11, 22)
epochs = 200


def make_data():
    """Make the single graph and train-only standardization in the stated order."""
    rng = np.random.default_rng(2026)
    n = 240
    y = np.repeat(np.arange(2), 120)
    upper = np.triu_indices(n, k=1)
    probabilities = np.where(y[upper[0]] == y[upper[1]], 0.12, 0.02)
    edges = rng.random(len(upper[0])) < probabilities
    A = np.zeros((n, n), dtype=np.float64)
    A[upper] = edges
    A += A.T
    Xraw = rng.standard_normal((8, n))
    Xraw[0] += 0.6 * (2 * y - 1)
    masks = {name: np.zeros(n, dtype=bool) for name in ("train", "validation", "test")}
    for label in range(2):
        order = rng.permutation(np.flatnonzero(y == label))
        masks["train"][order[:20]] = True
        masks["validation"][order[20:40]] = True
        masks["test"][order[40:]] = True
    mean = Xraw[:, masks["train"]].mean(axis=1, keepdims=True)
    std = Xraw[:, masks["train"]].std(axis=1, keepdims=True, ddof=0)
    X = (Xraw - mean) / std
    return {"A": torch.tensor(A, dtype=torch.float32), "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
            **{name: torch.tensor(mask) for name, mask in masks.items()},
            "raw_features": Xraw, "training_mean": mean, "training_std": std}


class GraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphLayer(8, 16, activation=torch.relu)
        self.layer2 = GraphLayer(16, 2, activation=None)

    def forward(self, X, S):
        return self.layer2(self.layer1(X, S), S)


class ColumnLinear(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.affine = nn.Linear(d, f)

    def forward(self, H):
        return self.affine(H.T).T


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ColumnLinear(8, 16)
        self.layer2 = ColumnLinear(16, 2)

    def forward(self, X, S=None):
        return self.layer2(torch.relu(self.layer1(X)))


def make_matched_models(seed):
    torch.manual_seed(seed)
    gcn = GraphClassifier()
    torch.manual_seed(seed)
    mlp = MLP()
    for graph_layer, affine_layer in ((gcn.layer1, mlp.layer1.affine), (gcn.layer2, mlp.layer2.affine)):
        assert torch.equal(graph_layer.W, affine_layer.weight), "Initial weights must match."
        assert torch.equal(graph_layer.b, affine_layer.bias), "Initial biases must match."
    return {"GCN": gcn, "MLP": mlp}


def cross_entropy_columns(Z, y):
    """PyTorch accepts (batch,classes,nodes); no node-feature transpose is needed."""
    return F.cross_entropy(Z.unsqueeze(0), y.unsqueeze(0))


def train(model, X, S, ytrain, yvalidation, train_mask, validation_mask, seed, name):
    """Only training/validation labels are passed here; no test labels are accepted."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        Z = model(X, S)
        loss = cross_entropy_columns(Z[:, train_mask], ytrain)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(cross_entropy_columns(model(X, S)[:, validation_mask], yvalidation))
        if not np.isfinite(validation) or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at seed {seed}, model {name}, epoch {epoch}.")
        # Strict inequality keeps the earliest epoch in an exact tie.
        if validation < best_loss:
            best_loss = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"seed": seed, "model": name, "epoch": epoch,
                        "training_crossentropy": float(loss.detach()),
                        "validation_crossentropy": validation})
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss, history


@torch.no_grad()
def test_accuracy(model, X, S, ytest, test_mask):
    Z = model(X, S)
    predicted = Z[:, test_mask].argmax(dim=0)
    return float((predicted == ytest).to(torch.float64).mean())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, seed in zip(axes, seeds):
        for name, color in (("GCN", "#225f93"), ("MLP", "#b63b2f")):
            selected = [row for row in rows if row["seed"] == seed and row["model"] == name]
            ax.plot([r["epoch"] for r in selected], [r["validation_crossentropy"] for r in selected], label=name, color=color)
        ax.set(title=f"Seed {seed}", xlabel="Epoch")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation crossentropy")
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=180)
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
    data = make_data()
    A, X, y = data["A"], data["X"], data["y"]
    S = normalized_adjacency(A)
    identity = torch.eye(A.shape[0], dtype=X.dtype)
    rows, interventions, history = [], [], []
    selected = []
    for seed in seeds:
        for name, model in make_matched_models(seed).items():
            model, best_epoch, best_loss, model_history = train(
                model, X, S, y[data["train"]], y[data["validation"]],
                data["train"], data["validation"], seed, name)
            selected.append((seed, name, model, best_epoch, best_loss))
            history.extend(model_history)
    # Test labels are accessed only after every prescribed checkpoint is selected.
    ytest = y[data["test"]]
    for seed, name, model, best_epoch, best_loss in selected:
        accuracy = test_accuracy(model, X, S, ytest, data["test"])
        row = {"seed": seed, "model": name, "best_epoch": best_epoch,
               "validation_crossentropy": best_loss, "test_accuracy": accuracy}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if name == "GCN":
            intervention = {"seed": seed, "model": "GCN_with_identity_at_inference",
                            "test_accuracy": test_accuracy(model, X, identity, ytest, data["test"])}
            interventions.append(intervention)
            print(json.dumps(intervention), flush=True)
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "identity_intervention.csv", interventions)
    write_csv(args.output / "training.csv", history)
    plot_history(history, args.output)
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "matplotlib": matplotlib.__version__, "torch_threads": torch.get_num_threads(),
                "dtype": "float32", "seeds": list(seeds), "epochs": epochs,
                "undirected_edges": int(A.sum().item() / 2),
                "mask_sizes": {key: int(data[key].sum()) for key in ("train", "validation", "test")},
                "elapsed_seconds": time.perf_counter() - started}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
