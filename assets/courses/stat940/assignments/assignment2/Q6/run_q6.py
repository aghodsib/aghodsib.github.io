#!/usr/bin/env python3
"""STAT 940 A2 Q6: supplied CPU training, evaluation and plotting harness."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from data import CENTERS, make_data
from model import TAU, VAE, sample_latent, analytic_kl, loss_terms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#28589e", "#e58b22", "#3f9367", "#a64e94"]






def diagnostics():
    mu = torch.tensor([0.5, -0.5], dtype=torch.float32).expand(50_000, -1)
    logvar = torch.log(torch.tensor([0.25, 4.0], dtype=torch.float32)).expand(50_000, -1)
    draws = sample_latent(mu, logvar, torch.Generator(device="cpu").manual_seed(31415))
    means = draws.mean(dim=0)
    variances = draws.var(dim=0, unbiased=False)
    assert torch.all(torch.abs(means-torch.tensor([0.5, -0.5])) < 0.05)
    assert torch.all(torch.abs(variances/torch.tensor([0.25, 4.0])-1.0) < 0.05)
    assert torch.allclose(analytic_kl(torch.zeros(3, 2), torch.zeros(3, 2)), torch.zeros(3))
    known_kl = analytic_kl(mu, logvar)
    assert torch.allclose(known_kl, torch.full((50_000,), 1.375), rtol=1e-6, atol=1e-6)
    assert torch.isfinite(known_kl).all()
    assert (known_kl >= -1e-6).all()
    return {"empirical_means": means.tolist(), "empirical_population_variances": variances.tolist(),
            "target_means": [0.5, -0.5], "target_variances": [0.25, 4.0],
            "expected_analytic_kl": 1.375, "observed_analytic_kl": known_kl[0].item(),
            "all_assertions_passed": True}


def model_digest(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def train_one(X, seed, weight):
    torch.manual_seed(seed)
    model = VAE()
    initial_hash = model_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed+1000)
    latent_generator = torch.Generator(device="cpu").manual_seed(seed+2000)
    history = []
    order_digest = hashlib.sha256()
    start = time.perf_counter()
    model.train()
    for epoch in range(1, 101):
        permutation = torch.randperm(len(X), generator=batch_generator)
        order_digest.update(permutation.numpy().tobytes())
        sums = torch.zeros(2, dtype=torch.float64)
        for indices in permutation.split(128):
            batch = X[indices]
            reconstructed, mu, logvar = model(batch, latent_generator)
            rec, kl = loss_terms(batch, reconstructed, mu, logvar)
            assert rec.shape == kl.shape == (len(batch),)
            assert torch.isfinite(rec).all() and torch.isfinite(kl).all()
            assert (kl >= -1e-6).all()
            loss = (rec + weight*kl).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += torch.stack([rec.detach().sum(), kl.detach().sum()]).to(torch.float64)
        history.append({"seed": seed, "lambda": weight, "epoch": epoch,
                        "reconstruction": float(sums[0]/len(X)), "unweighted_kl": float(sums[1]/len(X))})
    training_seconds = time.perf_counter()-start
    audit = {"initial_model_sha256": initial_hash, "batch_order_sha256": order_digest.hexdigest(),
             "final_latent_generator_state_sha256": hashlib.sha256(latent_generator.get_state().numpy().tobytes()).hexdigest()}
    return model, history, training_seconds, audit


@torch.no_grad()
def evaluate(model, heldout, seed):
    model.eval()
    mu, logvar = model.encode(heldout)
    reconstruction = model.decoder(mu)
    mse = (heldout-reconstruction).square().mean().item()
    kl_per_example = analytic_kl(mu, logvar)
    assert (kl_per_example >= -1e-6).all() and torch.isfinite(kl_per_example).all()
    generator = torch.Generator(device="cpu").manual_seed(seed+3000)
    latent = torch.randn((2048, 2), generator=generator)
    noise = torch.randn((2048, 2), generator=generator)
    generated = (model.decoder(latent) + TAU*noise).numpy()
    assert np.isfinite(generated).all()
    distances = np.linalg.norm(generated[:, None, :] - CENTERS[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)  # first listed center wins an exact tie
    metrics = {"deterministic_mse": mse, "mean_analytic_kl": kl_per_example.mean().item(),
               "within_0.6_fraction": float((distances.min(axis=1) <= 0.6).mean())}
    for j in range(4):
        metrics[f"nearest_center_{j}_fraction"] = float((nearest == j).mean())
    metrics["outside_view_count"] = int(np.any((generated < -4) | (generated > 4), axis=1).sum())
    extra = {"generation_latent_sha256": hashlib.sha256(latent.numpy().tobytes()).hexdigest(),
             "generation_noise_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
             "heldout_logvar_min": logvar.min().item(), "heldout_logvar_max": logvar.max().item(),
             "heldout_logvar_at_lower_clamp_fraction": (logvar <= -8).float().mean().item(),
             "heldout_mu_absolute_max": mu.abs().max().item()}
    return metrics, generated, nearest, extra


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_scatter(ax, points, labels, title):
    for j in range(4):
        subset = points[labels == j]
        ax.scatter(subset[:, 0], subset[:, 1], s=6, alpha=0.35, color=COLORS[j], rasterized=True)
    ax.scatter(CENTERS[:, 0], CENTERS[:, 1], marker="X", s=100, c="black", edgecolors="white", linewidths=0.8)
    ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x_1$", ylabel="$x_2$", title=title)
    ax.set_aspect("equal", adjustable="box")


def make_plots(output, histories, heldout, heldout_labels, generated_runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for seed in [11, 22]:
        for weight in [1, 0]:
            rows = [r for r in histories if r["seed"] == seed and r["lambda"] == weight]
            style = "-" if weight == 1 else "--"
            color = "#28589e" if seed == 11 else "#e58b22"
            label = f"seed {seed}, lambda={weight}"
            for ax, key in zip(axes, ["reconstruction", "unweighted_kl"]):
                ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], style, c=color, label=label)
                ax.set_xlabel("Epoch")
                ax.legend(fontsize=8)
    axes[0].set(ylabel="Mean reconstruction term (no constant)", title="Training reconstruction")
    axes[1].set(ylabel="Mean unweighted KL", title="Training analytic KL")
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    format_scatter(ax, heldout.numpy(), heldout_labels, "Held-out data (N=512)")
    fig.savefig(output / "heldout_scatter.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout="constrained")
    for ax, (seed, weight, generated, nearest, outside) in zip(axes.flat, generated_runs):
        format_scatter(ax, generated, nearest, f"seed {seed}, lambda={weight}; outside={outside}")
    fig.savefig(output / "generated_scatter.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    script_start = time.perf_counter()
    sampling_audit = diagnostics()
    print("SAMPLING DIAGNOSTIC", json.dumps(sampling_audit), flush=True)
    training, _ = make_data(2048, 2026)
    heldout, heldout_labels = make_data(512, 2027)
    all_metrics, histories, audits, generated_runs = [], [], [], []
    for seed in [11, 22]:
        for weight in [1, 0]:
            model, history, training_seconds, audit = train_one(training, seed, weight)
            metrics, generated, nearest, extra = evaluate(model, heldout, seed)
            row = {"seed": seed, "lambda": weight, **metrics}
            all_metrics.append(row)
            histories.extend(history)
            audits.append({"seed": seed, "lambda": weight, "training_seconds": training_seconds, **audit, **extra})
            generated_runs.append((seed, weight, generated, nearest, metrics["outside_view_count"]))
            # Data, not checkpoints: facilitates independent numeric/plot verification.
            np.savetxt(args.output_dir / f"generated_seed{seed}_lambda{weight}.csv", generated,
                       delimiter=",", header="x1,x2", comments="")
            print("RUN COMPLETE", json.dumps(row), flush=True)
    for seed in [11, 22]:
        pair = [a for a in audits if a["seed"] == seed]
        for key in ["initial_model_sha256", "batch_order_sha256", "final_latent_generator_state_sha256",
                    "generation_latent_sha256", "generation_noise_sha256"]:
            assert pair[0][key] == pair[1][key], (seed, key)
    write_csv(args.output_dir / "metrics.csv", all_metrics)
    write_csv(args.output_dir / "epoch_history.csv", histories)
    make_plots(args.output_dir, histories, heldout, heldout_labels, generated_runs)
    metadata = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
                "thread_environment": {k: os.environ.get(k, "unset") for k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]},
                "wall_seconds_including_plots": time.perf_counter()-script_start}
    report = {"metadata": metadata, "sampling_diagnostic": sampling_audit, "metrics": all_metrics,
              "pairing_audits": audits, "pairing_assertions_passed": True}
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL REPORT", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
