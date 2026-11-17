# STAT 940 Assignment 3 practical starter

Complete the four diffusion functions in `Q3/core.py` and the two graph operations
in `Q5/core.py`. Write your own `Q3/checks.py` and
`Q5/checks.py`; no test answers or completed checks are supplied. The assignment
states the required numerical checks and written deliverables. Use the fixed
`run.py` harnesses without changing the data, random seeds, architecture,
optimization, checkpoint selection, or metrics. Q3 generation uses your own
`sample_chain` function. Do not tune hyperparameters.

Before submitting, replace or edit this README to document the dependencies and
commands you actually used. Include any information needed to reproduce your
work. Both practical questions belong in one `WatIAM_A3.zip` with `Q3/`, `Q5/`,
and your README at its top level. Replace `WatIAM` with your identifier. Exclude
virtual environments, installed packages, cache directories, and checkpoints.

## Environment and commands

The harness was verified on CPU with Python 3.12.14, PyTorch 2.15.0+cpu,
NumPy 2.3.5, and Matplotlib 3.10.8. It uses one Torch thread and float32 for
training. Write the numerical checks using float64 and `rtol=atol=1e-6`.
The code uses standard PyTorch tensor and `torch.nn` APIs. If using a different
PyTorch version, record its exact version; small numerical differences can occur.

One possible installation in a fresh environment is:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On Windows, activate with `.venv\Scripts\activate` instead. Run the following
commands from the directory containing this README. The checks commands work
after you have written the respective files.

```sh
python -m Q3.checks
python -m Q3.run
python -m Q5.checks
python -m Q5.run
```

The harnesses write to `Q3/outputs/` and `Q5/outputs/`. To change only the output
directory, use `python -m Q3.run --output path/to/results` or the analogous Q5
command. There is deliberately no option to change a training seed or budget.

Q3 produces `metrics.csv`, `mode_counts.csv`, `training.csv`, `samples.png`,
`samples.pdf`, `samples.npz`, and `run_metadata.json`. Use `samples.png` or
`samples.pdf` for the required 2-by-2 figure. Q5 produces `metrics.csv`,
`identity_intervention.csv`, `training.csv`, `validation_curves.png`, and
`run_metadata.json`. For submission, the required output files are Q3 `outputs/metrics.csv` and
`outputs/samples.png`, plus Q5 `outputs/metrics.csv` and
`outputs/identity_intervention.csv`. Other generated files are optional.
The Q3 metrics CSV also includes `held_out_data_radial_error`, repeated across
rows; report this value below the four-row results table. The Q5 validation
plot is an optional aid to inspection.
No trained model checkpoint is written. Copy the required tables and figure
into your matching Crowdmark answer; uploading only the code does not replace
the written answer.

## Data layout and core APIs

All observations and node features are columns. The harness transposes at
`nn.Linear` boundaries and returns columns. Crossentropy uses a singleton batch
dimension to retain column-wise logits. Scalar counts, dimensions, and indices
use lowercase names in the mathematical statements.

All core inputs and outputs are `torch.Tensor` objects unless stated otherwise.
Keep their dtype and device, and use Torch operations so gradients are retained
where required for training.
The functions should work in float32 for the experiment and float64 for checks.

| Function or method | Inputs | Output |
|---|---|---|
| `forward_noising(X0, t, E, *, abar)` | `X0`, `E`: `(2,b)`; `t`: long tensor `(b,)`; `abar`: `(m+1,)` | Noised data `(2,b)` |
| `noise_loss(Ehat, E)` | Both `(2,b)` | Scalar tensor, sum over coordinates then mean over examples |
| `reverse_step(Xt, t, Ehat, Z, add_noise=True, *, alpha, abar)` | Data/noise `(2,b)`; `t`: Python integer; schedules `(m+1,)` | Previous-step data `(2,b)` |
| `sample_chain(model, X_start, Z_steps, *, alpha, abar, add_noise=True)` | `X_start`: `(2,n)`; `Z_steps`: `(m+1,2,n)`; schedules `(m+1,)` | Final generated samples `(2,n)` |
| `normalized_adjacency(A)` | Symmetric zero-diagonal adjacency `(n,n)` | Normalized adjacency `(n,n)` |
| `GraphLayer(d, f, activation=None)` | Input/output feature counts; optional activation | Layer with `.W` of shape `(f,d)` and `.b` of shape `(f,)` |
| `GraphLayer.forward(H, S)` | `H`: `(d,n)`; `S`: `(n,n)` | Features `(f,n)` |

The schedule arrays include an index-zero entry, with `alpha[0]=abar[0]=1`.
In Q3 checks, pass small artificial schedule tensors rather than modifying
the fixed experimental schedule. For example, for a reverse check at `t=2`,
the required current and previous cumulative coefficients go in `abar[2]`
and `abar[1]`, respectively. The function obtains `beta[t]` as `1-alpha[t]`.
The arrays of supplied noise are explicit arguments. No core function should
draw randomness. At `t=1`, the reverse step returns the mean for either value
of `add_noise`.

Import the actual implementations in your checks:

```python
import torch
from Q3.core import forward_noising, noise_loss, reverse_step, sample_chain
from Q5.core import normalized_adjacency, GraphLayer
```

For the sampling loop, the harness supplies all initial and per-step noise.
Entry `Z_steps[0]` is never read; entry `Z_steps[1]` has no effect on the update.
The supplied model is a `torch.nn.Module`. Pass a same-device
`torch.long` timestep tensor of shape `(n,)` to the model at each step, and
use the current updated sample columns for every model call. Preserve the input
tensors, use evaluation mode and disable gradient recording. Do not draw random
values inside `sample_chain`.

For its two-step check, use a small `torch.nn.Module` that returns
`0.5 * X` from `forward(X, times)` and records each timestep tensor it receives. Use a fresh
instance for each noise setting so each history should be `[2, 1]`. Derive the
expected final samples by applying the two reverse steps by hand; compare your
actual function against those values, using full precision in assertions.
Include the test results in the same Q3 check table.

For a graph check, construct `GraphLayer(d, f, activation=None).double()`.
Use `with torch.no_grad():` and `.W.copy_(...)`, `.b.copy_(...)` to assign the
given check parameters. Use float64 tensors for those assignments. Parameter
creation and initialization are supplied so the experimental models begin
with exactly matching weights and biases.

## Fixed Q3 protocol

The only model seeds are 11 and 22. Data use NumPy `default_rng`: labels are
drawn first, then a `(2,n)` standard normal array. The training and held-out
sets use 2048/512 observations and seeds 2026/2027. The radius-2 mixture has
eight equally weighted components with standard deviation 0.08 per coordinate.

The schedule has 100 steps and linearly spaced beta values from 0.0001 to 0.15.
The supplied `4 -> 64 -> 64 -> 2` network uses ReLU hidden activations and
time features `sin(pi*t/m)` and `cos(pi*t/m)`. Initialization is the default
`nn.Linear` initialization after resetting the model seed. Adam uses learning
rate 0.001, zero weight decay, 1500 updates, and batch size 128 with replacement.

Independent CPU `torch.Generator` streams use `seed+1000` for batch indices,
`seed+2000` for timesteps, and `seed+3000` for training noise. Evaluation uses
one CPU Torch generator seeded 4000, drawing all held-out timesteps first and
then the `(2,512)` noise array. This same fixed evaluation is used for both
trained models and is never used to select a checkpoint. The final update's
model is evaluated.

Each sampler starts with the same 1024 latent columns drawn using `seed+5000`.
The harness draws reverse noise with `seed+6000` in descending step order,
from `m` through `2`, and stores it in `Z_steps`. It calls your `sample_chain`
with separate copies of the same initial columns and noise for the two settings.
The zero-noise variant suppresses noise addition. Thus only reverse noise changes
within a seed. The held-out loss is evaluated once per
trained model and repeated for its two samplers. The harness counts angular
sectors with at least 1% of samples and resolves angular ties to the smaller
sector index. Its plots use identical axes and equal aspect ratios.

## Fixed Q5 protocol

One NumPy `default_rng(2026)` draws the graph, features, and masks in that order.
Labels are 120 zeros followed by 120 ones. It draws one uniform for each strict
upper-triangular pair in `np.triu_indices` order, with edge probabilities 0.12
within a class and 0.02 across classes, then symmetrizes. It next draws features
of shape `(8,240)`. Before standardization, the first coordinate has mean -0.6
for class 0 and +0.6 for class 1, with unit variance. The remaining seven
coordinates are independent standard normals.
It finally permutes each class once, assigning the first 20 nodes to training,
next 20 to validation, and last 80 to testing. Each feature is standardized
using only training-node mean and population standard deviation.

For each seed 11 and 22, the MLP and GCN begin with exactly the same affine
weights and biases in the supplied `8 -> 16 -> 2` architectures. The GCN uses
the full normalized adjacency; the MLP uses its identity-adjacency equivalent.
Both train full-batch for 200 epochs with Adam, learning rate 0.01, zero weight
decay, ReLU, and no dropout. Only training-node labels enter the training loss.

Validation crossentropy is computed after each epoch's optimizer update.
Epochs are numbered 1 through 200. Select the lowest validation crossentropy,
retaining the earliest epoch in a tie. Test labels are used only after
checkpoint selection for the prescribed final evaluations. For the additional
GCN intervention, keep the selected weights fixed and replace the adjacency
with identity only at inference. Do not train or select another checkpoint.

## Your numerical checks and write-up

Write checks that call the actual functions, compare against your hand
calculations, assert the specified shapes, and print expected/observed values
and pass/fail. The provided task statements specify the cases to include.
Do not replace your functions with separate formulas inside the checks.
Include the compact test tables and calculations, experimental tables,
required Q3 plot, and concise interpretations in Crowdmark. Interpret the
observed outputs; no particular ranking of methods is required.
