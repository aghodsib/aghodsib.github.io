"""Supplied NumPy model mechanics, not an experiment solution.

All observations are columns, all arithmetic after generation is float64.
Architecture: affine 3->16, optional per-feature BatchNorm, tanh, affine 16->1.
BatchNorm uses epsilon=1e-5, population batch variance (denominator b),
running <- .9*running + .1*batch for BOTH mean and variance, initially 0 and 1.
This explicit educational convention is not PyTorch's unbiased running variance.
No derivative implementation is required from students for Question 7.
"""
from copy import deepcopy
import json
import numpy as np


def affine_initial_state(seed):
    """Independent normal weights with variance 1/fan-in and zero biases."""
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0.0, 1 / np.sqrt(3), (16, 3)),
            "b1": np.zeros((16, 1)),
            "W2": rng.normal(0.0, 1 / np.sqrt(16), (1, 16)),
            "b2": np.zeros((1, 1))}


def binary_metrics(logits, y):
    """Numerically stable mean BCE and accuracy; logit zero predicts label 1."""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if z.shape != target.shape:
        raise ValueError("The number of logits must equal the number of labels")
    loss = np.mean(np.logaddexp(0.0, z) - target * z)
    accuracy = np.mean((z >= 0.0) == target)
    return float(loss), float(accuracy)


class MLP:
    def __init__(self, seed=11, batchnorm=False, initial_affine=None,
                 epsilon=1e-5, running_decay=0.9):
        self.batchnorm = bool(batchnorm)
        self.epsilon = float(epsilon)
        self.running_decay = float(running_decay)
        if self.epsilon <= 0 or not 0 <= self.running_decay < 1:
            raise ValueError("Require epsilon > 0 and 0 <= running_decay < 1")
        base = affine_initial_state(seed) if initial_affine is None else initial_affine
        shapes = {"W1": (16, 3), "b1": (16, 1),
                  "W2": (1, 16), "b2": (1, 1)}
        self.params = {}
        for name, shape in shapes.items():
            value = np.asarray(base[name], dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"Incorrect shape for {name}: {value.shape}")
            self.params[name] = value.copy()
        if self.batchnorm:
            self.params.update(gamma=np.ones((16, 1)), beta=np.zeros((16, 1)))
        self.running_mean = np.zeros((16, 1))
        self.running_var = np.ones((16, 1))
        self.batches_seen = 0

    def _forward(self, x, training, update_running):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != 3 or x.shape[1] == 0:
            raise ValueError("Input must have shape (3, nonzero batch size)")
        p = self.params
        a = p["W1"] @ x + p["b1"]
        cache = {"x": x}
        if self.batchnorm:
            if training:
                mean = a.mean(axis=1, keepdims=True)
                var = ((a - mean) ** 2).mean(axis=1, keepdims=True)
                if update_running:
                    decay = self.running_decay
                    self.running_mean = decay * self.running_mean + (1 - decay) * mean
                    self.running_var = decay * self.running_var + (1 - decay) * var
                    self.batches_seen += 1
            else:
                mean, var = self.running_mean, self.running_var
            invstd = 1.0 / np.sqrt(var + self.epsilon)
            xhat = (a - mean) * invstd
            z = p["gamma"] * xhat + p["beta"]
            cache.update(xhat=xhat, invstd=invstd)
        else:
            z = a
        h = np.tanh(z)
        logits = p["W2"] @ h + p["b2"]
        cache["h"] = h
        return logits, cache

    def forward(self, x, training=False, update_running=True):
        """Explicit mode. Eval does not change buffers; train optionally does."""
        return self._forward(x, bool(training), bool(update_running))[0]

    def loss_and_grad(self, x, y, update_running=True):
        """Return TRAIN-mode mean BCE and gradients, without a parameter update.

        The default updates BN buffers exactly once. Set update_running=False
        for numerical derivative checks. Gradients include averaging over b.
        """
        logits, cache = self._forward(x, training=True, update_running=update_running)
        loss, _ = binary_metrics(logits, y)
        target = np.asarray(y, dtype=np.float64).reshape(1, -1)
        b = target.shape[1]
        sigmoid = np.exp(-np.logaddexp(0.0, -logits))
        dlogit = (sigmoid - target) / b
        p = self.params
        grads = {"W2": dlogit @ cache["h"].T,
                 "b2": dlogit.sum(axis=1, keepdims=True)}
        dz = (p["W2"].T @ dlogit) * (1 - cache["h"] ** 2)
        if self.batchnorm:
            xhat = cache["xhat"]
            grads["gamma"] = (dz * xhat).sum(axis=1, keepdims=True)
            grads["beta"] = dz.sum(axis=1, keepdims=True)
            dxhat = dz * p["gamma"]
            da = cache["invstd"] * (
                dxhat - dxhat.mean(axis=1, keepdims=True)
                - xhat * (dxhat * xhat).mean(axis=1, keepdims=True))
        else:
            da = dz
        grads["W1"] = da @ cache["x"].T
        grads["b1"] = da.sum(axis=1, keepdims=True)
        return loss, grads

    def state_dict(self):
        """Independent copies include all learned parameters AND BN buffers."""
        return {"params": deepcopy(self.params), "batchnorm": self.batchnorm,
                "epsilon": self.epsilon, "running_decay": self.running_decay,
                "running_mean": self.running_mean.copy(),
                "running_var": self.running_var.copy(), "batches_seen": self.batches_seen}

    def load_state_dict(self, state):
        if bool(state["batchnorm"]) != self.batchnorm:
            raise ValueError("Checkpoint BatchNorm configuration does not match")
        if set(state["params"]) != set(self.params):
            raise ValueError("Checkpoint parameter names do not match")
        for key, value in state["params"].items():
            if value.shape != self.params[key].shape:
                raise ValueError(f"Checkpoint shape mismatch for {key}")
        self.params = deepcopy(state["params"])
        for key in ("epsilon", "running_decay", "batches_seen"):
            setattr(self, key, state[key])
        self.running_mean = state["running_mean"].copy()
        self.running_var = state["running_var"].copy()

    def clone(self):
        other = MLP(batchnorm=self.batchnorm)
        other.load_state_dict(self.state_dict())
        return other

    def all_finite(self):
        arrays = list(self.params.values()) + [self.running_mean, self.running_var]
        return all(np.isfinite(a).all() for a in arrays)


class SGD:
    """Classical momentum: v <- beta*v + gradient; theta <- theta - lr*v.

    Set beta=0 for plain SGD. Initial velocity is zero. No dampening,
    Nesterov acceleration, weight decay, or (1-beta) gradient factor is used.
    """
    def __init__(self, model, lr, beta=0.0):
        self.lr, self.beta = float(lr), float(beta)
        if self.lr <= 0 or not 0 <= self.beta < 1:
            raise ValueError("Require lr > 0 and 0 <= beta < 1")
        self.velocity = {k: np.zeros_like(v) for k, v in model.params.items()}

    def step(self, model, grads):
        if set(grads) != set(model.params):
            raise ValueError("Every model parameter requires exactly one gradient")
        for key in model.params:
            self.velocity[key] *= self.beta
            self.velocity[key] += grads[key]
            model.params[key] -= self.lr * self.velocity[key]

    def state_dict(self):
        return {"lr": self.lr, "beta": self.beta, "velocity": deepcopy(self.velocity)}

    def load_state_dict(self, state):
        if set(state["velocity"]) != set(self.velocity):
            raise ValueError("Optimizer checkpoint parameter names do not match")
        self.lr, self.beta = float(state["lr"]), float(state["beta"])
        self.velocity = deepcopy(state["velocity"])


def save_checkpoint(path, model, optimizer=None, metadata=None):
    """Write a safe .npz checkpoint; optional optimizer enables exact resume."""
    state = model.state_dict()
    payload = {"param_" + k: v for k, v in state.pop("params").items()}
    payload["running_mean"] = state.pop("running_mean")
    payload["running_var"] = state.pop("running_var")
    payload["model_config"] = np.array(json.dumps(state))
    payload["metadata"] = np.array(json.dumps(metadata or {}))
    if optimizer is not None:
        opt = optimizer.state_dict()
        payload.update({"velocity_" + k: v for k, v in opt.pop("velocity").items()})
        payload["optimizer_config"] = np.array(json.dumps(opt))
    np.savez_compressed(path, **payload)


def load_checkpoint(path):
    """Return (model, optimizer_or_None, metadata), without pickle loading."""
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["model_config"]))
        model = MLP(batchnorm=config["batchnorm"])
        state = dict(config)
        state["params"] = {k[6:]: data[k].copy() for k in data.files if k.startswith("param_")}
        state["running_mean"] = data["running_mean"].copy()
        state["running_var"] = data["running_var"].copy()
        model.load_state_dict(state)
        optimizer = None
        if "optimizer_config" in data:
            config = json.loads(str(data["optimizer_config"]))
            optimizer = SGD(model, **config)
            config["velocity"] = {k[9:]: data[k].copy() for k in data.files
                                  if k.startswith("velocity_")}
            optimizer.load_state_dict(config)
        metadata = json.loads(str(data["metadata"]))
    return model, optimizer, metadata
