#!/usr/bin/env python3
"""Shared setup for the SGD stability-regime experiment.

Everything here is bit-faithful to the eoss `config.py` path that produced
`marc_optimizer_sweep_fixed_u_stride1/*_SGD_lr0.02_b4096`: same dataset subset,
same init, same epoch shuffling, same `SquaredLoss`. Nothing in this file
re-implements the model or the loss.

The scalar observables are the gain along a frozen direction q,

    h_B = q^T H_B q ,      a_B = 1 - eta * h_B ,

and the batch-gradient gain 1 - eta*s_B, where
s_B = g_B^T H_B g_B / ||g_B||^2. These define scalar surrogate products at a
frozen theta. A general network perturbation rotates under (I - eta*H_B), so
neither scalar surrogate is automatically the full network Lyapunov exponent.
"""
from __future__ import annotations

import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EOSS = Path("/accounts/projects/binyu/ghoshavr/eoss")
if str(EOSS) not in sys.path:
    sys.path.insert(0, str(EOSS))

os.environ.setdefault("DATASETS", "/scratch/users/ghoshavr/datasets")
os.environ.setdefault("RESULTS", "/scratch/users/ghoshavr/eoss/results")

# utils.nets imports timm at module scope for ViT. Use the real package when
# present; only stub it so the MLP path stays importable without timm.
if "timm" not in sys.modules:
    try:
        import timm  # noqa: F401
    except ImportError:
        _timm = types.ModuleType("timm")

        def _no_timm(*_a, **_k):
            raise RuntimeError("timm is stubbed; only the MLP path is supported here")

        _timm.create_model = _no_timm
        sys.modules["timm"] = _timm

from utils.data import get_dataset_presets, prepare_dataset  # noqa: E402
from utils.nets import (  # noqa: E402
    SquaredLoss,
    get_model_presets,
    initialize_net,
    prepare_net,
)
from utils.optimizer import create_optimizer  # noqa: E402


@dataclass(frozen=True)
class Cfg:
    """Marc CIFAR-10 MLP run, SGD eta=0.02, half batch."""

    eta: float = 0.02
    batch: int = 4096
    n_train: int = 8192
    steps: int = 40000
    seed: int = 88881
    dataset_seed: int = 888
    init_seed: int = 8888
    order_seed: int = 42
    init_scale: float = 0.2

    @property
    def edge(self) -> float:
        """Deterministic stability threshold: h_B above this expands."""
        return 2.0 / self.eta

    @property
    def root(self) -> Path:
        return Path("/scratch/users/ghoshavr/eoss/results/marc_repro_lyap/stability_regimes")


CFG = Cfg()

# Architecture registry for the cross-architecture confirmation sweep. All entries
# are CIFAR-10 + SquaredLoss and all are BatchNorm-free. BN would require
# batch-aware Hessian measurements: splitting a batch changes its loss, although
# a batch-specific scalar gain remains well-defined.
#
# lam_chunk is the chunk size for the full-batch power iteration. 0 means the
# whole training set fits in one Hessian-vector product; conv/attention models
# need it split or the double backward OOMs.
ARCHS: dict[str, dict] = {
    "mlp_d1":   dict(preset="mlp_s",    init_scale=0.2, eta=0.02,  lam_chunk=0),
    "mlp_d2":   dict(preset="mlp",      init_scale=0.2, eta=0.02,  lam_chunk=0),
    "mlp_d3":   dict(preset="mlp3",     init_scale=0.2, eta=0.02,  lam_chunk=0),
    "mlp_d4":   dict(preset="mlp_l",    init_scale=0.2, eta=0.02,  lam_chunk=0),
    "mlp_tanh": dict(preset="mlp_tanh", init_scale=0.2, eta=0.02,  lam_chunk=0),
    "cnn":      dict(preset="cnn",      init_scale=0.2, eta=0.005, lam_chunk=1024),
    "resnet20": dict(preset="resnet",   init_scale=0.2, eta=0.005, lam_chunk=512),
    "vit":      dict(preset="vit",      init_scale=0.2, eta=0.005, lam_chunk=512),
    "mlp_silu": dict(preset="mlp_silu", init_scale=0.2, eta=0.02, lam_chunk=0),
    "cnn_silu": dict(preset="cnn_silu", init_scale=0.2, eta=0.08, lam_chunk=256),
    "wrn_no_bn": dict(preset="wrn_no_bn", init_scale=1.0, eta=0.001,
                       lam_chunk=128, initialization="native_fixup"),
    "resnet32": dict(preset="resnet32", init_scale=0.2, eta=0.05, lam_chunk=128),
}

ARCH_LABEL = {
    "mlp_d1": "MLP-1x256", "mlp_d2": "MLP-2x512", "mlp_d3": "MLP-3x256",
    "mlp_d4": "MLP-4x512", "mlp_tanh": "MLP-2x512 tanh", "cnn": "CNN-3conv",
    "resnet20": "ResNet-20 (no BN)", "vit": "ViT-2x64",
    "mlp_silu": "MLP-2x512 SiLU", "cnn_silu": "CNN-3conv SiLU",
    "wrn_no_bn": "WideResNet-10-2 (Fixup, no BN)",
    "resnet32": "ResNet-32 (no BN)",
}


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(cfg: Cfg, device: torch.device):
    """Dataset + model + loss + optimizer, identical to eoss config.py."""
    X, Y, _, _ = prepare_dataset(
        "cifar10", Path(os.environ["DATASETS"]), cfg.n_train, [], cfg.dataset_seed, loss_type="mse"
    )
    X = X.to(device)
    Y = Y.to(device)

    mp = get_model_presets()
    dp = get_dataset_presets()
    params = dict(mp["mlp"]["params"])
    params["input_dim"] = dp["cifar10"]["input_dim"]
    params["output_dim"] = dp["cifar10"]["output_dim"]
    net = prepare_net(model_type=mp["mlp"]["type"], params=params)
    initialize_net(net, scale=cfg.init_scale, seed=cfg.init_seed)
    net = net.to(device).train().float()

    loss_fn = SquaredLoss()
    optimizer = create_optimizer("SGD", net, cfg.eta, {})
    return X, Y, net, loss_fn, optimizer


def build_arch(
    arch: str,
    device: torch.device,
    *,
    eta: float,
    n_train: int = CFG.n_train,
    dataset_seed: int = CFG.dataset_seed,
    init_seed: int = CFG.init_seed,
    init_scale: float | None = None,
):
    """Same dataset/loss/optimizer path as `build`, for any registered arch.

    `build` is kept as-is so the original MLP results stay bit-reproducible; this
    is the general entry point used by the cross-architecture sweep.
    """
    if arch not in ARCHS:
        raise KeyError(f"unknown arch {arch!r}; known: {sorted(ARCHS)}")
    spec = ARCHS[arch]
    scale = spec["init_scale"] if init_scale is None else init_scale
    if spec.get("initialization") == "native_fixup" and scale != 1.0:
        raise ValueError("wrn_no_bn uses native Fixup; init_scale must be 1.0")

    X, Y, _, _ = prepare_dataset(
        "cifar10", Path(os.environ["DATASETS"]), n_train, [], dataset_seed, loss_type="mse"
    )
    X = X.to(device)
    Y = Y.to(device)

    mp = get_model_presets()
    dp = get_dataset_presets()
    params = dict(mp[spec["preset"]]["params"])
    params["input_dim"] = dp["cifar10"]["input_dim"]
    params["output_dim"] = dp["cifar10"]["output_dim"]
    net = prepare_net(model_type=mp[spec["preset"]]["type"], params=params)
    initialize_net(net, scale=scale, seed=init_seed)
    net = net.to(device).train().float()

    loss_fn = SquaredLoss()
    optimizer = create_optimizer("SGD", net, eta, {})
    return X, Y, net, loss_fn, optimizer


def n_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _flat_grad(output, params, *, create_graph=False, retain_graph=None):
    """Flatten derivatives, representing genuinely unused parameters by zeros."""
    if not output.requires_grad:
        return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
    grads = torch.autograd.grad(output, params, create_graph=create_graph,
                                retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(params, grads)])


def hvp(net, loss_fn, Xb, Yb, vec: torch.Tensor) -> torch.Tensor:
    """Exact Hessian-vector product of the mini-batch loss (double backward)."""
    params = [p for p in net.parameters() if p.requires_grad]
    loss = loss_fn(net(Xb).squeeze(dim=-1), Yb)
    gflat = _flat_grad(loss, params, create_graph=True)
    return _flat_grad(torch.dot(gflat, vec.detach()), params).detach()


def hvp_chunked(net, loss_fn, X, Y, vec: torch.Tensor, chunk: int = 0) -> torch.Tensor:
    """Exact mean-loss HVP accumulated over samples, valid without batch coupling."""
    n = len(X)
    if chunk <= 0 or n <= chunk:
        return hvp(net, loss_fn, X, Y, vec)
    total = torch.zeros_like(vec)
    for i in range(0, n, chunk):
        xb, yb = X[i : i + chunk], Y[i : i + chunk]
        total.add_(hvp(net, loss_fn, xb, yb, vec), alpha=len(xb) / n)
    return total


def rayleigh(net, loss_fn, Xb, Yb, q: torch.Tensor) -> float:
    """h_B = q^T H_B q for a unit q."""
    return float(torch.dot(q, hvp(net, loss_fn, Xb, Yb, q)).item())


def grad_chunked(net, loss_fn, X, Y, chunk: int = 0) -> torch.Tensor:
    """Detached gradient of the mean loss over (X, Y), accumulated in chunks."""
    params = [p for p in net.parameters() if p.requires_grad]
    n = len(X)
    if chunk <= 0 or n <= chunk:
        loss = loss_fn(net(X).squeeze(dim=-1), Y)
        return _flat_grad(loss, params).detach()
    total = torch.zeros(sum(p.numel() for p in params), device=X.device,
                        dtype=params[0].dtype)
    for i in range(0, n, chunk):
        xb, yb = X[i : i + chunk], Y[i : i + chunk]
        loss = loss_fn(net(xb).squeeze(dim=-1), yb) * (len(xb) / n)
        total += _flat_grad(loss, params).detach()
    return total


def gradient_curvature_sample(net, loss_fn, Xb, Yb, chunk: int = 0):
    """Return s_B, detached g_B, and H_B g_B; no denominator clipping.

    A zero gradient makes its normalized direction undefined and yields NaN s_B.
    This is recorded explicitly instead of fabricating a multiplier of one.
    """
    if chunk <= 0 or len(Xb) <= chunk:
        params = [p for p in net.parameters() if p.requires_grad]
        loss = loss_fn(net(Xb).squeeze(dim=-1), Yb)
        g = _flat_grad(loss, params, create_graph=True)
        gd = g.detach()
        hg = _flat_grad(torch.dot(g, gd), params).detach()
    else:
        gd = grad_chunked(net, loss_fn, Xb, Yb, chunk)
        hg = hvp_chunked(net, loss_fn, Xb, Yb, gd, chunk)
    gg = float(torch.dot(gd, gd).item())
    s_B = float(torch.dot(gd, hg).item()) / gg if gg > 0 else float("nan")
    return s_B, gd, hg


def gain_sample(net, loss_fn, Xb, Yb, q: torch.Tensor | None, chunk: int = 0,
                *, return_details: bool = False, eta: float | None = None):
    """Return (s_B, h_B, g_B) on a frozen checkpoint and fresh mini-batch.

    Optional details include the exact one-step norm gain along q_B=g_B/||g_B||.
    The norm includes rotation out of q_B; the signed projection is 1-eta*s_B.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    if chunk <= 0 or len(Xb) <= chunk:
        loss = loss_fn(net(Xb).squeeze(dim=-1), Yb)
        g = _flat_grad(loss, params, create_graph=True)
        gd = g.detach()
        hq = (_flat_grad(torch.dot(g, q.detach()), params, retain_graph=True).detach()
              if q is not None else None)
        hg = _flat_grad(torch.dot(g, gd), params).detach()
    else:
        gd = grad_chunked(net, loss_fn, Xb, Yb, chunk)
        hq = hvp_chunked(net, loss_fn, Xb, Yb, q, chunk) if q is not None else None
        hg = hvp_chunked(net, loss_fn, Xb, Yb, gd, chunk)
    gg = float(torch.dot(gd, gd).item())
    s_B = float(torch.dot(gd, hg).item()) / gg if gg > 0 else float("nan")
    h_B = float(torch.dot(q, hq).item()) if q is not None else float("nan")
    if not return_details:
        return s_B, h_B, gd
    hgnorm_sq = float(torch.dot(hg, hg).item()) / gg if gg > 0 else float("nan")
    details = {"grad_sq": gg, "h_grad_unit_norm_sq": hgnorm_sq,
               "orthogonal_h_grad_norm_sq": (max(0.0, hgnorm_sq - s_B ** 2)
                                                if np.isfinite(hgnorm_sq) and np.isfinite(s_B)
                                                else float("nan"))}
    if eta is not None:
        details["a_B"] = 1.0 - eta * s_B
        details["step_norm_gain"] = (float((gd - eta * hg).norm().item()) / gg ** .5
                                      if gg > 0 else float("nan"))
    return s_B, h_B, gd, details


def batch_sharpness(net, loss_fn, X, Y, batch: int, n_probe: int, gen,
                    *, return_details: bool = False):
    """Fresh-batch E[s_B] (mean of ratios), with weighted variant optional.

    The weighted ratio E[g_B^T H_B g_B]/E[||g_B||^2] is returned separately as
    batch_sharpness_weighted. It is not the paper's default batch sharpness.
    Even for unweighted E[s_B], A1=eta*E[s_B]-1 additionally requires a_B<=0.
    """
    return batch_sharpness_chunked(net, loss_fn, X, Y, batch, n_probe, gen,
                                   return_details=return_details)


def batch_sharpness_chunked(net, loss_fn, X, Y, batch: int, n_probe: int, gen,
                            chunk: int = 0, *, return_details: bool = False):
    """Mean of per-batch Rayleigh quotients; separately retain weighted estimate."""
    if not 1 <= batch <= len(X) or n_probe < 1:
        raise ValueError("batch must be in [1,n_train] and n_probe must be positive")
    num, den = 0.0, 0.0
    values, gradient_norms = [], []
    for _ in range(n_probe):
        idx = torch.randperm(len(X), generator=gen)[:batch].to(X.device)
        s_B, gd, hg = gradient_curvature_sample(net, loss_fn, X[idx], Y[idx], chunk)
        gg = float(torch.dot(gd, gd).item())
        num += float(torch.dot(gd, hg).item())
        den += gg
        values.append(s_B)
        gradient_norms.append(gg ** .5)
    mean = float(np.mean(values))
    if not return_details:
        return mean
    return {"batch_sharpness": mean,
            "batch_sharpness_weighted": num / den if den > 0 else float("nan"),
            "batch_sharpness_se": float(np.std(values, ddof=1) / n_probe ** .5)
                                   if n_probe > 1 else float("nan"),
            "s_B": np.asarray(values), "g_norm": np.asarray(gradient_norms),
            "undefined_gradient_count": int(np.sum(np.asarray(gradient_norms) == 0))}


def power_iterate(net, loss_fn, X, Y, q0=None, n_iter: int = 40, tol: float = 0.0,
                  chunk: int = 0):
    """Dominant-magnitude Hessian eigenpair by warm-started power iteration.

    This is the largest algebraic eigenvalue only when that eigenvalue dominates
    the negative spectrum. The signed Rayleigh quotient is retained; a finite
    iteration budget gives an estimate, not a certified eigenvalue.

    Returns (lambda_1, v_1). Warm starting matters: along a training trajectory
    the top eigenvector moves slowly, so 3-4 iterations track it, while a cold
    start needs tens. `chunk` splits the full-batch HVP for memory-heavy archs.
    """
    device = next(net.parameters()).device
    q = q0.detach().clone() if q0 is not None else torch.randn(n_params(net), device=device)
    q = q / q.norm().clamp_min(1e-30)
    lam = float("nan")
    for _ in range(n_iter):
        Hq = hvp_chunked(net, loss_fn, X, Y, q, chunk)
        nrm = Hq.norm()
        if not torch.isfinite(nrm) or nrm < 1e-30:
            return lam, q
        lam_new = float(torch.dot(q, Hq).item())
        q = Hq / nrm
        if tol > 0.0 and np.isfinite(lam) and abs(lam_new - lam) <= tol * max(abs(lam_new), 1e-12):
            lam = lam_new
            break
        lam = lam_new
    # One final Rayleigh quotient at the returned vector.
    Hq = hvp_chunked(net, loss_fn, X, Y, q, chunk)
    if torch.isfinite(Hq).all():
        lam = float(torch.dot(q, Hq).item())
    return lam, q.detach()


def sample_batch_curvatures(
    net,
    loss_fn,
    X,
    Y,
    q: torch.Tensor,
    batch: int,
    n_batches: int,
    generator: torch.Generator,
) -> np.ndarray:
    """Draw n_batches fresh mini-batches at frozen theta and return h_B for each.

    Batches are uniform random subsets of size `batch` drawn without replacement
    within a batch and independently across batches. This is the standard fresh
    mini-batch idealisation of the SGD noise at a frozen point.
    """
    n = X.shape[0]
    out = np.empty(n_batches, dtype=np.float64)
    for i in range(n_batches):
        idx = torch.randperm(n, generator=generator, device=X.device)[:batch]
        out[i] = rayleigh(net, loss_fn, X[idx], Y[idx], q)
    return out


def params_flat(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters() if p.requires_grad])
