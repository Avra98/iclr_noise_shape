#!/usr/bin/env python3
"""Replay plain-SGD checkpoints while measuring occupancy in a fixed frame.

The center and directions NEVER follow the trajectory. This records parameter
occupancy, not a histogram of stochastic multipliers. Fresh, paired scalar-gain
probes use those same fixed axes and additionally retain batch-gradient gains.
Neither scalar gain is identified with the full network Lyapunov exponent.

Each invocation runs one (checkpoint, replay seed, eta multiplier) branch. A
shared --frame-file caches only a Ritz vector, random-axis recipe and checkpoint
fingerprint; it does not duplicate a model or store parameter histories.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from stability_common import (ARCHS, CFG, _flat_grad, build_arch, get_device,
                              gradient_curvature_sample, hvp_chunked,
                              seed_everything)


def atomic_json(path, obj):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=True))
    tmp.replace(path)


def atomic_npz(path, obj):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **obj)
    tmp.replace(path)


def atomic_torch(path, obj):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(2**20):
            h.update(b)
    return h.hexdigest()


@contextlib.contextmanager
def diagnostic_rng(device):
    """Diagnostics cannot consume the training stream, even with stochastic ops."""
    ns, ps = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        try:
            yield
        finally:
            np.random.set_state(ns)
            random.setstate(ps)


def leading_ritz(matvec, dimension, device, seed, iterations=40, tol=1e-3, dtype=torch.float32):
    """Largest algebraic Ritz value of a fully reorthogonalized Lanczos space.

    Unlike ordinary power iteration this chooses the largest algebraic value,
    not automatically the largest magnitude. A small residual certifies an
    approximate eigenpair, but does not certify that no unseen eigenvalue is
    larger. Record both residual and iteration cap instead of calling it exact.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dimension, generator=gen, dtype=dtype).to(device)
    q /= q.norm()
    vectors, alpha, beta = [], [], []
    oldq = torch.zeros_like(q)
    oldb = 0.0
    for _ in range(min(iterations, dimension)):
        vectors.append(q)
        z = matvec(q)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite Hessian-vector product while building fixed frame")
        a = float(torch.dot(q.double(), z.double()))
        z = z - a*q - oldb*oldq
        # Two modified Gram-Schmidt passes reduce loss of orthogonality.
        for _pass in range(2):
            for v in vectors:
                z = z - torch.dot(v.double(), z.double()).to(z.dtype)*v
        b = float(z.norm())
        alpha.append(a)
        if b < 1e-10 or len(vectors) >= min(iterations, dimension):
            break
        beta.append(b)
        oldq, oldb, q = q, b, z/b
    tridiag = np.diag(alpha)
    if beta:
        tridiag += np.diag(beta, 1) + np.diag(beta, -1)
    vals, vecs = np.linalg.eigh(tridiag)
    q = sum(float(w)*v for w, v in zip(vecs[:, -1], vectors))
    q /= q.norm()
    # Fix one deterministic orientation, then freeze it across every replay.
    pivot = int(q.abs().argmax())
    if float(q[pivot]) < 0:
        q = -q
    hq = matvec(q)
    rayleigh = float(torch.dot(q.double(), hq.double()))
    residual = float((hq - rayleigh*q).double().norm())
    relative = residual/max(float(hq.double().norm()), 1e-30)
    return q.detach(), dict(method="fully_reorthogonalized_Lanczos_largest_algebraic_Ritz",
                           iterations=len(alpha), requested_iterations=iterations,
                           rayleigh=rayleigh, ritz_value=float(vals[-1]),
                           ritz_min=float(vals[0]), absolute_residual=residual,
                           relative_residual=relative, residual_tolerance=tol,
                           residual_converged=bool(relative <= tol),
                           extremality_certified=False, sign_pivot=pivot)


def regenerate_basis(top, seed, random_axes):
    """Portable CPU recipe for identical random axes across devices/branches."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    top = top.detach().cpu()
    basis = [top / top.norm()]
    for _ in range(random_axes):
        v = torch.randn(top.numel(), generator=gen, dtype=top.dtype)
        for _pass in range(2):
            for q in basis:
                v -= torch.dot(q.double(), v.double()).to(v.dtype)*q
        n = float(v.norm())
        if n < 1e-10:
            raise ValueError("Insufficient independent dimensions for requested fixed axes")
        basis.append(v/n)
    return torch.stack(basis)


def provenance(checkpoint_path, checkpoint, metadata_path=None, arch_override=None):
    """Read both current arch_train and historical MLP/ViT checkpoint formats."""
    if metadata_path is None:
        base = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
        metadata_path = base / "trajectory_meta.json"
        if not metadata_path.exists() and "config" in checkpoint:
            metadata_path = base / "summary.json"
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    continuation = "config" in checkpoint and "completed_updates" in checkpoint
    if continuation:
        # These checkpoints were produced by nonconvex_campaign/real_networks/
        # continuation.py. Its sampling and online target noise are part of
        # the training rule and must not be replaced by arch_train defaults.
        merged = dict(meta.get("parent_provenance", {}))
        merged.update(checkpoint["config"])
        merged["continuation_parent_step"] = meta.get("parent_step")
        meta = merged
    meta.update(checkpoint.get("provenance", {}))
    arch = arch_override or checkpoint.get("arch") or meta.get("arch") or meta.get("model")
    if arch == "mlp":
        arch = "mlp_d2"
    if arch not in ARCHS:
        raise ValueError("Cannot identify architecture; pass --arch explicitly for historical checkpoints")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    eta = meta.get("eta", checkpoint.get("target_eta", checkpoint.get("eta")))
    if eta is None or "batch" not in meta:
        raise ValueError("Source metadata must identify eta and batch; use --metadata")
    if meta.get("optimizer", "SGD") != "SGD" or meta.get("momentum", 0) != 0 or meta.get("weight_decay", 0) != 0:
        raise ValueError("This runner supports original plain SGD only; do not substitute it for other optimizers")
    params = dict(arch=arch, eta=float(eta), batch=int(meta["batch"]),
                  n_train=int(meta.get("n_train", CFG.n_train)),
                  dataset_seed=int(meta.get("dataset_seed", CFG.dataset_seed)),
                  init_seed=int(meta.get("init_seed", CFG.init_seed)),
                  init_scale=float(meta.get("init_scale", ARCHS[arch]["init_scale"])),
                  warmup=int(meta.get("warmup", 0)), train_chunk=int(meta.get("train_chunk", 0)),
                  lam_chunk=int(meta.get("lam_chunk", ARCHS[arch]["lam_chunk"])),
                  checkpoint_step=int(checkpoint.get("next_step", checkpoint.get("step", -1))),
                  order_seed=int(meta.get("order_seed", CFG.order_seed)),
                  sampling="random_reshuffling_drop_remainder", label_noise=0.,
                  dtype="float32", update_rule="torch_optimizer_SGD", continuation=False)
    if continuation:
        parent_step = meta.get("continuation_parent_step")
        if parent_step is None:
            raise ValueError("Continuation source summary.json must identify parent_step")
        params.update(checkpoint_step=int(parent_step)+int(checkpoint["completed_updates"]),
                      sampling=meta.get("sampling", "iid_subsets"),
                      label_noise=float(meta.get("label_noise", 0.)),
                      dtype=meta.get("dtype", "float32"), warmup=0, train_chunk=128,
                      update_rule="flat_parameter_subtract_eta_gradient", continuation=True)
    elif meta.get("sampling", "random_reshuffling_drop_remainder") != "random_reshuffling_drop_remainder":
        raise ValueError("Unknown source sampling rule; an explicit adapter is required")
    if params["sampling"] not in ("iid_subsets", "reshuffle", "random_reshuffling_drop_remainder"):
        raise ValueError(f"Unsupported source sampling rule: {params['sampling']}")
    if params["dtype"] not in ("float32", "float64"):
        raise ValueError("Unsupported source precision")
    if params["checkpoint_step"] < 0:
        raise ValueError("Source checkpoint is missing its pre-update/next-step index")
    return params, meta, str(metadata_path.resolve())


def create_or_load_frame(path, net, loss_fn, X, Y, cp_hash, args, chunk, dataset_seed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    requested_n = len(X) if args.frame_n == 0 else min(len(X), args.frame_n)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            frame = torch.load(path, map_location="cpu", weights_only=False)
            if frame["checkpoint_sha256"] != cp_hash:
                raise ValueError("Fixed frame and source checkpoint have different hashes")
            if frame["random_axes"] != args.random_axes:
                raise ValueError("Fixed frame has a different number of random axes")
            expected = dict(frame_seed=args.frame_seed, hessian_examples=requested_n)
            for key, value in expected.items():
                if frame[key] != value:
                    raise ValueError(f"Fixed frame {key} mismatch: cached {frame[key]}, requested {value}")
            if frame["eigenpair"]["requested_iterations"] != args.frame_iters:
                raise ValueError("Fixed frame iteration budget changed; use a new frame-file")
            if frame.get("dataset_seed", dataset_seed) != dataset_seed or frame.get("n_train", len(X)) != len(X):
                raise ValueError("Fixed frame training dataset provenance changed")
            return frame
        n = requested_n
        gen = torch.Generator().manual_seed(args.frame_seed + 1)
        ids = torch.randperm(len(X), generator=gen)[:n].to(X.device)
        q, evidence = leading_ritz(
            lambda v: hvp_chunked(net, loss_fn, X[ids], Y[ids], v, chunk),
            sum(p.numel() for p in net.parameters() if p.requires_grad),
            X.device, args.frame_seed, args.frame_iters, args.frame_tol, X.dtype)
        frame = dict(schema_version=1, checkpoint_sha256=cp_hash,
                     checkpoint_path=str(args.checkpoint.resolve()), top_ritz_vector=q.cpu(),
                     axis_names=["fixed_top_ritz"]+[f"fixed_random_{i+1}" for i in range(args.random_axes)],
                     random_axes=args.random_axes, random_axis_seed=args.frame_seed+2,
                     random_axis_recipe="CPU torch.randn in saved vector dtype; two-pass modified Gram-Schmidt against earlier axes",
                     torch_version=torch.__version__, frame_seed=args.frame_seed,
                     dataset_seed=dataset_seed, n_train=len(X),
                     loss="SquaredLoss; mean over examples; sum over outputs",
                     hessian_sample_indices=ids.cpu(), hessian_examples=n,
                     hessian_dataset="entire_training_set" if n == len(X) else "fixed_random_training_subset",
                     eigenpair=evidence, center="exact source checkpoint parameters; never recentered")
        atomic_torch(path, frame)
        atomic_json(path.with_suffix(".json"), {k: v for k, v in frame.items()
                                              if k not in ("top_ritz_vector", "hessian_sample_indices")})
        return frame


def run(args):
    resume_path = args.outdir / 'resume_state.pt'
    resume = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path.exists() else None
    if args.outdir.exists() and (args.outdir / "meta.json").exists():
        if resume is None:
            raise FileExistsError(f"Existing branch has no resumable state: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cp_hash = sha256(args.checkpoint)
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, source, metadata_path = provenance(args.checkpoint, cp, args.metadata, args.arch)
    source_sampling=cfg['sampling']
    cfg['sampling']='iid_subsets'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    # A rolling source can change between the hash and load. Reject rather than
    # mixing a frame fingerprint with a different center; parent should pin it.
    if cp_hash != sha256(args.checkpoint):
        raise RuntimeError("Checkpoint changed while loading; pin a stable checkpoint before probing")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    seed_everything(args.replay_seed)
    eta = cfg["eta"]*args.eta_multiplier
    X, Y, net, loss_fn, opt = build_arch(cfg["arch"], device, eta=eta,
        n_train=cfg["n_train"], dataset_seed=cfg["dataset_seed"],
        init_seed=cfg["init_seed"], init_scale=cfg["init_scale"])
    if cfg["dtype"] == "float64":
        X, Y, net = X.double(), Y.double(), net.double()
    net.load_state_dict(cp["model_state_dict"], strict=True)
    if "optimizer_state_dict" in cp:
        opt.load_state_dict(cp["optimizer_state_dict"])
    for group in opt.param_groups:
        if group.get("momentum", 0) or group.get("weight_decay", 0) or group.get("nesterov", False):
            raise ValueError("Saved optimizer is not plain SGD")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules()):
        raise ValueError("BatchNorm model unsupported: probes would mutate running statistics")
    if any(isinstance(m, torch.nn.Dropout) and m.p for m in net.modules()):
        raise ValueError("Nonzero dropout requires explicit mask-aware Hessian protocol")
    chunk = cfg["lam_chunk"] if args.probe_chunk is None else args.probe_chunk
    train_chunk = cfg["train_chunk"] if args.train_chunk is None else args.train_chunk
    if args.original_order and cfg["continuation"]:
        raise ValueError("Continuation checkpoint lacks sampler RNG state: --original-order cannot be exact")
    frame_path = args.frame_file or args.outdir.parent / f"frame_{cp_hash[:16]}.pt"
    with diagnostic_rng(device):
        frame = create_or_load_frame(frame_path, net, loss_fn, X, Y, cp_hash, args, chunk, cfg["dataset_seed"])
    basis = regenerate_basis(frame["top_ritz_vector"], frame["random_axis_seed"], frame["random_axes"]).to(device)
    basis64 = basis.double()
    basis_norm64 = basis64.norm(dim=1)
    params = [p for p in net.parameters() if p.requires_grad]
    reference = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    # Free the (potentially large) optimizer/history snapshot after reading it.
    if args.original_order and "torch_rng_state" in cp:
        torch.set_rng_state(cp["torch_rng_state"].cpu())
        if device.type == "cuda" and cp.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([s.cpu() for s in cp["cuda_rng_states"]])
        np.random.set_state(cp["numpy_rng_state"])
        random.setstate(cp["python_rng_state"])
    else:
        seed_everything(args.replay_seed)
    del cp
    order_seed = cfg["order_seed"] if args.original_order else args.replay_seed
    ordering = torch.Generator(device=device)
    ordering.manual_seed(order_seed)
    noise_gen = torch.Generator(device=device).manual_seed(args.replay_seed+1919)
    probe_gen = torch.Generator().manual_seed(args.probe_seed)
    probe_noise_gen = torch.Generator(device=device).manual_seed(args.probe_seed+77177)
    meta = dict(schema_version=1, run_id=args.outdir.name, architecture=cfg["arch"],
        seed=args.replay_seed, source_seed=source.get("seed"),
        parent_seed=source.get("parent_seed", source.get("seed", cfg["init_seed"])),
        checkpoint_step=cfg["checkpoint_step"],
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=cp_hash,
        source_metadata=metadata_path, batch_size=cfg["batch"], learning_rate=eta,
        parent_learning_rate=cfg["eta"], eta_multiplier=args.eta_multiplier,
        requested_steps=args.steps, axis_names=frame["axis_names"], frame_file=str(frame_path.resolve()),
        frame_hessian_examples=frame["hessian_examples"], frame_hessian_dataset=frame["hessian_dataset"],
        eigenpair=frame["eigenpair"], frame_seed=frame["frame_seed"],
        fixed_center="source checkpoint parameters", projection="q_k^T(theta_t-theta_ref)",
        coordinate_precision=f"float64 accumulation of differences between {cfg['dtype']} parameters",
        axis_gram_max_error=float((basis64 @ basis64.T-torch.eye(len(basis), device=device)).abs().max()),
        optimizer="SGD", momentum=0, weight_decay=0, loss="SquaredLoss; mean over examples; sum over outputs",
        dtype=cfg["dtype"], n_train=cfg["n_train"], dataset_seed=cfg["dataset_seed"],
        init_seed=cfg["init_seed"], warmup=cfg["warmup"], train_chunk=train_chunk, original_train_chunk=cfg["train_chunk"],
        train_chunk_override=bool(train_chunk != cfg["train_chunk"]), probe_chunk=chunk,
        source_sampling=source_sampling, training_sampling=cfg["sampling"], source_update_rule=cfg["update_rule"],
        label_noise=cfg["label_noise"], label_noise_seed=args.replay_seed+1919,
        label_noise_interpretation="independent Gaussian per sampled target/output, shared by gradient and Hessian",
        frame_hessian_target_noise="expectation over zero-mean target noise; clean-target Hessian (squared loss)",
        training_order_seed=order_seed, original_order=args.original_order,
        probe_seed=args.probe_seed, probe_sampling="independent_uniform_subsets_without_replacement",
        probe_every=args.probe_every, probe_draws=args.probe_draws,
        probe_adjacent=args.probe_adjacent,
        probe_rng_isolated=True, status="running", completed_updates=0,
        stationarity_assumed=False, scalar_gain_is_full_network_lyapunov=False,
        divergence_loss=args.divergence_loss, displacement_cutoff=args.displacement_cutoff,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        torch_version=torch.__version__, source_sha256=sha256(Path(__file__)))
    trace = {k: [] for k in ("step", "absolute_step", "projection", "parameter_norm", "displacement_norm",
                             "loss_step", "loss", "lr", "gradient_axis_cosine", "gradient_norm")}
    probes = {k: [] for k in ("probe_step", "absolute_step", "lr", "a", "axis_h", "axis_norm_gain",
                              "batch_gradient_a", "s_B", "gradient_norm", "gradient_axis_cosine")}
    alignment=[];full_loss=[]
    stops = []
    probe_schedule = {0, args.steps}
    if args.probe_every:
        probe_schedule.update(range(0, args.steps+1, args.probe_every))
    if args.probe_adjacent:
        probe_schedule.update(t+1 for t in list(probe_schedule) if t+1 <= args.steps)
    meta["requested_probe_steps"] = sorted(probe_schedule) if args.probe_draws else []
    signal.signal(signal.SIGTERM, lambda n, _: stops.append(n))
    signal.signal(signal.SIGINT, lambda n, _: stops.append(n))

    def lr_at(t):
        a = cfg["checkpoint_step"]+t
        return eta*min(1., (a+1)/cfg["warmup"]) if cfg["warmup"] else eta

    def record(t):
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        delta64 = flat.double()-reference.double()
        projection = (basis64 @ delta64).cpu().numpy()
        trace["step"].append(t)
        trace["absolute_step"].append(cfg["checkpoint_step"]+t)
        trace["projection"].append(projection)
        trace["parameter_norm"].append(float(flat.double().norm()))
        trace["displacement_norm"].append(float(delta64.norm()))
        return bool(torch.isfinite(flat).all()), trace["displacement_norm"][-1]

    def probe(t):
        if t%512==0:
            with diagnostic_rng(device):
                q,info=leading_ritz(lambda v:hvp_chunked(net,loss_fn,X,Y,v,chunk),len(reference),device,19817,40,1e-3)
            alignment.append(dict(step=t,alignment=abs(float(q.double()@basis64[0]))/float(q.double().norm()*basis_norm64[0]),**info))
            print(json.dumps(dict(alignment=alignment[-1])),flush=True)
        with torch.no_grad():
            lv=sum(float(loss_fn(net(X[i:i+256]).squeeze(dim=-1),Y[i:i+256]))*len(X[i:i+256])/len(X) for i in range(0,len(X),256))
        full_loss.append(dict(step=t,loss=lv))
        with diagnostic_rng(device):
            lr = lr_at(t)
            for _ in range(args.probe_draws):
                ids = torch.randperm(len(X), generator=probe_gen)[:cfg["batch"]].to(device)
                xb, yb = X[ids], Y[ids]
                if cfg["label_noise"]:
                    yb = yb+cfg["label_noise"]*torch.randn(yb.shape, device=device, dtype=yb.dtype, generator=probe_noise_gen)
                s, g, hg = gradient_curvature_sample(net, loss_fn, xb, yb, chunk)
                hq = torch.stack([hvp_chunked(net, loss_fn, xb, yb, q, chunk) for q in basis])
                h = ((basis64*hq.double()).sum(dim=1)/basis_norm64.square()).cpu().numpy()
                gn = float(g.double().norm())
                values = dict(probe_step=t, absolute_step=cfg["checkpoint_step"]+t, lr=lr,
                    a=1-lr*h, axis_h=h,
                    axis_norm_gain=((basis64-lr*hq.double()).norm(dim=1)/basis_norm64).cpu().numpy(),
                    batch_gradient_a=1-lr*s, s_B=s, gradient_norm=gn,
                    gradient_axis_cosine=((basis64 @ g.double())/basis_norm64).cpu().numpy()/gn if gn else np.full(len(basis), np.nan))
                for k, v in values.items():
                    probes[k].append(v)

    def flush():
        ints = {"step", "absolute_step", "loss_step", "probe_step"}
        atomic_npz(args.outdir / "trace.npz", {k: np.asarray(v, dtype=np.int64 if k in ints else np.float64)
                                              for k, v in trace.items()})
        atomic_npz(args.outdir / "same_axis_gains.npz", {k: np.asarray(v, dtype=np.int64 if k in ints else np.float64)
                                                        for k, v in probes.items()})
        meta.update(elapsed_s=time.time()-started, completed_updates=trace["step"][-1],
                    stop_signals=stops, projection_observations=len(trace["step"]),
                    gain_draws=len(probes["probe_step"]))
        atomic_json(args.outdir / "meta.json", meta)
        atomic_json(args.outdir / "eigenvector_alignment.json", alignment)
        atomic_json(args.outdir / "full_loss.json", full_loss)
        atomic_torch(resume_path, dict(completed_updates=int(trace['step'][-1]),
            model={k:v.detach().cpu() for k,v in net.state_dict().items()},
            ordering=ordering.get_state(), probe_gen=probe_gen.get_state(),
            noise_gen=noise_gen.get_state(), probe_noise_gen=probe_noise_gen.get_state(),
            checkpoint_sha256=cp_hash))

    atomic_json(args.outdir / "meta.json", meta)
    start = 0
    if resume is not None:
        assert resume['checkpoint_sha256'] == cp_hash
        start = int(resume['completed_updates'])
        net.load_state_dict(resume['model'])
        ordering.set_state(resume['ordering'])
        probe_gen.set_state(resume['probe_gen'])
        noise_gen.set_state(resume['noise_gen'])
        probe_noise_gen.set_state(resume['probe_noise_gen'])
        saved = dict(np.load(args.outdir/'trace.npz'))
        for k in trace:
            timekey='loss_step' if k in ['loss_step','loss','lr','gradient_axis_cosine','gradient_norm'] else 'step'
            mask=saved[timekey]<start if timekey=='loss_step' else saved[timekey]<=start
            trace[k]=list(saved[k][mask])
        saved=dict(np.load(args.outdir/'same_axis_gains.npz'))
        for k in probes:probes[k]=list(saved[k][saved['probe_step']<=start])
        alignment[:]=[v for v in json.loads((args.outdir/'eigenvector_alignment.json').read_text()) if v['step']<=start]
        full_loss[:]=[v for v in json.loads((args.outdir/'full_loss.json').read_text()) if v['step']<=start]
        finite=True;displacement=trace['displacement_norm'][-1]
        meta['resumed_from_update']=start
        print(f'Resuming saved IID continuation at {start}',flush=True)
    else:
        finite, displacement = record(0)
    status = "completed"
    if not finite:
        status = "nonfinite_initial_checkpoint"
    else:
        if args.probe_draws and resume is None:
            probe(0)
        flush()
        last_epoch, permutation, position = None, None, len(X)
        steps_per_epoch = len(X)//cfg["batch"]
        for t in range(start,args.steps):
            absolute_step = cfg["checkpoint_step"]+t
            if cfg["sampling"] == "iid_subsets":
                ids = torch.randperm(len(X), generator=ordering, device=device)[:cfg["batch"]]
            elif cfg["sampling"] == "reshuffle":
                if position+cfg["batch"] > len(X):
                    permutation = torch.randperm(len(X), generator=ordering, device=device)
                    position = 0
                ids = permutation[position:position+cfg["batch"]]
                position += cfg["batch"]
            else:
                epoch, i = divmod(absolute_step, steps_per_epoch)
                if epoch != last_epoch:
                    ordering.manual_seed((abs(order_seed)+epoch) % (2**63-1))
                    permutation = torch.randperm(len(X), generator=ordering, device=device)
                    last_epoch = epoch
                ids = permutation[i*cfg["batch"]:(i+1)*cfg["batch"]]
            xb, yb, lr = X[ids], Y[ids], lr_at(t)
            if cfg["label_noise"]:
                yb = yb+cfg["label_noise"]*torch.randn(yb.shape, device=device, dtype=yb.dtype, generator=noise_gen)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad()
            tc = train_chunk or len(xb)
            lv = 0.
            for j in range(0, len(xb), tc):
                loss = loss_fn(net(xb[j:j+tc]).squeeze(dim=-1), yb[j:j+tc])*len(xb[j:j+tc])/len(xb)
                lv += float(loss.detach())
                loss.backward()
            grad = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                              for p in params])
            gn = float(grad.double().norm())
            trace["loss_step"].append(t)
            trace["loss"].append(lv)
            trace["lr"].append(lr)
            trace["gradient_norm"].append(gn)
            trace["gradient_axis_cosine"].append(((basis64 @ grad.double())/basis_norm64).cpu().numpy()/gn if gn else np.full(len(basis), np.nan))
            if not np.isfinite(lv) or not torch.isfinite(grad).all():
                status = "nonfinite_loss_or_gradient"
                break
            if lv > args.divergence_loss:
                status = "escaped_loss_cutoff"
                break
            if cfg["update_rule"] == "flat_parameter_subtract_eta_gradient":
                # Match continuation.py's subtraction, including its rounding
                # behavior; torch SGD may use a fused axpy operation instead.
                with torch.no_grad():
                    updated = torch.cat([p.detach().reshape(-1) for p in params])-lr*grad
                    offset = 0
                    for param in params:
                        param.copy_(updated[offset:offset+param.numel()].view_as(param))
                        offset += param.numel()
            else:
                opt.step()
            finite, displacement = record(t+1)
            if not finite:
                status = "nonfinite_parameters"
                break
            if displacement > args.displacement_cutoff:
                status = "escaped_displacement_cutoff"
                break
            if args.probe_draws and t+1 in probe_schedule:
                probe(t+1)
                flush()
            if (t+1) % args.flush_every == 0:
                flush()
                print(f"step={t+1}/{args.steps} loss={lv:.7g} displacement={displacement:.7g} elapsed={time.time()-started:.1f}s", flush=True)
            if stops:
                status = "interrupted"
                break
    meta["status"] = status
    meta["escape_or_failure_step"] = trace["step"][-1] if status != "completed" else None
    flush()
    if args.save_final:
        atomic_torch(args.outdir / "final_state.pt", dict(next_step=cfg["checkpoint_step"]+trace["step"][-1],
            model_state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
            optimizer_state_dict=opt.state_dict(), provenance={**source, "eta": eta},
            frame_file=str(frame_path.resolve())))
    print(json.dumps({k: meta[k] for k in ("architecture", "status", "completed_updates", "elapsed_s", "gain_draws")}), flush=True)
    return meta


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--arch", choices=sorted(ARCHS))
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--replay-seed", type=int, default=93101)
    p.add_argument("--eta-multiplier", type=float, default=1.)
    p.add_argument("--frame-file", type=Path)
    p.add_argument("--frame-seed", type=int, default=51571)
    p.add_argument("--frame-n", type=int, default=1024, help="0 uses whole training set; otherwise explicitly labeled fixed subset")
    p.add_argument("--frame-iters", type=int, default=40)
    p.add_argument("--frame-tol", type=float, default=1e-3)
    p.add_argument("--random-axes", type=int, default=2)
    p.add_argument("--probe-every", type=int, default=1000, help="0 probes start/end only")
    p.add_argument("--probe-draws", type=int, default=64)
    p.add_argument("--probe-seed", type=int, default=161803)
    p.add_argument("--probe-adjacent", action="store_true", help="Also probe one update after each scheduled time (including0), exposing even/odd phase differences")
    p.add_argument("--probe-chunk", type=int)
    p.add_argument("--train-chunk", type=int, help="Explicit exact-mean accumulation override; metadata records numerical change")
    p.add_argument("--original-order", action="store_true", help="Use original training permutation sequence for reproduction control")
    p.add_argument("--divergence-loss", type=float, default=1e12)
    p.add_argument("--displacement-cutoff", type=float, default=1e6)
    p.add_argument("--flush-every", type=int, default=250)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--save-final", action="store_true", help="Optional validation/resume artifact; off to limit storage")
    return p


if __name__ == "__main__":
    ap = parser()
    args = ap.parse_args()
    if args.steps < 1 or args.eta_multiplier <= 0 or args.frame_iters < 1 or args.frame_n < 0 or args.random_axes < 0:
        ap.error("Require positive steps, eta-multiplier, frame-iters and nonnegative frame-n/random-axes")
    if args.probe_every < 0 or args.probe_draws < 0 or args.flush_every < 1:
        ap.error("Require nonnegative probe frequencies/counts and positive flush-every")
    run(args)
