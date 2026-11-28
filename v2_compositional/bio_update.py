# -*- coding: utf-8 -*-
"""Conservative resource allocation for paperplaneMoe training.

This module changes *optimizer allocation*, not expert routing.  The model's
compositional forward pass remains:

    effective parameters = sum_i wiring_i * organelle_i

followed by one nonlinear forward evaluation.  No top-k expert selection is
introduced here.

The biological language is deliberately narrow:

* ``phi`` is a proteome-allocation analogue on a simplex.
* ``phi0`` is the parameter-count share of each optimizer compartment.
* exponentiated-gradient is the mathematical update on that simplex.
* gradient RMS is an optimisation-pressure proxy, not a measured metabolic
  flux-control coefficient.

Detailed balance, Haldane relations and MWC allostery require physical states
and forward/reverse rates that AdamW does not provide.  They therefore remain
documented scope limits rather than claimed laws of this optimiser.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # Pure-math tests and documentation builds can still run.
    torch = None  # type: ignore


EPS = 1e-12


def _no_grad(function):
    if torch is None:
        return function
    return torch.no_grad()(function)


def _normalise(values: Sequence[float]) -> List[float]:
    """Return a strictly positive simplex vector."""
    if not values:
        raise ValueError("simplex must contain at least one value")
    clean = [max(float(v), EPS) for v in values]
    total = sum(clean)
    return [v / total for v in clean]


def mirror_allocation(
    phi: Sequence[float],
    phi0: Sequence[float],
    signal: Sequence[float],
    eta: float,
    restore: float,
) -> List[float]:
    """One negative-entropy mirror step on the allocation simplex.

    Positive ``signal[i]`` moves allocation toward compartment ``i``.  The
    restoring term is relative to the parameter-share baseline ``phi0``.
    Softmax is evaluated with a max subtraction for numerical stability.
    """
    if not (len(phi) == len(phi0) == len(signal)):
        raise ValueError("phi, phi0 and signal must have equal length")
    if eta < 0.0 or restore < 0.0:
        raise ValueError("eta and restore must be non-negative")
    logits = []
    for p, p0, s in zip(phi, phi0, signal):
        lp = math.log(max(float(p), EPS))
        home = lp - math.log(max(float(p0), EPS))
        logits.append(lp + float(eta) * (float(s) - float(restore) * home))
    peak = max(logits)
    weights = [math.exp(v - peak) for v in logits]
    return _normalise(weights)


def capped_simplex(
    values: Sequence[float], phi0: Sequence[float], trust: float
) -> List[float]:
    """KL-style water filling under ``phi_i <= trust * phi0_i``.

    ``trust >= 1`` makes the capped simplex feasible because the cap mass is
    ``trust * sum(phi0)``.  At ``trust == 1`` the only feasible point is
    exactly ``phi0``; this is a useful frozen-allocation control arm.
    """
    if len(values) != len(phi0):
        raise ValueError("values and phi0 must have equal length")
    if trust < 1.0:
        raise ValueError("trust must be >= 1 so the capped simplex is feasible")
    base = _normalise(phi0)
    raw = [max(float(v), EPS) for v in values]
    caps = [float(trust) * p for p in base]
    if trust == 1.0:
        return base

    def mass(scale: float) -> float:
        return sum(min(c, scale * v) for c, v in zip(caps, raw))

    lo, hi = 0.0, 1.0
    while mass(hi) < 1.0:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mass(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    out = [min(c, hi * v) for c, v in zip(caps, raw)]
    # Remove the last few floating-point ulps without violating the caps.
    total = sum(out)
    residual = 1.0 - total
    if abs(residual) > 1e-14:
        for i in range(len(out)):
            room = caps[i] - out[i]
            delta = min(room, residual) if residual > 0 else max(-out[i], residual)
            out[i] += delta
            residual -= delta
            if abs(residual) <= 1e-14:
                break
    return out


def weighted_scale_std(phi: Sequence[float], phi0: Sequence[float]) -> float:
    """RMS departure of relative learning-rate scales from the baseline."""
    base = _normalise(phi0)
    return math.sqrt(sum(p0 * (p / p0 - 1.0) ** 2 for p, p0 in zip(phi, base)))


def _unique_parameters(parameters: Iterable[Any], seen: set) -> List[Any]:
    out = []
    for parameter in parameters:
        ident = id(parameter)
        if ident not in seen:
            seen.add(ident)
            out.append(parameter)
    return out


def optimizer_compartments(model, cells: Sequence[Any]) -> List[Tuple[str, str, List[Any]]]:
    """Partition every model parameter into one shared or one cell compartment."""
    if torch is None:
        raise RuntimeError("optimizer compartments require torch")
    cell_ids = {id(p) for cell in cells for p in cell.parameters()}
    seen: set = set()
    shared = _unique_parameters((p for p in model.parameters() if id(p) not in cell_ids), seen)
    groups: List[Tuple[str, str, List[Any]]] = []
    if shared:
        groups.append(("shared", "shared", shared))
    for index, cell in enumerate(cells):
        params = _unique_parameters(cell.parameters(), seen)
        if params:
            groups.append((f"cell:{index}", "cell", params))

    model_params = list(model.parameters())
    if len(seen) != len({id(p) for p in model_params}):
        raise RuntimeError("optimizer compartment partition lost or duplicated parameters")
    if not groups:
        raise RuntimeError("no trainable optimizer compartments were found")
    return groups


def rebind_optimizer_groups(optimizer, model, cells: Sequence[Any], base_lr: float) -> int:
    """Create one shared group plus one complete group per compositional cell.

    Optimizer state remains keyed by parameter object.  Repartitioning therefore
    preserves existing state for parameters that survive.  A split-created
    optimizer starts empty and is partitioned before its first step.
    """
    compartments = optimizer_compartments(model, cells)
    template: Dict[str, Any] = dict(optimizer.defaults)
    if optimizer.param_groups:
        template.update({
            key: value
            for key, value in optimizer.param_groups[0].items()
            if key not in {"params", "bio_group", "bio_role", "bio_cell"}
        })
    template["lr"] = float(base_lr)
    new_groups = []
    live_ids = set()
    for key, role, params in compartments:
        group = dict(template)
        group.update(
            params=list(params),
            bio_group=key,
            bio_role=role,
            bio_cell=(int(key.split(":", 1)[1]) if role == "cell" else -1),
        )
        new_groups.append(group)
        live_ids.update(id(p) for p in params)
    optimizer.param_groups[:] = new_groups
    # Discard state for parameters replaced by a structural split.
    for parameter in list(optimizer.state):
        if id(parameter) not in live_ids:
            del optimizer.state[parameter]
    return len(new_groups)


@dataclass
class BioUpdateSnapshot:
    phi: List[float]
    phi0: List[float]
    pressure: List[float]
    scale: List[float]
    sum_phi: float
    budget_gap: float
    scale_std: float
    max_scale: float
    min_scale: float
    resync: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phi": self.phi,
            "phi0": self.phi0,
            "pressure": self.pressure,
            "scale": self.scale,
            "sum_phi": self.sum_phi,
            "budget_gap": self.budget_gap,
            "scale_std": self.scale_std,
            "max_scale": self.max_scale,
            "min_scale": self.min_scale,
            "resync": self.resync,
        }


class BioUpdate:
    """Simplex-constrained learning-rate allocation across compose compartments."""

    def __init__(
        self,
        optimizer,
        *,
        eta_phi: float = 0.05,
        kappa: float = 0.30,
        tau_e: float = 100.0,
        c_weight: float = 0.0,
        trust: float = 4.0,
        trace_path: Optional[str] = None,
        trace_flush_every: int = 10,
    ):
        if torch is None:
            raise RuntimeError("BioUpdate requires torch")
        if trust < 1.0:
            raise ValueError("bio_trust must be >= 1")
        if eta_phi < 0.0 or kappa < 0.0 or tau_e < 1.0:
            raise ValueError("invalid bio update hyperparameters")
        self.optimizer = optimizer
        self.eta_phi = float(eta_phi)
        self.kappa = float(kappa)
        self.tau_e = float(tau_e)
        self.c_weight = float(c_weight)
        self.trust = float(trust)
        self.trace_path = trace_path
        self.trace_flush_every = max(1, int(trace_flush_every))
        self._trace_handle = None
        self._trace_count = 0
        self._resync_pending = False
        self._loss0: Optional[float] = None
        self._max_scale_std = 0.0
        self._configure_from_optimizer()

    @classmethod
    def from_args(cls, args, optimizer, model, cells, base_lr: float, trace_path: str):
        mode = str(getattr(args, "bio_update", "off")).lower()
        if mode in {"off", "none", "", "false", "0"}:
            return None
        if str(getattr(args, "enz_rate_mode", "off")).lower() != "off":
            raise ValueError("--bio-update on cannot be combined with --enz-rate-mode != off")
        rebind_optimizer_groups(optimizer, model, cells, base_lr)
        c_weight = float(getattr(args, "bio_c_weight", 0.0))
        if bool(getattr(args, "bio_use_C", False)) and c_weight == 0.0:
            c_weight = 1.0
        return cls(
            optimizer,
            eta_phi=float(getattr(args, "bio_eta_phi", 0.05)),
            kappa=float(getattr(args, "bio_kappa", 0.30)),
            tau_e=float(getattr(args, "bio_tau_e", 100.0)),
            c_weight=c_weight,
            trust=float(getattr(args, "bio_trust", 4.0)),
            trace_path=trace_path,
            trace_flush_every=int(getattr(args, "bio_trace_flush_every", 10)),
        )

    def _configure_from_optimizer(
        self,
        old_scales: Optional[Mapping[str, float]] = None,
        old_pressure: Optional[Mapping[str, float]] = None,
    ) -> None:
        groups = list(self.optimizer.param_groups)
        if len(groups) < 2:
            raise RuntimeError("G-PHI requires at least two optimizer compartments")
        self.keys = [str(group.get("bio_group", f"group:{i}")) for i, group in enumerate(groups)]
        if len(set(self.keys)) != len(self.keys):
            raise RuntimeError("optimizer compartment names must be unique")
        counts = [sum(int(p.numel()) for p in group["params"]) for group in groups]
        if any(count <= 0 for count in counts):
            raise RuntimeError("BioUpdate received an empty optimizer group")
        self.device = groups[0]["params"][0].device
        self.phi0 = torch.tensor(_normalise(counts), dtype=torch.float64, device=self.device)
        if old_scales:
            raw = torch.tensor(
                [float(self.phi0[i]) * float(old_scales.get(key, 1.0))
                 for i, key in enumerate(self.keys)],
                dtype=torch.float64,
                device=self.device,
            )
            raw = raw / raw.sum()
            projected = capped_simplex(raw.detach().cpu().tolist(), self.phi0.cpu().tolist(), self.trust)
            self.phi = torch.tensor(projected, dtype=torch.float64, device=self.device)
        else:
            self.phi = self.phi0.clone()
        self.pressure_ema = torch.tensor(
            [float((old_pressure or {}).get(key, 0.0)) for key in self.keys],
            dtype=torch.float64,
            device=self.device,
        )
        self._last_snapshot = self._snapshot(resync=self._resync_pending)

    def rebind(self, optimizer, model, cells: Sequence[Any], base_lr: float) -> int:
        """Repartition a split-created optimizer and preserve named allocation state."""
        old_scales = {
            key: float(self.phi[i].item() / self.phi0[i].item())
            for i, key in enumerate(self.keys)
        }
        old_pressure = {key: float(self.pressure_ema[i].item()) for i, key in enumerate(self.keys)}
        self.optimizer = optimizer
        count = rebind_optimizer_groups(optimizer, model, cells, base_lr)
        self._resync_pending = True
        self._configure_from_optimizer(old_scales, old_pressure)
        return count

    @_no_grad
    def observe(self, loss: float) -> Dict[str, Any]:
        """Measure per-parameter gradient RMS for each optimizer compartment."""
        if self._loss0 is None:
            self._loss0 = float(loss)
        rms = []
        for group in self.optimizer.param_groups:
            squared = None
            count = 0
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float()
                term = grad.square().sum()
                squared = term if squared is None else squared + term
                count += grad.numel()
            value = 0.0 if squared is None or count == 0 else math.sqrt(float(squared.item()) / count)
            rms.append(value)
        current = torch.tensor(rms, dtype=torch.float64, device=self.device)
        beta = 1.0 - 1.0 / self.tau_e
        self.pressure_ema.mul_(beta).add_(current, alpha=1.0 - beta)
        progress = (self._loss0 - float(loss)) / max(abs(self._loss0), EPS)
        return {"gradient_rms": rms, "progress": float(progress)}

    @_no_grad
    def step(self, base_lr: float) -> BioUpdateSnapshot:
        """Move allocation on the simplex, then apply budget-conserving LR scales."""
        pressure = self.pressure_ema.clamp(min=EPS)
        log_pressure = pressure.log()
        centre = (self.phi0 * log_pressure).sum()
        relative = (log_pressure - centre).clamp(min=-5.0, max=5.0)
        signal = self.c_weight * relative
        home = (self.phi.clamp(min=EPS).log() - self.phi0.log())
        logits = self.phi.clamp(min=EPS).log() + self.eta_phi * (signal - self.kappa * home)
        raw_phi = torch.softmax(logits, dim=0)
        projected = capped_simplex(
            raw_phi.detach().cpu().tolist(), self.phi0.detach().cpu().tolist(), self.trust
        )
        self.phi = torch.tensor(projected, dtype=torch.float64, device=self.device)
        scales = self.phi / self.phi0
        budget_gap = float((self.phi0 * scales).sum().item() - 1.0)
        if not torch.isfinite(scales).all() or abs(budget_gap) > 1e-10:
            raise RuntimeError(f"BioUpdate conservation failure: budget_gap={budget_gap!r}")
        for group, scale in zip(self.optimizer.param_groups, scales):
            group["lr"] = float(base_lr) * float(scale.item())
            group["bio_scale"] = float(scale.item())
        snapshot = self._snapshot(resync=self._resync_pending)
        self._resync_pending = False
        self._last_snapshot = snapshot
        self._max_scale_std = max(self._max_scale_std, snapshot.scale_std)
        return snapshot

    def _snapshot(self, resync: bool) -> BioUpdateSnapshot:
        phi = [float(v) for v in self.phi.detach().cpu().tolist()]
        phi0 = [float(v) for v in self.phi0.detach().cpu().tolist()]
        pressure = [float(v) for v in self.pressure_ema.detach().cpu().tolist()]
        scale = [p / p0 for p, p0 in zip(phi, phi0)]
        gap = sum(p0 * s for p0, s in zip(phi0, scale)) - 1.0
        return BioUpdateSnapshot(
            phi=phi,
            phi0=phi0,
            pressure=pressure,
            scale=scale,
            sum_phi=sum(phi),
            budget_gap=gap,
            scale_std=weighted_scale_std(phi, phi0),
            max_scale=max(scale),
            min_scale=min(scale),
            resync=bool(resync),
        )

    def write_trace(self, step: int, snapshot: BioUpdateSnapshot) -> None:
        if not self.trace_path:
            return
        if self._trace_handle is None:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            self._trace_handle = open(self.trace_path, "w", encoding="utf-8", newline="\n")
        row = {
            "step": int(step),
            "G": len(self.keys),
            "group_keys": self.keys,
            "c_weight": self.c_weight,
            "eta_phi": self.eta_phi,
            "kappa": self.kappa,
            "trust": self.trust,
            **snapshot.as_dict(),
        }
        self._trace_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._trace_count += 1
        if self._trace_count % self.trace_flush_every == 0:
            self._trace_handle.flush()

    def require_active(self, minimum_scale_std: float = 0.005) -> None:
        if self.c_weight <= 0.0:
            return
        if self._trace_count == 0:
            raise RuntimeError("G-PHI: no bio trace rows were written")
        if self._max_scale_std < float(minimum_scale_std):
            raise RuntimeError(
                "G-PHI: allocation mechanism inactive: "
                f"max_scale_std={self._max_scale_std:.6g} < {minimum_scale_std:.6g}"
            )

    def summary(self) -> Dict[str, Any]:
        snapshot = self._last_snapshot
        return {
            "G": len(self.keys),
            "group_keys": self.keys,
            "c_weight": self.c_weight,
            "eta_phi": self.eta_phi,
            "kappa": self.kappa,
            "tau_e": self.tau_e,
            "trust": self.trust,
            "trace_path": self.trace_path,
            "trace_rows": self._trace_count,
            "max_scale_std": self._max_scale_std,
            **snapshot.as_dict(),
        }

    def close(self) -> None:
        if self._trace_handle is not None:
            self._trace_handle.flush()
            self._trace_handle.close()
            self._trace_handle = None


def validate_trace(
    path: str,
    require_active: bool = False,
    minimum_scale_std: float = 0.005,
    require_resync: bool = False,
) -> Dict[str, Any]:
    """Validate a completed JSONL trace without importing torch."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    errors = []
    if not rows:
        errors.append("trace is empty")
    for index, row in enumerate(rows):
        if int(row.get("G", 0)) < 2:
            errors.append(f"row {index}: G < 2")
        if abs(float(row.get("sum_phi", 0.0)) - 1.0) > 1e-9:
            errors.append(f"row {index}: simplex conservation failed")
        if abs(float(row.get("budget_gap", 1.0))) > 1e-9:
            errors.append(f"row {index}: learning-rate budget failed")
    maximum = max((float(row.get("scale_std", 0.0)) for row in rows), default=0.0)
    resync_rows = sum(bool(row.get("resync", False)) for row in rows)
    if require_active and maximum < float(minimum_scale_std):
        errors.append(
            f"allocation inactive: max_scale_std={maximum:.6g} < {minimum_scale_std:.6g}"
        )
    if require_resync and resync_rows == 0:
        errors.append("split resynchronisation was not observed")
    return {
        "ok": not errors,
        "rows": len(rows),
        "max_scale_std": maximum,
        "resync_rows": resync_rows,
        "errors": errors,
    }
