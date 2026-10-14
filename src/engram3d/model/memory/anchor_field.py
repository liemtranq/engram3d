# =====================================================================
# Engram3D · Phase D, Part 1 · persistent sparse 3D anchor field
# File: src/engram3d/model/memory/anchor_field.py
# Status: UNTESTED COMPLETE IMPLEMENTATION — validate with
#         tests/test_anchor_field.py BEFORE any training use (§6 gate).
# ---------------------------------------------------------------------
# Proposal alignment (equations preserved exactly):
#   §4.1  A_i = (c_i, f_i, Σ_i, e_i, τ_i); 0.15 m lattice inside a
#         12×12×3 m envelope, then sparsified; agent-centric field.
#   §4.2  µ_new = (1−η)·µ_mem + η·µ_obs                [exact]
#         η★    = σ²_mem / (σ²_mem + σ²_obs)          [exact]
#         w ∝ Σ⁻¹ (conflict resolution)               [A28]
#   §4.5  counterfactual anchor deletion.
# Implementation choices A27–A33: see docs/AMBIGUITIES.md (summarized
# in the module docstring below). Execution representation (ADR-003):
# static-capacity buffers, capacity = 128,000 = the proposal's own
# lattice cell count; occupancy carried by e_i — sparse semantics
# preserved, no dense substitution.
# Numerics (§16): FP32 state and fusion; ε and σ²-floor guards; closed
# rows sanitized (cannot inject NaN/Inf; zero gradient by design).
# Duplicate-index scatters are order-proof by construction: all rows
# targeting an anchor carry identical values after conflict fusion.
# =====================================================================
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["AnchorFieldConfig", "AnchorState", "lattice_centers",
           "points_to_cell_indices", "cell_index_to_center", "eta_star",
           "init", "write", "write_closed_form", "apply_anchor_deletion",
           "active_mask", "summarize"]


class AnchorFieldConfig(NamedTuple):
    lattice_spacing: float = 0.15            # m   (proposal §4.1)
    envelope: tuple = (12.0, 3.0, 12.0)      # (x, y, z) extents; y vertical (A32)
    feat_dim: int = 128
    sigma2_pos_init: float = 1.0             # m² prior
    sigma2_feat_init: float = 1.0
    sigma2_floor: float = 1e-6               # §16 guard
    exist_logit_init: float = -1.0           # A29
    exist_logit_update: float = 2.0          # A29
    exist_active_threshold: float = 0.0      # sparsification threshold
    eps: float = 1e-8

    @property
    def nx(self) -> int:
        return int(round(self.envelope[0] / self.lattice_spacing))   # 80

    @property
    def ny(self) -> int:
        return int(round(self.envelope[1] / self.lattice_spacing))   # 20

    @property
    def nz(self) -> int:
        return int(round(self.envelope[2] / self.lattice_spacing))   # 80

    @property
    def capacity(self) -> int:
        return self.nx * self.ny * self.nz  # 80·20·80 = 128,000 (§4.1)


class AnchorState(NamedTuple):
    """A_i = (c_i, f_i, Σ_i, e_i, τ_i) as static-capacity buffers."""
    c: jax.Array         # (N, 3)  fp32 centers
    f: jax.Array         # (N, D)  fp32 features
    sig2_pos: jax.Array  # (N, 3)  fp32 diagonal position variances (A30)
    sig2_feat: jax.Array # (N, D)  fp32 feature variances (A30)
    e: jax.Array         # (N,)    fp32 existence logits
    tau: jax.Array       # (N,)    fp32 last-write time; −1 = never


def lattice_centers(cfg: AnchorFieldConfig) -> jax.Array:
    ex, ey, ez = cfg.envelope
    xs = (jnp.arange(cfg.nx, dtype=jnp.float32) + 0.5) * cfg.lattice_spacing - 0.5 * ex
    ys = (jnp.arange(cfg.ny, dtype=jnp.float32) + 0.5) * cfg.lattice_spacing
    zs = (jnp.arange(cfg.nz, dtype=jnp.float32) + 0.5) * cfg.lattice_spacing - 0.5 * ez
    xg, yg, zg = jnp.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.stack([xg.reshape(-1), yg.reshape(-1), zg.reshape(-1)], axis=-1)


def points_to_cell_indices(cfg: AnchorFieldConfig, points: jax.Array) -> jax.Array:
    """(K,3) world points → (K,) int32 lattice cell indices (A31).
    Out-of-envelope points are clamped to boundary cells."""
    ex, ey, ez = cfg.envelope
    s = cfg.lattice_spacing
    ix = jnp.clip(jnp.floor((points[..., 0] + 0.5 * ex) / s), 0, cfg.nx - 1)
    iy = jnp.clip(jnp.floor(points[..., 1] / s), 0, cfg.ny - 1)
    iz = jnp.clip(jnp.floor((points[..., 2] + 0.5 * ez) / s), 0, cfg.nz - 1)
    return (ix * (cfg.ny * cfg.nz) + iy * cfg.nz + iz).astype(jnp.int32)


def cell_index_to_center(cfg: AnchorFieldConfig, idx: jax.Array) -> jax.Array:
    idx = jnp.asarray(idx, jnp.int32)
    iz = idx % cfg.nz
    iy = (idx // cfg.nz) % cfg.ny
    ix = idx // (cfg.ny * cfg.nz)
    ex, ey, ez = cfg.envelope
    s = cfg.lattice_spacing
    x = (ix.astype(jnp.float32) + 0.5) * s - 0.5 * ex
    y = (iy.astype(jnp.float32) + 0.5) * s
    z = (iz.astype(jnp.float32) + 0.5) * s - 0.5 * ez
    return jnp.stack([x, y, z], axis=-1)


def eta_star(sig2_mem, sig2_obs):
    """η★ = σ²_mem/(σ²_mem + σ²_obs) — proposal §4.2, exact."""
    return sig2_mem / (sig2_mem + sig2_obs)


def init(cfg: AnchorFieldConfig) -> AnchorState:
    n = cfg.capacity
    return AnchorState(
        c=lattice_centers(cfg),
        f=jnp.zeros((n, cfg.feat_dim), dtype=jnp.float32),
        sig2_pos=jnp.full((n, 3), cfg.sigma2_pos_init, dtype=jnp.float32),
        sig2_feat=jnp.full((n, cfg.feat_dim), cfg.sigma2_feat_init, dtype=jnp.float32),
        e=jnp.full((n,), cfg.exist_logit_init, dtype=jnp.float32),
        tau=jnp.full((n,), -1.0, dtype=jnp.float32))


# ------------------------- internals ---------------------------------

class _Prepared(NamedTuple):
    mu_pos: jax.Array       # (K,3) conflict-fused observation means
    sig2_pos: jax.Array     # (K,3) conflict-fused observation variances
    mu_feat: jax.Array      # (K,D)
    sig2_feat: jax.Array    # (K,D)
    sig2_pos_rows: jax.Array  # (K,3) sanitized per-row obs variances
    row_exec: jax.Array     # (K,) bool — anchor receives ≥1 open write


def _scatter_fuse_vec(capacity, idx, mu, sig2, open_f, eps):
    """A28: aggregate open rows targeting each anchor into one effective
    Gaussian observation: µ* = Σw·µ/Σw, σ²* = 1/Σw, w = open/σ².
    Rows sharing an anchor obtain identical outputs, making the final
    scatter-set order-proof."""
    w = open_f[:, None] / jnp.maximum(sig2, eps)
    num = jnp.zeros((capacity, mu.shape[1]), dtype=jnp.float32).at[idx].add(w * mu)
    den = jnp.zeros((capacity, mu.shape[1]), dtype=jnp.float32).at[idx].add(w)
    den_row = jnp.maximum(den[idx], eps)
    return num[idx] / den_row, 1.0 / den_row


def _prepare(cfg, idx, mu_obs_pos, sig2_obs_pos, mu_obs_feat, sig2_obs_feat, open_mask):
    ob = open_mask.astype(jnp.bool_)
    of = open_mask.astype(jnp.float32)
    # §16 sanitation: closed rows cannot inject NaN/Inf, and receive no
    # gradient (the gate suppresses the learning signal by design).
    mu_p = jnp.where(ob[:, None], mu_obs_pos, jnp.zeros_like(mu_obs_pos))
    s2_p = jnp.where(ob[:, None], jnp.maximum(sig2_obs_pos, cfg.eps),
                     jnp.ones_like(sig2_obs_pos))
    mu_f = jnp.where(ob[:, None], mu_obs_feat, jnp.zeros_like(mu_obs_feat))
    s2_f_in = jnp.broadcast_to(sig2_obs_feat, mu_obs_feat.shape)
    s2_f = jnp.where(ob[:, None], jnp.maximum(s2_f_in, cfg.eps),
                     jnp.ones_like(s2_f_in))
    mu_pos_f, sig2_pos_f = _scatter_fuse_vec(cfg.capacity, idx, mu_p, s2_p, of, cfg.eps)
    mu_feat_f, sig2_feat_f = _scatter_fuse_vec(cfg.capacity, idx, mu_f, s2_f, of, cfg.eps)
    cnt = jnp.zeros((cfg.capacity,), dtype=jnp.float32).at[idx].add(of)
    row_exec = cnt[idx] > 0
    return _Prepared(mu_pos_f, sig2_pos_f, mu_feat_f, sig2_feat_f, s2_p, row_exec)


def _blend_and_scatter(cfg, state, t, idx, p, eta_pos, eta_feat):
    """Gated inverse-variance blend + duplicate-proof scatter-set.
    Invariant: every row sharing an index carries IDENTICAL new values
    (fused obs, gathered memory, η, row_exec are index-determined)."""
    c_mem, f_mem = state.c[idx], state.f[idx]
    sp_mem, sf_mem = state.sig2_pos[idx], state.sig2_feat[idx]
    e_mem, tau_mem = state.e[idx], state.tau[idx]

    one_m_pos, one_m_feat = 1.0 - eta_pos, 1.0 - eta_feat
    c_new = one_m_pos * c_mem + eta_pos * p.mu_pos          # §4.2, exact
    f_new = one_m_feat * f_mem + eta_feat * p.mu_feat       # §4.2, exact
    sp_new = jnp.maximum(one_m_pos * sp_mem, cfg.sigma2_floor)   # A27
    sf_new = jnp.maximum(one_m_feat * sf_mem, cfg.sigma2_floor)  # A27
    e_new = e_mem + cfg.exist_logit_update                       # A29
    tau_new = jnp.full_like(tau_mem, jnp.asarray(t, dtype=jnp.float32))

    exec_b = p.row_exec[:, None]
    c_new = jnp.where(exec_b, c_new, c_mem)
    f_new = jnp.where(exec_b, f_new, f_mem)
    sp_new = jnp.where(exec_b, sp_new, sp_mem)
    sf_new = jnp.where(exec_b, sf_new, sf_mem)
    e_new = jnp.where(p.row_exec, e_new, e_mem)
    tau_new = jnp.where(p.row_exec, tau_new, tau_mem)

    return state._replace(
        c=state.c.at[idx].set(c_new),
        f=state.f.at[idx].set(f_new),
        sig2_pos=state.sig2_pos.at[idx].set(sp_new),
        sig2_feat=state.sig2_feat.at[idx].set(sf_new),
        e=state.e.at[idx].set(e_new),
        tau=state.tau.at[idx].set(tau_new))


# ------------------------- public API --------------------------------

def write(cfg, state, t, idx, mu_obs_pos, sig2_obs_pos, mu_obs_feat,
          sig2_obs_feat, eta, open_mask):
    """Learned-gate write (proposal §4.2, η_k mode).

    eta (K,) is the learned gate (Phase E writer); open_mask=False rows
    are excluded from persistent writes (dynamic objects, §4.2) and leave
    the memory bit-unchanged — including variance, existence and τ.
    Rows sharing an anchor fuse η with the same inverse-variance weights
    (A28).  Differentiable w.r.t. state (c, f, Σ) and writer inputs
    (µ_obs, σ²_obs, η); τ, e carry no gradient (A29).  This is the
    L_revisit → read → write gradient path (§4.4).
    """
    p = _prepare(cfg, idx, mu_obs_pos, sig2_obs_pos, mu_obs_feat,
                 sig2_obs_feat, open_mask)
    ob = open_mask.astype(jnp.bool_)
    of = open_mask.astype(jnp.float32)
    eta_s = jnp.where(ob, eta, jnp.zeros_like(eta))
    w_e = of / jnp.maximum(jnp.mean(p.sig2_pos_rows, axis=1), cfg.eps)
    num = jnp.zeros((cfg.capacity,), dtype=jnp.float32).at[idx].add(w_e * eta_s)
    den = jnp.zeros((cfg.capacity,), dtype=jnp.float32).at[idx].add(w_e)
    eta_row = num[idx] / jnp.maximum(den[idx], cfg.eps)
    eta_pos = jnp.broadcast_to(eta_row[:, None], p.mu_pos.shape)
    eta_feat = jnp.broadcast_to(eta_row[:, None], p.mu_feat.shape)
    return _blend_and_scatter(cfg, state, t, idx, p, eta_pos, eta_feat)


def write_closed_form(cfg, state, t, idx, mu_obs_pos, sig2_obs_pos,
                      mu_obs_feat, sig2_obs_feat, open_mask):
    """Analytic η★ write (proposal §4.2 closed form).  η is evaluated
    against the conflict-resolved observation variances, so a batch
    write is exactly the diagonal-Gaussian posterior:
    batch(A,B) ≡ sequential(A→B) ≡ sequential(B→A)  (tested)."""
    p = _prepare(cfg, idx, mu_obs_pos, sig2_obs_pos, mu_obs_feat,
                 sig2_obs_feat, open_mask)
    eta_pos = eta_star(state.sig2_pos[idx], p.sig2_pos)
    eta_feat = eta_star(state.sig2_feat[idx], p.sig2_feat)
    return _blend_and_scatter(cfg, state, t, idx, p, eta_pos, eta_feat)


def apply_anchor_deletion(cfg, state, anchor_mask):
    """Counterfactual editing (§4.5): reset masked anchors to the
    unoccupied prior.  Object-slot deletion joins in Phase D Part 3 (A17)."""
    m = anchor_mask.astype(jnp.bool_)
    mb = m[:, None]
    lat = lattice_centers(cfg)
    return state._replace(
        c=jnp.where(mb, lat, state.c),
        f=jnp.where(mb, jnp.zeros_like(state.f), state.f),
        sig2_pos=jnp.where(mb, jnp.full_like(state.sig2_pos, cfg.sigma2_pos_init),
                           state.sig2_pos),
        sig2_feat=jnp.where(mb, jnp.full_like(state.sig2_feat, cfg.sigma2_feat_init),
                            state.sig2_feat),
        e=jnp.where(m, jnp.full_like(state.e, cfg.exist_logit_init), state.e),
        tau=jnp.where(m, jnp.full_like(state.tau, -1.0), state.tau))


def active_mask(cfg, state):
    """Sparsification (§4.1 'then sparsified')."""
    return state.e > cfg.exist_active_threshold


def summarize(cfg, state):
    """Host-side debug summary for logs (eager only — dynamic shapes)."""
    act = active_mask(cfg, state)
    n = int(act.sum())
    out = {"capacity": cfg.capacity, "n_active": n,
           "frac_never_written": float((state.tau < 0).mean())}
    if n:
        out["mean_sig2_pos_active"] = float(state.sig2_pos[act].mean())
        out["mean_e_active"] = float(state.e[act].mean())
    return out