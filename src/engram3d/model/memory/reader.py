# =====================================================================
# Engram3D · Phase D, Part 2 · pose-conditioned frustum memory read
# File: src/engram3d/model/memory/reader.py
# Status: UNTESTED COMPLETE IMPLEMENTATION — validate with
#         tests/test_reader.py BEFORE training use (prompt §6 gate).
# ---------------------------------------------------------------------
# Proposal alignment:
#   §4.3  "Frustum gather at the current pose → 256 memory tokens →
#          injected into the AR generator as residual conditioning."
#   §4.4  R(Read(M_{t-1}, p_t)) — the revisit-consistency render head
#          consumes this read. Gradient path render head → read →
#          write: selection (top-k) is non-differentiable; gradients
#          flow through the gathered anchor VALUES — exactly the
#          "memory learns what to remember" mechanism.
#   §4.4  anti-bypass: memory-frustum dropout p = 0.15 (A38).
# Choices: A11' ranking e_i/(1+d_i); A28' log σ² in payload, explicit
#          read-side inverse-variance fusion deferred to render head
#          (Phase G); ADR-005 zero-init output projection (zero-conv
#          insurance, §4.3): memory conditioning starts as exact zero.
# Camera convention: camera frame x=right, y=up, z=forward; R maps
# camera→world (columns = camera axes in world); world→camera = Rᵀ.
# yaw=0 → forward +z; yaw=π/2 → forward +x; pitch>0 → up (A39: THOR
# sign conventions calibrated in Phase C.2).
# =====================================================================
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

try:
    from engram3d.model.memory.anchor_field import AnchorFieldConfig, AnchorState
except Exception:  # flat Kaggle layout
    from anchor_field import AnchorFieldConfig, AnchorState

__all__ = ["FrustumConfig", "FrustumRead", "pose_from_yaw_pitch",
           "payload_dim", "init_embed_params", "read", "embed_tokens",
           "apply_read_dropout"]


class FrustumConfig(NamedTuple):
    n_tokens: int = 256              # proposal §4.3
    fov_h_deg: float = 90.0          # A36 — must match Phase C.2 renderer
    fov_v_deg: float = 90.0          # A36 (square frames)
    near: float = 0.1                # m
    far: float = 5.0                 # m (indoor visibility)
    feat_dim: int = 128              # must equal AnchorFieldConfig.feat_dim
    token_dim: int = 512             # A37 — generator conditioning width
    hidden_dim: int = 256
    max_age_steps: float = 1000.0    # age normalization (τ; §4.1)


class FrustumRead(NamedTuple):
    idx: jax.Array       # (T,) int32 — selected anchor indices
    valid: jax.Array     # (T,) bool  — False = null-token padding
    score: jax.Array     # (T,) fp32  — descending; < = padding
    payload: jax.Array   # (T, feat_dim+9) fp32, invalid rows zeroed
    # payload layout: [ f (D) | rel_pos (3) | log σ²_pos (3)
    #                  | log mean σ²_feat (1) | e (1) | age (1) ]


def payload_dim(cfg_f: FrustumConfig) -> int:
    return cfg_f.feat_dim + 9


def pose_from_yaw_pitch(pos, yaw, pitch):
    """(pos (3,), yaw, pitch) → camera-to-world rotation R (3,3)."""
    sy, cy = jnp.sin(yaw), jnp.cos(yaw)
    sp, cp = jnp.sin(pitch), jnp.cos(pitch)
    f = jnp.stack([sy * cp, sp, cy * cp])          # forward
    r = jnp.stack([cy, jnp.asarray(0.0), -sy])     # right
    u = jnp.cross(f, r)                            # up
    return jnp.stack([r, u, f], axis=-1)           # columns


def init_embed_params(key, cfg_f: FrustumConfig):
    """ADR-005: w2/b2/null zero-init ⇒ tokens ≡ 0 at init (zero-conv);
    w2 still receives gradient (via hidden activations) from step 0."""
    d_in = payload_dim(cfg_f)
    k1, _ = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (d_in, cfg_f.hidden_dim)) * (d_in ** -0.5),
        "b1": jnp.zeros((cfg_f.hidden_dim,), jnp.float32),
        "w2": jnp.zeros((cfg_f.hidden_dim, cfg_f.token_dim), jnp.float32),
        "b2": jnp.zeros((cfg_f.token_dim,), jnp.float32),
        "null_embed": jnp.zeros((cfg_f.token_dim,), jnp.float32),
    }


def read(cfg_f: FrustumConfig, cfg_a: AnchorFieldConfig, state: AnchorState,
         pos, R, t):
    """Frustum gather (§4.3). Static shapes: exactly n_tokens returned.
    Differentiable w.r.t. state (f, Σ, e), pos, R, t via gathered values;
    selection itself is top-k (non-differentiable, deterministic per
    compiled executable; tests avoid score ties)."""
    if cfg_f.feat_dim != cfg_a.feat_dim:
        raise ValueError("cfg_f.feat_dim must equal cfg_a.feat_dim")
    if cfg_f.n_tokens > cfg_a.capacity:
        raise ValueError("n_tokens exceeds anchor capacity")

    act = state.e > cfg_a.exist_active_threshold
    rel = (state.c - pos) @ R                        # world→camera (N,3)
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    tan_h = jnp.tan(jnp.radians(cfg_f.fov_h_deg) / 2.0)
    tan_v = jnp.tan(jnp.radians(cfg_f.fov_v_deg) / 2.0)
    inside = ((z >= cfg_f.near) & (z <= cfg_f.far)
              & (jnp.abs(x) <= z * tan_h) & (jnp.abs(y) <= z * tan_v))
    d = jnp.sqrt(jnp.sum(rel * rel, axis=-1))
    score = jnp.where(act & inside, state.e / (1.0 + d),
                      jnp.asarray(-1e30, jnp.float32))
    top_score, top_idx = jax.lax.top_k(score, cfg_f.n_tokens)
    valid = top_score > -1e29

    sel_lsf = jnp.log(jnp.maximum(
        jnp.mean(state.sig2_feat[top_idx], axis=-1), 1e-12))
    payload = jnp.concatenate([
        state.f[top_idx],                            # (T, D)
        rel[top_idx],                                # (T, 3)
        jnp.log(jnp.maximum(state.sig2_pos[top_idx], 1e-12)),  # (T, 3)
        sel_lsf[:, None],                            # (T, 1)
        state.e[top_idx][:, None],                   # (T, 1)
        (jnp.maximum(t - state.tau[top_idx], 0.0) / cfg_f.max_age_steps)[:, None],
    ], axis=-1)
    payload = jnp.where(valid[:, None], payload, jnp.zeros_like(payload))
    return FrustumRead(idx=top_idx, valid=valid, score=top_score,
                       payload=payload)


def embed_tokens(params, cfg_f: FrustumConfig, payload, valid):
    """(T, feat_dim+9) → (T, token_dim). Valid rows: MLP(payload);
    invalid rows: learned null_embed (both ≡ 0 at init, ADR-005)."""
    h = jax.nn.gelu(payload @ params["w1"] + params["b1"])
    tok = h @ params["w2"] + params["b2"]
    return jnp.where(valid[:, None], tok, params["null_embed"][None, :])


def apply_read_dropout(rng, tokens, valid, p=0.15, mode="token"):
    """Anti-bypass protocol (§4.4): memory-frustum dropout p=0.15.
    mode="token": each valid token independently dropped (→ hard zero).
    mode="read":  the whole read dropped with prob p.
    `mode` must be a Python string constant under jit (two compiles)."""
    if mode == "read":
        drop = jax.random.bernoulli(rng, p)
        keep = jnp.where(drop, jnp.zeros_like(valid), valid)
    else:
        keep = jax.random.bernoulli(rng, 1.0 - p, (tokens.shape[0],)) & valid
    return jnp.where(keep[:, None], tokens, jnp.zeros_like(tokens)), keep