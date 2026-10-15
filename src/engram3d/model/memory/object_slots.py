# =====================================================================
# Engram3D · Phase D, Part 3b · object slots with lifecycle
# File: src/engram3d/model/memory/object_slots.py
# Status: UNTESTED COMPLETE IMPLEMENTATION — validate with
#         tests/test_slots.py BEFORE training use (§6 gate).
# ---------------------------------------------------------------------
# Proposal alignment:
#   §4.1  "Object slots carry instance ID, velocity, and lifecycle
#          logits (existence / insertion), giving the generator an
#          object-permanent state variable it can attend to."
#          Separate dynamic/static streams (InfiniCube precedent).
#   §4.5  counterfactual editing: delete an object's slots.
# Choices: A40 existence decay 0 (permanence by default); A41 eviction
#          policy (empty-first, then lowest existence, matched slots
#          protected); A42 no dead-reckoning (pos = last-sighted);
#          A35 observation source (GT at train / predicted at eval)
#          is a Phase G decision — this module is source-agnostic.
# Precondition: instance ids are unique within one update() call and
#          unique across slots (inductive invariant; GT/simulator data
#          guarantees this — Phase C.2 validates).
# Determinism: stable argsort; allocation order is a pure function of
#          (ids, existence logits) — no data-dependent control flow.
# ======================================================================
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

__all__ = ["ObjectSlotConfig", "SlotState", "SlotObservation", "init",
           "update", "apply_slot_deletion", "occupied_mask",
           "dynamic_mask"]


class ObjectSlotConfig(NamedTuple):
    n_slots: int = 64                 # A10
    feat_dim: int = 128
    pos_blend: float = 0.7            # α — position update on re-sighting
    feat_blend: float = 0.7           # γ — feature update on re-sighting
    vel_ema: float = 0.8              # β — velocity EMA
    dt_floor: float = 1.0             # min Δt (steps) for velocity
    exist_logit_update: float = 2.0   # per positive sighting
    exist_logit_init: float = 2.0     # on insertion
    exist_logit_decay: float = 0.0    # A40: permanence by default
    insert_logit_boost: float = 4.0   # insertion event signal
    insert_logit_decay: float = 0.5   # per unseen step
    empty_exist: float = -1e30


class SlotState(NamedTuple):
    instance_id: jax.Array    # (N,) int32; −1 = empty
    feat: jax.Array           # (N, D) fp32
    pos: jax.Array            # (N, 3) fp32 — last-sighted (A42)
    vel: jax.Array            # (N, 3) fp32
    exist_logit: jax.Array    # (N,) fp32 — lifecycle: existence
    insert_logit: jax.Array   # (N,) fp32 — lifecycle: insertion event
    is_dynamic: jax.Array     # (N,) bool — dynamic/static stream
    tau: jax.Array            # (N,) fp32 — last-sighting time


class SlotObservation(NamedTuple):
    instance_id: jax.Array    # (K,) int32
    feat: jax.Array           # (K, D) fp32
    pos: jax.Array            # (K, 3) fp32
    is_dynamic: jax.Array     # (K,) bool


def init(cfg: ObjectSlotConfig) -> SlotState:
    n, d = cfg.n_slots, cfg.feat_dim
    return SlotState(
        instance_id=jnp.full((n,), -1, dtype=jnp.int32),
        feat=jnp.zeros((n, d), dtype=jnp.float32),
        pos=jnp.zeros((n, 3), dtype=jnp.float32),
        vel=jnp.zeros((n, 3), dtype=jnp.float32),
        exist_logit=jnp.full((n,), cfg.empty_exist, dtype=jnp.float32),
        insert_logit=jnp.zeros((n,), dtype=jnp.float32),
        is_dynamic=jnp.zeros((n,), dtype=jnp.bool_),
        tau=jnp.full((n,), -1.0, dtype=jnp.float32))


def update(cfg: ObjectSlotConfig, state: SlotState, t,
           obs: SlotObservation) -> SlotState:
    """One observation step. Matched ids update (blend pos/feat, EMA vel,
    existence += Δ, insertion event consumed); unmatched ids are inserted
    into the highest-priority available slot (empty first, then lowest
    existence; matched slots protected — A41); unseen occupied slots decay
    insertion/existence logits only (A40/A42: no position extrapolation).
    Differentiable w.r.t. float fields of state and obs; id matching is
    exact equality (non-differentiable by design)."""
    n = cfg.n_slots
    match = ((obs.instance_id[:, None] == state.instance_id[None, :])
             & (state.instance_id[None, :] >= 0))            # (K, N)
    has_match = match.any(axis=1)                            # (K,)
    slot_of = jnp.argmax(match, axis=1).astype(jnp.int32)    # valid iff matched
    unmatched = ~has_match
    rank = jnp.cumsum(unmatched.astype(jnp.int32)) - 1       # 0-based, unmatched
    matched_slot = match.any(axis=0)                         # (N,)
    prio = jnp.where(state.instance_id >= 0, state.exist_logit, cfg.empty_exist)
    prio = jnp.where(matched_slot, 1e30, prio)               # protect matched
    order = jnp.argsort(prio, kind="stable")                 # (N,)
    tgt = jnp.where(has_match, slot_of,
                    order[jnp.clip(rank, 0, n - 1)])         # (K,) distinct

    mb = has_match[:, None]
    old_f, old_p, old_v = state.feat[tgt], state.pos[tgt], state.vel[tgt]
    old_exist, old_tau = state.exist_logit[tgt], state.tau[tgt]
    dt = jnp.maximum(t - old_tau, cfg.dt_floor)
    vel_obs = (obs.pos - old_p) / dt
    vel_new = jnp.where(mb, cfg.vel_ema * old_v
                        + (1.0 - cfg.vel_ema) * vel_obs, jnp.zeros_like(old_v))
    pos_new = jnp.where(mb, (1.0 - cfg.pos_blend) * old_p
                        + cfg.pos_blend * obs.pos, obs.pos)
    feat_new = jnp.where(mb, (1.0 - cfg.feat_blend) * old_f
                         + cfg.feat_blend * obs.feat, obs.feat)
    exist_new = jnp.where(has_match, old_exist + cfg.exist_logit_update,
                          jnp.asarray(cfg.exist_logit_init, jnp.float32))
    insert_new = jnp.where(has_match, jnp.zeros((), jnp.float32),
                           jnp.asarray(cfg.insert_logit_boost, jnp.float32))
    tau_new = jnp.full((obs.instance_id.shape[0],),
                       jnp.asarray(t, jnp.float32))

    st = state._replace(
        instance_id=state.instance_id.at[tgt].set(obs.instance_id),
        feat=state.feat.at[tgt].set(feat_new),
        pos=state.pos.at[tgt].set(pos_new),
        vel=state.vel.at[tgt].set(vel_new),
        exist_logit=state.exist_logit.at[tgt].set(exist_new),
        insert_logit=state.insert_logit.at[tgt].set(insert_new),
        is_dynamic=state.is_dynamic.at[tgt].set(obs.is_dynamic),
        tau=state.tau.at[tgt].set(tau_new))

    touched = jnp.zeros((n,), dtype=jnp.bool_).at[tgt].set(True)
    unseen = (~touched) & (state.instance_id >= 0)
    return st._replace(
        exist_logit=jnp.where(unseen, st.exist_logit - cfg.exist_logit_decay,
                              st.exist_logit),
        insert_logit=jnp.where(unseen, st.insert_logit * cfg.insert_logit_decay,
                               st.insert_logit))


def apply_slot_deletion(cfg: ObjectSlotConfig, state: SlotState,
                        instance_ids) -> SlotState:
    """Counterfactual editing (§4.5): reset the named objects' slots to
    the empty prior. (Anchor-side deletion joins in Phase L per A17.)"""
    ids = jnp.asarray(instance_ids, jnp.int32)
    m = ((state.instance_id[:, None] == ids[None, :]).any(axis=1)
         & (state.instance_id >= 0))
    mb = m[:, None]
    return state._replace(
        instance_id=jnp.where(m, jnp.full_like(state.instance_id, -1),
                              state.instance_id),
        feat=jnp.where(mb, jnp.zeros_like(state.feat), state.feat),
        pos=jnp.where(mb, jnp.zeros_like(state.pos), state.pos),
        vel=jnp.where(mb, jnp.zeros_like(state.vel), state.vel),
        exist_logit=jnp.where(m, jnp.full_like(state.exist_logit,
                                               cfg.empty_exist),
                              state.exist_logit),
        insert_logit=jnp.where(m, jnp.zeros_like(state.insert_logit),
                               state.insert_logit),
        is_dynamic=jnp.where(m, False, state.is_dynamic),
        tau=jnp.where(m, jnp.full_like(state.tau, -1.0), state.tau))


def occupied_mask(state: SlotState) -> jax.Array:
    return state.instance_id >= 0


def dynamic_mask(state: SlotState) -> jax.Array:
    """Dynamic/static stream separation (§4.1, InfiniCube precedent)."""
    return state.is_dynamic & (state.instance_id >= 0)