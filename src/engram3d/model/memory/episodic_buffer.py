# =====================================================================
# Engram3D · Phase D, Part 3a · short-term episodic latent buffer
# File: src/engram3d/model/memory/episodic_buffer.py
# Status: UNTESTED COMPLETE IMPLEMENTATION.
# Proposal §4.1: hybrid memory = short-term latent buffer | persistent
# 3D anchor field | object slots. This is the short-term component:
# a bounded FIFO of recent latents (A9: capacity 8, most-recent-first).
# =====================================================================
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

__all__ = ["EpisodicBufferConfig", "EpisodicState", "init", "push",
           "read_tokens"]


class EpisodicBufferConfig(NamedTuple):
    capacity: int = 8        # A9
    latent_dim: int = 512    # A37


class EpisodicState(NamedTuple):
    z: jax.Array             # (K, D) fp32, most-recent-first


def init(cfg: EpisodicBufferConfig) -> EpisodicState:
    return EpisodicState(
        z=jnp.zeros((cfg.capacity, cfg.latent_dim), dtype=jnp.float32))


def push(cfg: EpisodicBufferConfig, state: EpisodicState, z_t) -> EpisodicState:
    """Push latent z_t (D,); oldest row evicted. Differentiable w.r.t. z_t."""
    z_t = jnp.asarray(z_t, dtype=jnp.float32)
    return state._replace(
        z=jnp.concatenate([z_t[None, :], state.z[:-1]], axis=0))


def read_tokens(state: EpisodicState) -> jax.Array:
    return state.z