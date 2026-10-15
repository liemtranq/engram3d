# =====================================================================
# Engram3D · Phase D Part 3 · object-slot & episodic-buffer tests
# File: tests/test_slots.py   (pytest-compatible; plain runner)
# Status: UNTESTED COMPLETE IMPLEMENTATION — run and return output.
# GATES all later training (prompt §6). All expectations analytic.
# ======================================================================
from __future__ import annotations

import os
import sys
import time
import traceback

import jax
import jax.numpy as jnp

try:
    from engram3d.model.memory.object_slots import (
        ObjectSlotConfig, SlotState, SlotObservation, init, update,
        apply_slot_deletion, occupied_mask, dynamic_mask)
    from engram3d.model.memory import episodic_buffer as eb
except Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from object_slots import (ObjectSlotConfig, SlotState, SlotObservation,
                              init, update, apply_slot_deletion,
                              occupied_mask, dynamic_mask)
    import episodic_buffer as eb

CFG = ObjectSlotConfig(n_slots=8, feat_dim=4)
TOL = 1e-6


def obs(ids, feats, poss, dyns, t=None):
    return SlotObservation(instance_id=jnp.asarray(ids, jnp.int32),
                           feat=jnp.asarray(feats, jnp.float32),
                           pos=jnp.asarray(poss, jnp.float32),
                           is_dynamic=jnp.asarray(dyns, jnp.bool_))


# S1 — insertion into an empty slot (deterministic: lowest index first)
def test_insertion():
    st = init(CFG)
    st1 = update(CFG, st, 5.0, obs([101], [[1., 2., 3., 4.]],
                                   [(1., 0., 2.)], [True]))
    assert int(st1.instance_id[0]) == 101
    assert float(jnp.max(jnp.abs(st1.feat[0] - jnp.array([1., 2., 3., 4.])))) < TOL
    assert float(jnp.max(jnp.abs(st1.pos[0] - jnp.array([1., 0., 2.])))) < TOL
    assert float(jnp.max(jnp.abs(st1.vel[0]))) == 0.0
    assert float(st1.exist_logit[0]) == CFG.exist_logit_init
    assert float(st1.insert_logit[0]) == CFG.insert_logit_boost
    assert bool(st1.is_dynamic[0]) and float(st1.tau[0]) == 5.0
    assert bool(jnp.array_equal(st1.instance_id[1:], jnp.full((7,), -1)))


# S2 — matching updates the SAME slot; exact blend/velocity arithmetic
def test_matching_updates_and_velocity():
    st = init(CFG)
    st1 = update(CFG, st, 5.0, obs([101], [[1., 2., 3., 4.]],
                                   [(1., 0., 2.)], [True]))
    st2 = update(CFG, st1, 10.0, obs([101], [[2., 2., 2., 2.]],
                                     [(1., 0., 3.)], [False]))
    assert int((st2.instance_id == 101).sum()) == 1        # no duplication
    assert float(jnp.max(jnp.abs(st2.pos[0] - jnp.array([1., 0., 2.7])))) < TOL
    assert float(jnp.max(jnp.abs(st2.vel[0] - jnp.array([0., 0., 0.04])))) < TOL
    assert float(jnp.max(jnp.abs(
        st2.feat[0] - jnp.array([1.7, 2.0, 2.3, 2.6])))) < TOL
    assert float(st2.exist_logit[0]) == CFG.exist_logit_init + CFG.exist_logit_update
    assert float(st2.insert_logit[0]) == 0.0               # event consumed
    assert float(st2.tau[0]) == 10.0
    assert not bool(st2.is_dynamic[0])                     # stream can flip


# S3 — velocity EMA over a second interval
def test_velocity_ema():
    st = init(CFG)
    st1 = update(CFG, st, 5.0, obs([101], [[0., 0., 0., 0.]],
                                   [(1., 0., 2.)], [True]))
    st2 = update(CFG, st1, 10.0, obs([101], [[0., 0., 0., 0.]],
                                     [(1., 0., 3.)], [True]))
    st3 = update(CFG, st2, 20.0, obs([101], [[0., 0., 0., 0.]],
                                     [(1., 0., 5.)], [True]))
    assert float(jnp.max(jnp.abs(st3.vel[0] - jnp.array([0., 0., 0.078])))) < TOL


# S4 — occlusion persistence (A40): unseen slot persists, insert decays
def test_occlusion_persistence():
    st = init(CFG)
    st1 = update(CFG, st, 5.0, obs([101], [[1., 1., 1., 1.]],
                                   [(0., 0., 1.)], [True]))
    st2 = update(CFG, st1, 6.0, obs([202], [[2., 2., 2., 2.]],
                                    [(0., 0., 2.)], [False]))
    assert int(st2.instance_id[0]) == 101                  # persists
    assert float(jnp.max(jnp.abs(st2.pos[0] - st1.pos[0]))) < TOL
    assert float(jnp.max(jnp.abs(st2.vel[0] - st1.vel[0]))) < TOL
    assert float(jnp.max(jnp.abs(st2.feat[0] - st1.feat[0]))) < TOL
    assert float(st2.exist_logit[0]) == float(st1.exist_logit[0])
    assert abs(float(st2.insert_logit[0])
               - CFG.insert_logit_boost * CFG.insert_logit_decay) < TOL
    assert int(st2.instance_id[1]) == 202


# S5 — capacity eviction: lowest existence evicted (A41)
def test_eviction_policy():
    cfg = ObjectSlotConfig(n_slots=3, feat_dim=4)
    st = init(cfg)
    st = update(cfg, st, 1.0, obs([1], [[0., 0., 0., 0.]], [(0., 0., 1.)], [True]))
    st = update(cfg, st, 2.0, obs([2], [[0., 0., 0., 0.]], [(0., 0., 2.)], [True]))
    st = update(cfg, st, 3.0, obs([3], [[0., 0., 0., 0.]], [(0., 0., 3.)], [True]))
    st = update(cfg, st, 10.0, obs([2], [[0., 0., 0., 0.]], [(0., 0., 2.5)], [True]))
    st = update(cfg, st, 11.0, obs([4], [[0., 0., 0., 0.]], [(0., 0., 4.)], [True]))
    ids = jnp.asarray(st.instance_id).tolist()
    assert ids == [4, 2, 3]        # id1 (lowest existence) evicted
    assert float(st.exist_logit[1]) == 2.0 + 2.0            # B re-sighted


# S6 — dynamic/static stream masks
def test_stream_masks():
    st = init(CFG)
    st = update(CFG, st, 1.0, obs([7, 8], [[0.] * 4, [0.] * 4],
                                  [(0., 0., 1.), (0., 0., 2.)], [True, False]))
    assert bool(jnp.array_equal(dynamic_mask(st),
                                jnp.array([True, False] + [False] * 6)))
    assert int(occupied_mask(st).sum()) == 2


# S7 — counterfactual slot deletion resets to the empty prior (§4.5)
def test_counterfactual_slot_deletion():
    st = init(CFG)
    st1 = update(CFG, st, 5.0, obs([101, 202], [[1.] * 4, [2.] * 4],
                                   [(0., 0., 1.), (0., 0., 2.)], [True, True]))
    st2 = apply_slot_deletion(CFG, st1, jnp.array([202], jnp.int32))
    st0 = init(CFG)
    for fld in SlotState._fields:
        assert bool(jnp.array_equal(getattr(st2, fld)[1], getattr(st0, fld)[1])), fld
        assert bool(jnp.array_equal(getattr(st2, fld)[0], getattr(st1, fld)[0])), fld


# S8 — gradient flow to obs and matched-slot state; untouched slot: zero
def test_gradient_flow():
    st1 = update(CFG, init(CFG), 5.0,
                 obs([101], [[1., 2., 3., 4.]], [(1., 0., 2.)], [True]))
    obs2 = obs([101, 202], [[3., 3., 3., 3.], [4., 4., 4., 4.]],
               [(1., 0., 3.), (0., 0., 2.)], [True, False])

    def loss(state_f, state_p, obs_f, obs_p):
        stt = st1._replace(feat=state_f, pos=state_p)
        o = obs2._replace(feat=obs_f, pos=obs_p)
        return jnp.sum(update(CFG, stt, 10.0, o).feat)

    g_sf, g_sp, g_of, g_op = jax.grad(loss)(st1.feat, st1.pos,
                                            obs2.feat, obs2.pos)
    assert float(jnp.max(jnp.abs(g_sf[0]))) > 0.0          # matched slot
    assert float(jnp.max(jnp.abs(g_sf[1]))) == 0.0         # untouched slot
    assert float(jnp.max(jnp.abs(g_of))) > 0.0             # both obs rows
    assert float(jnp.max(jnp.abs(g_op))) > 0.0             # pos & vel paths


# S9 — JIT == eager (all fields, exact for ints)
def test_jit_matches_eager():
    st = update(CFG, init(CFG), 5.0,
                obs([101], [[1., 2., 3., 4.]], [(1., 0., 2.)], [True]))
    o2 = obs([101, 202], [[3., 3., 3., 3.], [4., 4., 4., 4.]],
             [(1., 0., 3.), (0., 0., 2.)], [True, False])
    ea = update(CFG, st, 10.0, o2)
    ji = jax.jit(lambda s, o: update(CFG, s, 10.0, o))(st, o2)
    for fld in SlotState._fields:
        a, b = getattr(ea, fld), getattr(ji, fld)
        if fld in ("instance_id", "is_dynamic"):
            assert bool(jnp.array_equal(a, b)), fld
        else:
            assert float(jnp.max(jnp.abs(a - b))) < 1e-6, fld


# E1–E3 — episodic buffer FIFO semantics + gradient
def test_buffer_fifo_and_eviction():
    cfg = eb.EpisodicBufferConfig(capacity=4, latent_dim=8)
    st = eb.init(cfg)
    st = eb.push(cfg, st, jnp.ones(8))
    st = eb.push(cfg, st, 2.0 * jnp.ones(8))
    st = eb.push(cfg, st, 3.0 * jnp.ones(8))
    assert bool(jnp.array_equal(eb.read_tokens(st)[:, 0],
                                jnp.array([3., 2., 1., 0.])))
    st = eb.push(cfg, st, 4.0 * jnp.ones(8))
    st = eb.push(cfg, st, 5.0 * jnp.ones(8))                # z1 evicted
    assert bool(jnp.array_equal(eb.read_tokens(st)[:, 0],
                                jnp.array([5., 4., 3., 2.])))


def test_buffer_gradient():
    cfg = eb.EpisodicBufferConfig(capacity=4, latent_dim=8)
    z = jnp.ones(8)
    g = jax.grad(lambda zt: jnp.sum(eb.push(cfg, eb.init(cfg), zt).z))(z)
    assert bool(jnp.allclose(g, jnp.ones(8)))


def run_benchmark(iters: int = 50):
    cfg = ObjectSlotConfig()                               # 64 slots, D=128
    st = init(cfg)
    key = jax.random.PRNGKey(0)
    o = SlotObservation(
        instance_id=jnp.arange(1, 17, dtype=jnp.int32),
        feat=jax.random.normal(key, (16, cfg.feat_dim)),
        pos=jax.random.normal(jax.random.fold_in(key, 1), (16, 3)),
        is_dynamic=jnp.ones((16,), jnp.bool_))
    fn = jax.jit(lambda s, ob: update(cfg, s, 100.0, ob))
    out = fn(st, o); jax.block_until_ready(out.feat)
    t0 = time.perf_counter()
    for i in range(iters):
        out = fn(st, o)
    jax.block_until_ready(out.feat)
    ms = (time.perf_counter() - t0) / iters * 1e3
    print("Engram3D object-slot update benchmark")
    print(f"  backend={jax.default_backend()}  n_slots={cfg.n_slots}  K=16")
    print(f"  steady-state={ms:.3f} ms/update")


TESTS = [("insertion", test_insertion),
         ("matching_updates_and_velocity", test_matching_updates_and_velocity),
         ("velocity_ema", test_velocity_ema),
         ("occlusion_persistence", test_occlusion_persistence),
         ("eviction_policy", test_eviction_policy),
         ("stream_masks", test_stream_masks),
         ("counterfactual_slot_deletion", test_counterfactual_slot_deletion),
         ("gradient_flow", test_gradient_flow),
         ("jit_matches_eager", test_jit_matches_eager),
         ("buffer_fifo_and_eviction", test_buffer_fifo_and_eviction),
         ("buffer_gradient", test_buffer_gradient)]


def run_all():
    failed = []
    for name, fn in TESTS:
        t0 = time.time()
        try:
            fn()
            print(f"[PASS] {name}  ({time.time() - t0:.2f}s)")
        except Exception as exc:
            failed.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("=" * 60)
    print(f"PHASE D3 SLOT/BUFFER TESTS: {len(TESTS) - len(failed)}/{len(TESTS)} PASS"
          + ("  — DO NOT PROCEED (prompt §6)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_benchmark()
        sys.exit(0)
    sys.exit(run_all())