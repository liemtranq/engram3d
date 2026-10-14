# =====================================================================
# Engram3D · Phase D Part 1 · anchor-field correctness tests
# File: tests/test_anchor_field.py  (pytest-compatible; plain runner)
# Status: UNTESTED COMPLETE IMPLEMENTATION — run and return output.
# All expectations are analytically derived (prompt §6).  These tests
# GATE all later training: "Do not proceed to expensive training until
# the memory subsystem passes these tests."
# =====================================================================
from __future__ import annotations

import os
import sys
import time
import traceback

import jax
import jax.numpy as jnp

try:
    from engram3d.model.memory.anchor_field import (
        AnchorFieldConfig, AnchorState, lattice_centers,
        points_to_cell_indices, cell_index_to_center, eta_star,
        init, write, write_closed_form, apply_anchor_deletion, active_mask)
except Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anchor_field import (
        AnchorFieldConfig, AnchorState, lattice_centers,
        points_to_cell_indices, cell_index_to_center, eta_star,
        init, write, write_closed_form, apply_anchor_deletion, active_mask)

CFG = AnchorFieldConfig()                     # proposal defaults (§4.1)
SMALL = AnchorFieldConfig(feat_dim=4)
TOL = 1e-6


def _eq(a, b) -> bool:
    return bool(jnp.array_equal(a, b))


# T1 — lattice geometry (§4.1: 0.15 m lattice, 12×12×3 m envelope)
def test_lattice_geometry():
    c = lattice_centers(CFG)
    n = CFG.capacity
    assert c.shape == (n, 3) and n == 128_000                 # 80·20·80
    assert float(jnp.max(jnp.abs(c[0] - jnp.array([-5.925, 0.075, -5.925])))) < TOL
    assert float(jnp.max(jnp.abs(c[n - 1] - jnp.array([5.925, 2.925, 5.925])))) < TOL
    assert float(jnp.max(jnp.abs((c[CFG.ny * CFG.nz] - c[0]) - jnp.array([0.15, 0., 0.])))) < TOL
    assert float(jnp.max(jnp.abs((c[CFG.nz] - c[0]) - jnp.array([0., 0.15, 0.])))) < TOL
    assert float(jnp.max(jnp.abs((c[1] - c[0]) - jnp.array([0., 0., 0.15])))) < TOL


# T2 — index mapping: round-trip, hand-derived cell, clamping
def test_index_roundtrip_and_clamping():
    n = CFG.capacity
    assert _eq(points_to_cell_indices(CFG, lattice_centers(CFG)),
               jnp.arange(n, dtype=jnp.int32))
    assert int(points_to_cell_indices(CFG, jnp.array([[0., 0., 0.]]))[0]) == 64040
    assert int(points_to_cell_indices(CFG, jnp.array([[100., 100., 100.]]))[0]) == n - 1
    picks = jnp.array([0, 1, 80, 1600, 64040, n - 1], jnp.int32)
    cc = cell_index_to_center(CFG, picks)
    lc = lattice_centers(CFG)
    for k, i in enumerate([0, 1, 80, 1600, 64040, n - 1]):
        assert float(jnp.max(jnp.abs(cc[k] - lc[i]))) < TOL


# T3 — η★ is the Gaussian MLE; A27 ⇔ standard posterior at η★
def test_eta_star_is_gaussian_mle():
    s2m, s2o = 0.7, 1.3
    eta = float(eta_star(s2m, s2o))
    assert abs(eta - 0.35) < TOL                               # 0.7/(0.7+1.3)
    mu_m, mu_o = 2.0, -1.0
    assert abs((1 - eta) * mu_m + eta * mu_o
               - (s2o * mu_m + s2m * mu_o) / (s2m + s2o)) < TOL
    assert abs((1 - eta) * s2m - s2m * s2o / (s2m + s2o)) < TOL


# T4 — closed-form single write = exact Gaussian posterior
def test_closed_form_single_row_matches_gaussian():
    st = init(SMALL)
    idx = jnp.array([5], jnp.int32)
    st1 = write_closed_form(SMALL, st, 1.0, idx,
                            jnp.array([[1.0, 1.5, 2.0]]),
                            jnp.array([[0.25, 0.25, 0.25]]),
                            jnp.full((1, SMALL.feat_dim), 0.8),
                            jnp.array([[0.25]]),
                            jnp.array([True]))
    exp_c = (0.25 * st.c[5] + 1.0 * jnp.array([1.0, 1.5, 2.0])) / 1.25
    assert float(jnp.max(jnp.abs(st1.c[5] - exp_c))) < TOL
    assert float(jnp.max(jnp.abs(st1.sig2_pos[5] - 0.2))) < TOL      # 1·0.25/1.25
    assert float(jnp.max(jnp.abs(st1.f[5] - 0.64))) < TOL            # (0.25·0+1·0.8)/1.25
    assert float(jnp.max(jnp.abs(st1.sig2_feat[5] - 0.2))) < TOL
    assert float(st1.e[5]) == float(SMALL.exist_logit_init + SMALL.exist_logit_update)
    assert float(st1.tau[5]) == 1.0
    assert bool(active_mask(SMALL, st1)[5]) and not bool(active_mask(SMALL, st1)[0])


# T5 — associativity & order-invariance of batch writes (A28)
def test_batch_equals_sequential_and_order_invariant():
    D = SMALL.feat_dim
    idx2, idx1 = jnp.array([9, 9], jnp.int32), jnp.array([9], jnp.int32)
    z3, o3 = jnp.zeros((1, 3)), jnp.ones((1, 3))
    st = init(SMALL)
    muA, muB = jnp.zeros((1, D)), jnp.ones((1, D))
    sA = sB = jnp.ones((1, 1))
    stb = write_closed_form(SMALL, st, 1.0, idx2, jnp.zeros((2, 3)),
                            jnp.ones((2, 3)), jnp.stack([muA[0], muB[0]]),
                            jnp.ones((2, 1)), jnp.array([True, True]))
    stA = write_closed_form(SMALL, st, 1.0, idx1, z3, o3, muA, sA, jnp.array([True]))
    stAB = write_closed_form(SMALL, stA, 2.0, idx1, z3, o3, muB, sB, jnp.array([True]))
    stB = write_closed_form(SMALL, st, 1.0, idx1, z3, o3, muB, sB, jnp.array([True]))
    stBA = write_closed_form(SMALL, stB, 2.0, idx1, z3, o3, muA, sA, jnp.array([True]))
    stb2 = write_closed_form(SMALL, st, 1.0, idx2, jnp.zeros((2, 3)),
                             jnp.ones((2, 3)), jnp.stack([muB[0], muA[0]]),
                             jnp.ones((2, 1)), jnp.array([True, True]))
    # prior N(0,1) + obs N(0,1) + obs N(1,1):  mean = 1/3, var = 1/3
    for res in (stb, stAB, stBA, stb2):
        assert float(jnp.max(jnp.abs(res.f[9] - 1.0 / 3.0))) < TOL
        assert float(jnp.max(jnp.abs(res.sig2_feat[9] - 1.0 / 3.0))) < TOL
    for fld in AnchorState._fields:                             # order-invariance
        assert _eq(getattr(stb, fld), getattr(stb2, fld)), fld


# T6 — learned η blends exactly; A27 variance rule
def test_learned_eta_blends_and_A27_variance():
    st = init(SMALL)
    st1 = write(SMALL, st, 5.0, jnp.array([3], jnp.int32),
                jnp.array([[1., 2., 3.]]), jnp.ones((1, 3)),
                jnp.full((1, SMALL.feat_dim), 2.0), jnp.ones((1, 1)),
                jnp.array([0.3]), jnp.array([True]))
    assert float(jnp.max(jnp.abs(st1.f[3] - 0.6))) < TOL            # 0.7·0 + 0.3·2
    assert float(jnp.max(jnp.abs(st1.sig2_feat[3] - 0.7))) < TOL   # (1−η)σ²_mem
    assert float(jnp.max(jnp.abs(st1.c[3] - (0.7 * st.c[3] + 0.3 * jnp.array([1., 2., 3.]))))) < TOL
    assert float(jnp.max(jnp.abs(st1.sig2_pos[3] - 0.7))) < TOL


# T7 — η = 1 overwrites; σ² floor; activation
def test_eta_one_overwrites_with_floor():
    st = init(SMALL)
    st1 = write(SMALL, st, 2.0, jnp.array([6], jnp.int32),
                jnp.array([[1., 2., 3.]]), jnp.ones((1, 3)),
                jnp.full((1, SMALL.feat_dim), 5.0), jnp.ones((1, 1)),
                jnp.array([1.0]), jnp.array([True]))
    assert float(jnp.max(jnp.abs(st1.c[6] - jnp.array([1., 2., 3.])))) < TOL
    assert float(jnp.max(jnp.abs(st1.f[6] - 5.0))) < TOL
    assert float(jnp.max(jnp.abs(st1.sig2_pos[6] - SMALL.sigma2_floor))) < TOL
    assert float(jnp.max(jnp.abs(st1.sig2_feat[6] - SMALL.sigma2_floor))) < TOL


# T8 — gate closed ⇒ state BITWISE unchanged (persistence, §4.2)
def test_gate_closed_leaves_state_bitwise_unchanged():
    for fn, extra in ((write, {"eta": jnp.array([0.9])}),
                      (write_closed_form, {})):
        st = init(SMALL)
        st1 = fn(SMALL, st, 7.0, jnp.array([4], jnp.int32),
                 jnp.array([[1., 1., 1.]]), jnp.full((1, 3), 1e-9),
                 jnp.ones((1, SMALL.feat_dim)), jnp.full((1, 1), 1e-9),
                 open_mask=jnp.array([False]), **extra)
        for fld in AnchorState._fields:
            assert _eq(getattr(st, fld), getattr(st1, fld)), fld


# T9 — closed rows are excluded from fusion (dynamic-object exclusion)
def test_closed_row_excluded_from_fusion():
    st = init(SMALL)
    idx = jnp.array([7, 7], jnp.int32)
    common = dict(mu_obs_pos=jnp.zeros((2, 3)), sig2_obs_pos=jnp.ones((2, 3)),
                  mu_obs_feat=jnp.stack([jnp.full((SMALL.feat_dim,), 5.0),
                                         jnp.full((SMALL.feat_dim,), 100.0)]),
                  sig2_obs_feat=jnp.ones((2, 1)))
    st1 = write(SMALL, st, 1.0, idx, eta=jnp.array([0.5, 0.5]),
                open_mask=jnp.array([True, False]), **common)
    st2 = write_closed_form(SMALL, st, 1.0, idx,
                            open_mask=jnp.array([True, False]), **common)
    for res in (st1, st2):                                          # only open row counts
        assert float(jnp.max(jnp.abs(res.f[7] - 2.5))) < TOL
        assert float(jnp.max(jnp.abs(res.sig2_feat[7] - 0.5))) < TOL


# T10 — writes touch only targeted anchors (persistence)
def test_write_does_not_touch_other_anchors():
    st = init(SMALL)
    idx = jnp.arange(100, 105, dtype=jnp.int32)
    K = idx.shape[0]
    st1 = write(SMALL, st, 3.0, idx, jnp.ones((K, 3)), jnp.ones((K, 3)),
                jnp.ones((K, SMALL.feat_dim)), jnp.ones((K, 1)),
                jnp.full((K,), 0.5), jnp.ones((K,), jnp.bool_))
    for fld in AnchorState._fields:
        a, b = getattr(st, fld), getattr(st1, fld)
        assert _eq(a[:100], b[:100]) and _eq(a[105:], b[105:]), fld
    assert not _eq(st.f[100:105], st1.f[100:105])


# T11 — counterfactual deletion resets to the unoccupied prior (§4.5)
def test_counterfactual_deletion_resets_to_prior():
    st = init(SMALL)
    idx = jnp.array([10, 11], jnp.int32)
    st1 = write(SMALL, st, 1.0, idx, jnp.full((2, 3), 0.5), jnp.ones((2, 3)),
                jnp.ones((2, SMALL.feat_dim)), jnp.ones((2, 1)),
                jnp.full((2,), 0.8), jnp.array([True, True]))
    mask = (jnp.arange(SMALL.capacity) == 11)
    st2 = apply_anchor_deletion(SMALL, st1, mask)
    st0 = init(SMALL)
    for fld in AnchorState._fields:
        assert _eq(getattr(st2, fld)[11], getattr(st0, fld)[11]), fld
        assert _eq(getattr(st2, fld)[10], getattr(st1, fld)[10]), fld


# T12 — gradient reaches writer & memory; closed rows get ZERO signal
def test_gradient_reaches_writer_and_memory_not_closed_rows():
    st0 = init(SMALL)
    idx = jnp.array([12, 13, 14], jnp.int32)
    K = 3
    open_mask = jnp.array([True, True, False])
    mu_f0 = jnp.ones((K, SMALL.feat_dim))
    eta0 = jnp.full((K,), 0.5)

    def loss_writer(mu_feat, eta):
        st = write(SMALL, st0, 1.0, idx, jnp.zeros((K, 3)), jnp.ones((K, 3)),
                   mu_feat, jnp.ones((K, 1)), eta, open_mask)
        return jnp.sum(st.f[idx])

    g_mu, g_eta = jax.grad(loss_writer, argnums=(0, 1))(mu_f0, eta0)
    assert float(jnp.max(jnp.abs(g_mu[:2]))) > 0.0
    assert float(jnp.max(jnp.abs(g_mu[2]))) == 0.0        # closed row: no signal
    assert float(jnp.max(jnp.abs(g_eta[:2]))) > 0.0
    assert float(jnp.max(jnp.abs(g_eta[2]))) == 0.0

    def loss_mem(st):
        st2 = write(SMALL, st, 1.0, idx, jnp.zeros((K, 3)), jnp.ones((K, 3)),
                    mu_f0, jnp.ones((K, 1)), eta0, open_mask)
        return jnp.sum(st2.f[idx])

    gst = jax.grad(loss_mem)(st0)
    assert float(jnp.max(jnp.abs(gst.f[12]))) > 0.0       # read→write path
    assert float(jnp.max(jnp.abs(gst.f[0]))) == 0.0

    def loss_cf(st):                                      # σ² enters η★ ⇒ grad
        st2 = write_closed_form(SMALL, st, 1.0, idx, jnp.zeros((K, 3)),
                                jnp.ones((K, 3)), mu_f0, jnp.ones((K, 1)),
                                open_mask)
        return jnp.sum(st2.f[idx]) + jnp.sum(st2.sig2_feat[idx])

    gst2 = jax.grad(loss_cf)(st0)
    assert float(jnp.max(jnp.abs(gst2.f[12]))) > 0.0
    assert float(jnp.max(jnp.abs(gst2.sig2_feat[12]))) > 0.0


# T13 — JIT == eager
def test_jit_matches_eager():
    st = init(SMALL)
    K = 2
    idx = jnp.array([20, 21], jnp.int32)
    args = (st, 1.0, idx, jnp.ones((K, 3)), jnp.full((K, 3), 0.3),
            jnp.ones((K, SMALL.feat_dim)), jnp.full((K, 1), 0.4),
            jnp.full((K,), 0.6), jnp.array([True, True]))
    eager = write(SMALL, *args)
    jitted = jax.jit(lambda st, t, i, mp, sp, mf, sf, e, om:
                     write(SMALL, st, t, i, mp, sp, mf, sf, e, om))(*args)
    for fld in AnchorState._fields:
        assert float(jnp.max(jnp.abs(getattr(eager, fld) - getattr(jitted, fld)))) < TOL


# T14 — extreme variances / η extremes stay finite (§16)
def test_extreme_variances_and_eta_extremes_stay_finite():
    K = 2
    idx = jnp.array([30, 31], jnp.int32)
    st = init(SMALL)
    st1 = write_closed_form(SMALL, st, 1.0, idx,
                            jnp.full((K, 3), 1e6), jnp.full((K, 3), 1e-12),
                            jnp.full((K, SMALL.feat_dim), 1e6),
                            jnp.full((K, 1), 1e12), jnp.array([True, True]))
    st2 = write(SMALL, st, 1.0, idx, jnp.ones((K, 3)), jnp.ones((K, 3)),
                jnp.ones((K, SMALL.feat_dim)), jnp.ones((K, 1)),
                jnp.array([1.0, 0.0]), jnp.array([True, True]))
    for res in (st1, st2):
        for fld in AnchorState._fields:
            assert bool(jnp.all(jnp.isfinite(getattr(res, fld)))), fld
    # η = 0 (A27 identity): content bit-unchanged (τ/e bookkeeping updates)
    assert _eq(st2.f[31], st.f[31]) and _eq(st2.sig2_feat[31], st.sig2_feat[31])
    assert _eq(st2.c[31], st.c[31])


def run_benchmark(K: int = 4096, iters: int = 20, seed: int = 0):
    """Realistic-shape write microbenchmark (early Phase-O datapoint):
    full learned-gate write path (conflict fusion + blend + scatter) at
    proposal scale — 128,000 anchors × 128-d features, K candidate writes."""
    cfg = AnchorFieldConfig()
    st = init(cfg)
    key = jax.random.PRNGKey(seed)
    idx = jax.random.randint(key, (K,), 0, cfg.capacity).astype(jnp.int32)
    mu_p = jax.random.normal(key, (K, 3))
    s2_p = jnp.full((K, 3), 0.01)
    mu_f = jax.random.normal(jax.random.fold_in(key, 1), (K, cfg.feat_dim))
    s2_f = jnp.full((K, 1), 0.05)
    eta = jnp.full((K,), 0.5)
    om = jnp.ones((K,), jnp.bool_)
    fn = jax.jit(lambda st, t, i, mp, sp, mf, sf, e, m:
                 write(cfg, st, t, i, mp, sp, mf, sf, e, m))
    t0 = time.perf_counter()
    out = fn(st, 0.0, idx, mu_p, s2_p, mu_f, s2_f, eta, om)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(iters):
        out = fn(st, float(i), idx, mu_p, s2_p, mu_f, s2_f, eta, om)
    jax.block_until_ready(out)
    ms = (time.perf_counter() - t0) / iters * 1e3
    state_bytes = sum(int(x.size) * x.dtype.itemsize
                      for x in (st.c, st.f, st.sig2_pos, st.sig2_feat, st.e, st.tau))
    print("Engram3D anchor-field write benchmark")
    print(f"  backend={jax.default_backend()}  devices={jax.device_count()}")
    print(f"  capacity={cfg.capacity}  feat_dim={cfg.feat_dim}  K={K}")
    print(f"  state size={state_bytes / 2**20:.1f} MiB")
    print(f"  compile+first={compile_s:.2f}s   steady-state={ms:.2f} ms/write")
    print(f"  => one write per training step adds {ms:.2f} ms/step memory overhead")


TESTS = [
    ("lattice_geometry", test_lattice_geometry),
    ("index_roundtrip_and_clamping", test_index_roundtrip_and_clamping),
    ("eta_star_is_gaussian_mle", test_eta_star_is_gaussian_mle),
    ("closed_form_single_row_matches_gaussian", test_closed_form_single_row_matches_gaussian),
    ("batch_equals_sequential_and_order_invariant", test_batch_equals_sequential_and_order_invariant),
    ("learned_eta_blends_and_A27_variance", test_learned_eta_blends_and_A27_variance),
    ("eta_one_overwrites_with_floor", test_eta_one_overwrites_with_floor),
    ("gate_closed_leaves_state_bitwise_unchanged", test_gate_closed_leaves_state_bitwise_unchanged),
    ("closed_row_excluded_from_fusion", test_closed_row_excluded_from_fusion),
    ("write_does_not_touch_other_anchors", test_write_does_not_touch_other_anchors),
    ("counterfactual_deletion_resets_to_prior", test_counterfactual_deletion_resets_to_prior),
    ("gradient_reaches_writer_and_memory_not_closed_rows", test_gradient_reaches_writer_and_memory_not_closed_rows),
    ("jit_matches_eager", test_jit_matches_eager),
    ("extreme_variances_and_eta_extremes_stay_finite", test_extreme_variances_and_eta_extremes_stay_finite),
]


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
    print(f"PHASE D1 ANCHOR-FIELD TESTS: {len(TESTS) - len(failed)}/{len(TESTS)} PASS"
          + ("  — DO NOT PROCEED TO TRAINING (prompt §6)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_benchmark()
        sys.exit(0)
    sys.exit(run_all())