# =====================================================================
# Engram3D · Phase D Part 2 · frustum-reader correctness tests
# File: tests/test_reader.py   (pytest-compatible; plain runner)
# v2 CHANGELOG (after first execution, JAX 0.11.0 / TPU v5e-8):
#   FIX-1  make_state: jnp.stack(list-of-tuples) rejected by JAX 0.11
#          (stack now requires ndarray/scalar elements) →
#          jnp.asarray(centers, float32). Blocked 9 tests.
#   FIX-2  test_pose_conventions v1 asserted "R @ fwd == fwd", which
#          only holds for the rotation axis — a bug in the TEST, not the
#          module. Correct semantics: R (camera→world, columns = camera
#          axes in world) maps the camera z-axis onto the world forward:
#          R @ [0,0,1] == fwd  ⇔  Rᵀ @ fwd == [0,0,1].
# Status: UNTESTED (v2) — run and return output. GATES all later
# training (prompt §6). All expectations analytic.
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
        AnchorFieldConfig, AnchorState, init as anchor_init)
    from engram3d.model.memory.reader import (
        FrustumConfig, pose_from_yaw_pitch, payload_dim,
        init_embed_params, read, embed_tokens, apply_read_dropout)
except Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anchor_field import (AnchorFieldConfig, AnchorState,
                              init as anchor_init)
    from reader import (FrustumConfig, pose_from_yaw_pitch, payload_dim,
                        init_embed_params, read, embed_tokens,
                        apply_read_dropout)

CFG_A = AnchorFieldConfig(feat_dim=8, envelope=(3.0, 3.0, 3.0))   # 8000 cells
CFG_F = FrustumConfig(feat_dim=8, n_tokens=256, fov_h_deg=90.0,
                      fov_v_deg=90.0, near=0.1, far=5.0,
                      token_dim=32, hidden_dim=16)
TOL = 1e-5
Z3 = jnp.zeros(3)


def make_state(centers, exist, feat_seed=0):
    n = CFG_A.capacity
    m = len(exist)
    key = jax.random.PRNGKey(feat_seed)
    st = AnchorState(
        c=jnp.zeros((n, 3)), f=jnp.zeros((n, CFG_A.feat_dim)),
        sig2_pos=jnp.ones((n, 3)), sig2_feat=jnp.ones((n, CFG_A.feat_dim)),
        e=jnp.full((n,), -1.0), tau=jnp.full((n,), -1.0))
    return st._replace(
        c=st.c.at[:m].set(jnp.asarray(centers, dtype=jnp.float32)),
        f=st.f.at[:m].set(jax.random.normal(key, (m, CFG_A.feat_dim))),
        e=st.e.at[:m].set(jnp.asarray(exist, dtype=jnp.float32)),
        tau=st.tau.at[:m].set(0.0))


def _read(st, yaw=0.0, pitch=0.0, pos=None, t=100.0):
    pos = Z3 if pos is None else pos
    return read(CFG_F, CFG_A, st, pos, pose_from_yaw_pitch(pos, yaw, pitch), t)


# R1 — pose conventions: orthonormal, det=+1, camera→world / world→camera
def test_pose_conventions():
    cam_z = jnp.array([0.0, 0.0, 1.0], jnp.float32)
    cases = [(0.0, 0.0, (0.0, 0.0, 1.0)),
             (jnp.pi / 2, 0.0, (1.0, 0.0, 0.0)),
             (0.0, jnp.pi / 2, (0.0, 1.0, 0.0))]
    for yaw, pitch, fwd in cases:
        R = pose_from_yaw_pitch(Z3, yaw, pitch)
        fwd = jnp.asarray(fwd, jnp.float32)
        # camera→world: R maps the camera forward axis onto world forward
        assert float(jnp.max(jnp.abs(R @ cam_z - fwd))) < TOL
        # world→camera: Rᵀ maps world forward back onto the camera z-axis
        assert float(jnp.max(jnp.abs(R.T @ fwd - cam_z))) < TOL
        assert float(jnp.max(jnp.abs(R @ R.T - jnp.eye(3)))) < TOL
        assert abs(float(jnp.linalg.det(R)) - 1.0) < TOL
    R0 = pose_from_yaw_pitch(Z3, 0.0, 0.0)
    assert float(jnp.max(jnp.abs(R0[:, 0] - jnp.array([1., 0., 0.])))) < TOL  # right
    assert float(jnp.max(jnp.abs(R0[:, 1] - jnp.array([0., 1., 0.])))) < TOL  # up
    assert float(jnp.max(jnp.abs(R0[:, 2] - jnp.array([0., 0., 1.])))) < TOL  # forward


# R2 — frustum containment at origin, yaw 0, fov 90°, near .1, far 5
def test_frustum_containment():
    pts = [(0, 0, 2), (2.1, 0, 2), (1.9, 0, 2), (0, 2.1, 2),
           (0, 1.9, 2), (0, 0, 0.05), (0, 0, 5.0), (0, 0, 5.1)]
    st = make_state(pts, [5.0] * len(pts))
    rd = _read(st)
    assert sorted(jnp.asarray(rd.idx[rd.valid]).tolist()) == [0, 2, 4, 6]


# R3 — yaw rotates the frustum
def test_yaw_rotates_selection():
    st = make_state([(2, 0, 0), (0, 0, 2)], [5.0, 5.0])
    rd = _read(st, yaw=jnp.pi / 2)
    assert jnp.asarray(rd.idx[rd.valid]).tolist() == [0]


# R4 — pitch: looking straight up sees the zenith anchor
def test_pitch_looks_up():
    st = make_state([(0, 3, 0)], [5.0])
    rd = _read(st, pitch=jnp.pi / 2)
    assert bool(rd.valid[0]) and int(rd.valid.sum()) == 1


# R5 — top-256 ranking by e·(1/(1+d)); equal distances isolate e-ranking
def test_top256_ranking():
    import math
    m = 300
    phi = [2 * math.pi * i / m for i in range(m)]
    centers = [(1.2 * math.cos(p), 1.2 * math.sin(p), 1.5) for p in phi]
    exist = [1.0 + 0.05 * i for i in range(m)]
    st = make_state(centers, exist)
    rd = _read(st)
    assert int(rd.valid.sum()) == 256
    assert sorted(jnp.asarray(rd.idx[rd.valid]).tolist()) == list(range(44, 300))
    assert bool(jnp.all(jnp.diff(rd.score) <= 0))


# R6 — null-token padding; zero-init ⇒ tokens ≡ 0 (ADR-005)
def test_null_padding_and_zero_init():
    pts = [((j % 5 - 2) * 0.4, ((j // 5) - 0.5) * 0.8, 2.0) for j in range(10)]
    st = make_state(pts, [5.0] * 10)
    rd = _read(st)
    assert int(rd.valid.sum()) == 10
    params = init_embed_params(jax.random.PRNGKey(0), CFG_F)
    tok = embed_tokens(params, CFG_F, rd.payload, rd.valid)
    assert float(jnp.max(jnp.abs(tok))) == 0.0                 # zero-conv init
    p2 = dict(params); p2["null_embed"] = jnp.ones((CFG_F.token_dim,), jnp.float32)
    p2["w2"] = params["w2"] + 0.01
    tok2 = embed_tokens(p2, CFG_F, rd.payload, rd.valid)
    assert bool(jnp.all(tok2[~rd.valid] == 1.0))               # null token used
    assert float(jnp.max(jnp.abs(tok2[rd.valid]))) > 0.0


# R7 — gradient path: render-head → read → memory/pose/params
def test_gradient_flow():
    import math
    m = 300
    phi = [2 * math.pi * i / m for i in range(m)]
    centers = [(1.2 * math.cos(p), 1.2 * math.sin(p), 1.5) for p in phi]
    st = make_state(centers, [1.0 + 0.05 * i for i in range(m)])
    params = init_embed_params(jax.random.PRNGKey(1), CFG_F)

    def loss_prm(prm):
        rd = _read(st)
        return jnp.sum(embed_tokens(prm, CFG_F, rd.payload, rd.valid))

    g0 = jax.grad(loss_prm)(params)                # pristine: w2 zero-init
    assert float(jnp.max(jnp.abs(g0["w2"]))) > 0.0  # zero-conv: w2 learns at step 0

    p2 = dict(params); p2["w2"] = params["w2"] + 0.01   # open the value path

    def loss_st(s):
        rd = _read(s)
        return jnp.sum(embed_tokens(p2, CFG_F, rd.payload, rd.valid))

    gs = jax.grad(loss_st)(st)
    assert float(jnp.max(jnp.abs(gs.f[44:300]))) > 0.0        # selected anchors
    assert float(jnp.max(jnp.abs(gs.f[:44]))) == 0.0          # unselected actives
    assert float(jnp.max(jnp.abs(gs.f[300:]))) == 0.0         # inactive fillers

    def loss_pos(pos):
        rd = read(CFG_F, CFG_A, st, pos, pose_from_yaw_pitch(pos, 0.0, 0.0), 100.0)
        return jnp.sum(embed_tokens(p2, CFG_F, rd.payload, rd.valid))

    assert float(jnp.max(jnp.abs(jax.grad(loss_pos)(Z3)))) > 0.0

    def loss_yaw(yaw):
        rd = read(CFG_F, CFG_A, st, Z3, pose_from_yaw_pitch(Z3, yaw, 0.0), 100.0)
        return jnp.sum(embed_tokens(p2, CFG_F, rd.payload, rd.valid))

    assert float(jnp.max(jnp.abs(jax.grad(loss_yaw)(0.0)))) > 0.0


# R8 — anti-bypass frustum dropout (§4.4, p=0.15)
def test_read_dropout():
    import math
    m = 300
    phi = [2 * math.pi * i / m for i in range(m)]
    st = make_state([(1.2 * math.cos(p), 1.2 * math.sin(p), 1.5) for p in phi],
                    [1.0 + 0.05 * i for i in range(m)])
    params = init_embed_params(jax.random.PRNGKey(2), CFG_F)
    p2 = dict(params); p2["w2"] = params["w2"] + 0.01
    rd = _read(st)
    tok = embed_tokens(p2, CFG_F, rd.payload, rd.valid)
    rng = jax.random.PRNGKey(3)
    out, keep = apply_read_dropout(rng, tok, rd.valid, p=0.15, mode="token")
    n_drop = int((~keep).sum())
    assert 15 <= n_drop <= 62                                   # ~Binomial(256,.15)
    assert bool(jnp.all(out[~keep] == 0.0))
    assert bool(jnp.allclose(out[keep], tok[keep]))
    out2, keep2 = apply_read_dropout(rng, tok, rd.valid, p=0.15, mode="token")
    assert bool(jnp.array_equal(keep, keep2))                   # deterministic
    outr, keepr = apply_read_dropout(rng, tok, rd.valid, p=0.15, mode="read")
    assert bool(jnp.all((outr == 0.0) | keepr[:, None]))        # self-consistent


# R9 — JIT == eager
def test_jit_matches_eager():
    import math
    m = 300
    phi = [2 * math.pi * i / m for i in range(m)]
    st = make_state([(1.2 * math.cos(p), 1.2 * math.sin(p), 1.5) for p in phi],
                    [1.0 + 0.05 * i for i in range(m)])
    params = init_embed_params(jax.random.PRNGKey(4), CFG_F)

    def full(s, pos, R, t, prm):
        rd = read(CFG_F, CFG_A, s, pos, R, t)
        return rd.payload, embed_tokens(prm, CFG_F, rd.payload, rd.valid)

    pos, R, t = Z3, pose_from_yaw_pitch(Z3, 0.0, 0.0), 100.0
    ea = full(st, pos, R, t, params)
    ji = jax.jit(full)(st, pos, R, t, params)
    assert float(jnp.max(jnp.abs(ea[0] - ji[0]))) < 1e-6
    assert float(jnp.max(jnp.abs(ea[1] - ji[1]))) < 1e-6


# R10 — contamination guard: inactive / out-of-frustum never selected
def test_no_contamination():
    st = make_state([(10, 0, 0), (0, 0, 2), (0, 0, 2.2), (0, 0, 2.4),
                     (0, 0, 2.6), (0, 0, 2.8), (0, 0, 3.0)],
                    [100.0, -5.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    rd = _read(st)
    sel = set(jnp.asarray(rd.idx[rd.valid]).tolist())
    assert sel == {2, 3, 4, 5, 6}          # idx0 out-of-frustum, idx1 inactive


def run_benchmark(iters: int = 20):
    """Full-scale read+embed microbenchmark (Phase-O datapoint):
    128,000 anchors × 128-d features, 5,000 active, 256 tokens."""
    cfg_a = AnchorFieldConfig()
    cfg_f = FrustumConfig()
    st = anchor_init(cfg_a)
    key = jax.random.PRNGKey(0)
    act_idx = jax.random.randint(key, (5000,), 0, cfg_a.capacity)
    st = st._replace(e=st.e.at[act_idx].set(3.0), tau=st.tau.at[act_idx].set(0.0))
    pos = jnp.array([0.0, 1.5, 0.0])
    R = pose_from_yaw_pitch(pos, 0.0, 0.0)
    params = init_embed_params(jax.random.PRNGKey(1), cfg_f)

    def full(s, p, Rr, t, prm):
        rd = read(cfg_f, cfg_a, s, p, Rr, t)
        return embed_tokens(prm, cfg_f, rd.payload, rd.valid), rd.valid

    fn = jax.jit(full)
    t0 = time.perf_counter()
    tok, valid = fn(st, pos, R, 100.0, params)
    jax.block_until_ready(tok)
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(iters):
        tok, valid = fn(st, pos, R, float(i), params)
    jax.block_until_ready(tok)
    ms = (time.perf_counter() - t0) / iters * 1e3
    print("Engram3D frustum-read benchmark")
    print(f"  backend={jax.default_backend()}  devices={jax.device_count()}")
    print(f"  capacity={cfg_a.capacity}  active=5000  n_tokens={cfg_f.n_tokens}")
    print(f"  compile+first={compile_s:.2f}s   steady-state={ms:.2f} ms/read")
    print(f"  n_valid at test pose={int(valid.sum())}  tokens={tuple(tok.shape)}")
    print(f"  => combined memory-op budget/step (write+read, K=4096): "
          f"{3.54 + ms:.2f} ms (write measured previously)")


TESTS = [("pose_conventions", test_pose_conventions),
         ("frustum_containment", test_frustum_containment),
         ("yaw_rotates_selection", test_yaw_rotates_selection),
         ("pitch_looks_up", test_pitch_looks_up),
         ("top256_ranking", test_top256_ranking),
         ("null_padding_and_zero_init", test_null_padding_and_zero_init),
         ("gradient_flow", test_gradient_flow),
         ("read_dropout", test_read_dropout),
         ("jit_matches_eager", test_jit_matches_eager),
         ("no_contamination", test_no_contamination)]


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
    print(f"PHASE D2 READER TESTS: {len(TESTS) - len(failed)}/{len(TESTS)} PASS"
          + ("  — DO NOT PROCEED (prompt §6)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_benchmark()
        sys.exit(0)
    sys.exit(run_all())