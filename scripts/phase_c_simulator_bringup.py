# =====================================================================
# Engram3D · Phase C · AI2-THOR / ProcTHOR bring-up on Kaggle
# File: scripts/phase_c_simulator_bringup.py
# Status: UNTESTED COMPLETE IMPLEMENTATION.
# Validates:
#   C1 ai2thor install + exact version recording (reproducibility §17)
#   C2 Xvfb + Mesa llvmpipe headless CPU rendering
#   C3 Controller rollout: RGB, GT depth (§8 mitigation), instance
#      segmentation (OPS / Amodal IoU inputs, §5)
#   C4 scripted discrete action interface (world-model action input)
#   C5 seed-replay determinism probe (§5 "deterministic seeds")
#   C6 single-worker step throughput (data-generation planning)
# Output: phase_c_report.json. Expect API drift — return tracebacks.
# =====================================================================
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from importlib import metadata as ilm

REPORT = {"schema": "engram3d.phase-c/1",
          "generated_utc": datetime.now(timezone.utc).isoformat()}

def save():
    path = "/kaggle/working/phase_c_report.json" if os.path.isdir("/kaggle/working") \
        else "phase_c_report.json"
    with open(path, "w") as fh:
        json.dump(REPORT, fh, indent=2, default=str)
    print(f"[report] saved -> {path}")

def sh(cmd, timeout=900):
    entry = {"cmd": " ".join(map(str, cmd))}
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True,
                           text=True, timeout=timeout)
        entry.update(returncode=r.returncode,
                     stdout_tail=(r.stdout or "")[-3000:],
                     stderr_tail=(r.stderr or "")[-3000:])
    except Exception as exc:
        entry.update(returncode=-1, error=f"{type(exc).__name__}: {exc}")
    REPORT.setdefault("commands", []).append(entry)
    print(f"[sh rc={entry.get('returncode')}] {entry['cmd']}")
    return entry

# C1 — packages -------------------------------------------------------
sh([sys.executable, "-m", "pip", "install", "ai2thor"], timeout=1800)
sh([sys.executable, "-m", "pip", "install", "procthor"], timeout=1800)  # optional
REPORT["versions"] = {}
for pkg in ("ai2thor", "procthor"):
    try:
        REPORT["versions"][pkg] = ilm.version(pkg)
    except Exception:
        REPORT["versions"][pkg] = None
print(f"[versions] {REPORT['versions']}")

# C2 — headless rendering stack ---------------------------------------
sh(["apt-get", "update", "-qq"], timeout=600)
core = ["xvfb", "libgl1", "libglx-mesa0", "mesa-utils"]
extras = ["libegl1", "libxrandr2", "libxi6", "libxcursor1", "libxinerama1"]
if sh(["apt-get", "install", "-y", "-qq"] + core + extras).get("returncode") != 0:
    sh(["apt-get", "install", "-y", "-qq"] + core)
sh(["pkill", "-f", "Xvfb :99"], timeout=60)          # tolerant cleanup
os.environ["DISPLAY"] = ":99"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"            # force llvmpipe
xvfb = subprocess.Popen(
    ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3.0)
glx = sh(["glxinfo", "-B"], timeout=120)
renderer = ""
for line in (glx.get("stdout_tail") or "").splitlines():
    if "renderer" in line.lower():
        renderer = line.strip()
        break
REPORT["gl_renderer"] = renderer
print(f"[gl] {renderer}")

ACTIONS = ["RotateRight", "RotateRight", "MoveAhead", "MoveAhead",
           "LookDown", "MoveAhead", "RotateLeft", "MoveBack"]

def _sha(arr):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def _pose(event):
    p = event.metadata["agent"]["position"]
    return [round(float(p["x"]), 4), round(float(p["y"]), 4), round(float(p["z"]), 4)]

def rollout(seed):
    """One scripted rollout in a fresh Controller (fresh Unity process)."""
    from ai2thor.controller import Controller
    t0 = time.time()
    controller = Controller(
        platform="Linux64", width=256, height=256, seed=seed,
        renderDepth=True, renderInstanceSegmentation=True,
        scene="FloorPlan1_Kitchen")
    try:
        ev = controller.reset()
    except Exception:
        ev = controller.last_event
    rec = {"seed": seed, "init_seconds": round(time.time() - t0, 1),
           "hashes": [_sha(ev.frame)], "poses": [_pose(ev)],
           "step_ms": [], "frame_shape": list(ev.frame.shape),
           "depth_ok": getattr(ev, "depth", None) is not None,
           "instance_seg_ok": getattr(ev, "instance_segmentation", None) is not None}
    for act in ACTIONS:
        t1 = time.time()
        ev = controller.step(action=act)
        rec["step_ms"].append(round((time.time() - t1) * 1000.0, 1))
        rec["hashes"].append(_sha(ev.frame))
        rec["poses"].append(_pose(ev))
    rec["mean_step_ms"] = round(sum(rec["step_ms"]) / len(rec["step_ms"]), 1)
    rec["steps_per_second"] = round(1000.0 / rec["mean_step_ms"], 2)
    try:
        controller.stop()
    except Exception:
        pass
    return rec

try:
    r1, r2, r3 = rollout(seed=7), rollout(seed=7), rollout(seed=8)
    REPORT["rollouts"] = [r1, r2, r3]
    REPORT["determinism"] = {
        "same_seed_identical_frames": r1["hashes"] == r2["hashes"],
        "different_seed_differs": r1["hashes"] != r3["hashes"],
        "note": "Navigation-only rollouts are expected deterministic; THOR "
                "physics RNG can break equality. If False we harden this in "
                "Phase C.2 — required by §5 'deterministic seeds'."}
    print(f"[determinism] {REPORT['determinism']}")
    print(f"[throughput] mean step {r1['mean_step_ms']} ms "
          f"({r1['steps_per_second']} steps/s, single worker)")
except Exception:
    REPORT["rollouts"] = {"status": "error",
                          "trace": traceback.format_exc()[-6000:]}
    print("[ERROR] rollout failed — see report")

# ProcTHOR probe (API drift expected; full integration is Phase C.2) ---
try:
    from procthor.generate import House
    try:
        house = House.from_seed(7)
        REPORT["procthor"] = {"status": "ok", "api": "House.from_seed",
                              "type": type(house).__name__}
    except Exception as exc:
        REPORT["procthor"] = {"status": "api-drift",
                              "error": f"{type(exc).__name__}: {exc}",
                              "trace": traceback.format_exc()[-2000:]}
except Exception as exc:
    REPORT["procthor"] = {"status": "not-importable",
                          "error": f"{type(exc).__name__}: {exc}"}
print(f"[procthor] {REPORT['procthor']['status']}")

sh(["du", "-sh", os.path.expanduser("~/.ai2thor")], timeout=60)

rolls = REPORT.get("rollouts")
roll_ok = isinstance(rolls, list) and len(rolls) == 3
det = REPORT.get("determinism", {}) if roll_ok else {}
REPORT["checks"] = {
    "ai2thor_installed": REPORT["versions"].get("ai2thor") is not None,
    "xvfb_running": xvfb.poll() is None,
    "llvmpipe_renderer": "llvmpipe" in renderer.lower(),
    "rollout_ok": roll_ok,
    "depth_rendered": bool(roll_ok and rolls[0].get("depth_ok")),
    "instance_seg_rendered": bool(roll_ok and rolls[0].get("instance_seg_ok")),
    "same_seed_deterministic": bool(det.get("same_seed_identical_frames")),
    "different_seed_differs": bool(det.get("different_seed_differs")),
}
print("=" * 60)
for k, v in REPORT["checks"].items():
    print(f"  [{'PASS' if v else 'FAIL/WARN'}] {k}")
save()
try:
    xvfb.terminate()
except Exception:
    pass