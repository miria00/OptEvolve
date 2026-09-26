"""Generalisation stage (CAKE Section 6 transposed to solvers): the domain and its tune / held-out split are
declared in results/generalisation_20260911/domain.json before any tuning. Stages:
  baselines  cuML alone on every instance of a split, --reps separate runs each (resumable);
  tune       the guided diagnose loop on every tune instance, after measuring each dataset's bandwidth
             (the diagnosis depends on shape and device).
Instances are sharded with --shard i/N so several GPUs can share a stage; each process uses the one GPU
its CUDA_VISIBLE_DEVICES names. The portfolio and its dispatch rule are built from tune results only and
frozen before any held-out instance is run."""
import argparse, glob, json, os, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=("list", "baselines", "tune"))
ap.add_argument("--domain", default=os.path.join(ROOT, "results/generalisation_20260911/domain.json"))
ap.add_argument("--split", default="tune", choices=("tune", "held_out", "all"))
ap.add_argument("--shard", default="0/1"); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--peak-gbps", type=float, default=1792.0); ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--loop-reps", type=int, default=3)
ap.add_argument("--allow-failing-baseline", action="store_true",
                help="tune: when cuML never passes, tune against its closest run (flagged) instead of skipping")
ap.add_argument("--precision", default="float32", choices=("float32", "float64"), help="baselines: cuML fit precision")
a = ap.parse_args()
dom = json.load(open(a.domain))
OUT = os.path.join(os.path.dirname(a.domain))


def instances(split):
    blocks = [dict(dom["split"]["tune"], kind="tune")] if split in ("tune", "all") else []
    if split in ("held_out", "all"):
        blocks += [dict(h) for h in dom["split"]["held_out"]]
    rows = []
    for bl in blocks:
        for d in bl["datasets"]:
            for lf in bl["lam_fracs"]:
                for al in bl["alphas"]:
                    rows.append({"dataset": d, "lam_frac": lf, "alpha": al,
                                 "split": "tune" if bl.get("kind") == "tune" else "held_out",
                                 "id": f"{d}_a{al}_r{lf}"})
    return rows


def run(cmd, env_extra, log):
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n"); fh.flush()
        return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env_extra}, stdout=fh, stderr=subprocess.STDOUT).returncode


i, N = map(int, a.shard.split("/"))
todo = instances(a.split)[i::N]
if a.stage == "list":
    for r in todo:
        print(r)
    sys.exit(0)
os.makedirs(os.path.join(OUT, "baselines"), exist_ok=True); os.makedirs(os.path.join(OUT, "tune"), exist_ok=True)
for r in todo:
    data = os.path.join("data/enet_libsvm", r["dataset"])
    if a.stage == "baselines":
        for k in range(1, a.reps + 1):
            tag = "cuml" if a.precision == "float32" else "cuml64"
            out = os.path.join(OUT, "baselines", f"{r['id']}_{tag}_rep{k}.json")
            if os.path.exists(out):
                continue
            rc = run([PY, "scripts/baselines/logreg_headtohead.py", "--data", data, "--alpha", str(r["alpha"]),
                      "--lam-ratio", str(r["lam_frac"]), "--ours", "", "--baselines", "cuml", "--tol-start", "1e-4",
                      "--baseline-precision", a.precision,
                      "--out", out], {"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "2"}, os.path.join(OUT, "baselines", "log.txt"))
            print(r["id"], "rep", k, "rc", rc, flush=True)
    else:
        reps = sorted(glob.glob(os.path.join(OUT, "baselines", f"{r['id']}_cuml_rep*.json")))
        ok = [(json.load(open(f))["cuml"], f) for f in reps]
        ok = [(c, f) for c, f in ok if c.get("passed")]
        loop_extra = []
        if ok:
            fastest = min(ok, key=lambda t: t[0]["seconds"])[1]
        elif a.allow_failing_baseline and reps:
            allc = [(json.load(open(f))["cuml"], f) for f in reps]
            fastest = min(allc, key=lambda t: t[0].get("gap", float("inf")))[1]
            loop_extra = ["--allow-failing-baseline"]
            print(r["id"], "cuML never reached the target; tuning against its closest run", os.path.basename(fastest), flush=True)
        else:
            print(r["id"], "no passing cuML baseline; skipped", flush=True); continue
        bwj = os.path.join(OUT, "tune", f"bandwidth_{r['dataset']}.json")
        if not os.path.exists(bwj):
            run([PY, "scripts/diagnostics/logreg_bandwidth.py", "--data", data, "--out", bwj],
                {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, os.path.join(OUT, "tune", "log.txt"))
        out = os.path.join(OUT, "tune", f"{r['id']}_loop.json")
        if os.path.exists(out):
            continue
        rc = run([PY, "scripts/discovery/diagnose_loop_logreg.py", "--data", data, "--alpha", str(r["alpha"]),
                  "--lam-frac", str(r["lam_frac"]), "--baseline-json", fastest, "--bandwidth-json", bwj,
                  "--peak-gbps", str(a.peak_gbps), "--rounds", str(a.rounds), "--reps", str(a.loop_reps), "--out", out] + loop_extra,
                 {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, os.path.join(OUT, "tune", f"{r['id']}_loop.log"))
        print(r["id"], "loop rc", rc, flush=True)
print("STAGE_DONE", a.stage, a.split, a.shard)
