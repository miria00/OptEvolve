"""Portfolio stage of the generalisation protocol (domain and split in results/generalisation_20260911/
domain.json, declared before tuning):
  cross     time every candidate (the distinct best configurations of the guided loop on tune instances,
            plus plain FISTA) on every tune instance; sharded over GPUs with --shard i/N;
  build     fit the portfolio and dispatch rule on the tune matrix only (atlas.evolve.portfolio) and freeze
            it in portfolio.json with a content hash;
  evaluate  run the frozen dispatch on every held-out instance (sharded), plus the single best recipe and
            plain FISTA for reference, and compare with cuML (fastest and median of its runs).
Times are medians of --reps solves after an untimed warm-up solve; step size timed; certificate is the
full-problem relative duality gap."""
import argparse, glob, hashlib, json, math, os, statistics, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=("candidates", "cross", "build", "evaluate"))
ap.add_argument("--dir", default=os.path.join(ROOT, "results/generalisation_20260911"))
ap.add_argument("--shard", default="0/1"); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--max-evals", type=int, default=20000)
ap.add_argument("--max-routes", type=int, default=3)
a = ap.parse_args()
dom = json.load(open(os.path.join(a.dir, "domain.json")))
PLAIN = {"K": 64, "screen": False, "step_rule": "power"}


def instances(kind):
    blocks = [dict(dom["split"]["tune"], kind="tune")] if kind == "tune" else [dict(h) for h in dom["split"]["held_out"]]
    return [{"dataset": d, "lam_frac": lf, "alpha": al, "id": f"{d}_a{al}_r{lf}"}
            for bl in blocks for d in bl["datasets"] for lf in bl["lam_fracs"] for al in bl["alphas"]]


def key(cfg):
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)


def candidates():
    cs = {key(PLAIN): PLAIN}
    for f in sorted(glob.glob(os.path.join(a.dir, "tune", "*_loop.json"))):
        chain = json.load(open(f))["chain"]
        cfg = chain[-1]["config"]; cs[key(cfg)] = cfg
    return {f"c{j}": cfg for j, (_, cfg) in enumerate(sorted(cs.items()))}


def cuml_times(iid):
    ts = [json.load(open(f))["cuml"] for f in glob.glob(os.path.join(a.dir, "baselines", f"{iid}_cuml_rep*.json"))]
    ts = [t["seconds"] for t in ts if t.get("passed")]
    return (min(ts), statistics.median(ts), len(ts)) if ts else (None, None, 0)


def features(inst, shape):
    return {"log10_n": math.log10(shape[0]), "log10_p": math.log10(shape[1]), "lam_frac": inst["lam_frac"], "alpha": inst["alpha"]}


_cache = {}


def load(ds):
    import numpy as np
    if ds not in _cache:
        _cache.clear()
        d = os.path.join(ROOT, "data/enet_libsvm", ds)
        _cache[ds] = (np.load(d + "/X.npy").astype(np.float64), np.load(d + "/y.npy").astype(np.float64))
    return _cache[ds]


def time_config(inst, cfg):
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from atlas.bench import logreg_cell as lc
    from atlas.evolve.screened_runner import run_screened
    X, y = load(inst["dataset"])
    lam = inst["lam_frac"] * lc.lam_max(X, y, inst["alpha"])
    kw = {k: ({"float32": jnp.float32, "float64": jnp.float64}[v] if k == "dtype" else v) for k, v in cfg.items()}
    try:
        run_screened(X, y, lam, inst["alpha"], tol=a.tol, max_evals=a.max_evals, **kw)
        rs = [run_screened(X, y, lam, inst["alpha"], tol=a.tol, max_evals=a.max_evals, **kw) for _ in range(a.reps)]
    except Exception as e:
        jax.clear_caches(); return {"seconds": float("inf"), "converged": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    jax.clear_caches()
    ok = all(r["converged"] for r in rs)
    inf = float("inf")
    # Keep EVERY repetition, not just their median: cuML's reps are stored as separate files, so
    # discarding ours here is the only reason a paired comparison is impossible. Record the compile
    # cost too: `seconds` is the solve loop (right for a repeated-solve workload such as a lambda path),
    # `seconds_incl_compile` is the cold single-solve number. State which one any claim uses.
    return {"seconds": statistics.median(r["wall_s"] for r in rs) if ok else inf, "converged": ok,
            "evals": rs[0]["evals"], "gap": rs[0]["gap"], "shape": list(X.shape),
            "walls": [float(r["wall_s"]) for r in rs],
            "walls_incl_compile": [float(r["wall_incl_compile_s"]) for r in rs],
            "compile_s": float(rs[0]["compile_s"]),
            "seconds_incl_compile": statistics.median(r["wall_incl_compile_s"] for r in rs) if ok else inf,
            "timed_span": "solve loop only; seconds_incl_compile adds compilation"}


i, N = map(int, a.shard.split("/"))
if a.stage == "candidates":
    for cid, cfg in candidates().items():
        print(cid, cfg)
elif a.stage == "cross":
    import jax
    assert jax.devices()[0].platform == "gpu" or os.environ.get("ATLAS_ALLOW_CPU") == "1", "not on a GPU"
    os.makedirs(os.path.join(a.dir, "cross"), exist_ok=True)
    C = candidates()
    for inst in sorted(instances("tune"), key=lambda r: r["dataset"])[i::N]:
        out = os.path.join(a.dir, "cross", inst["id"] + ".json")
        rec = json.load(open(out)) if os.path.exists(out) else {"instance": inst, "times": {}}
        for cid, cfg in C.items():
            if key(cfg) in rec["times"]:
                continue
            rec["times"][key(cfg)] = time_config(inst, cfg)
            json.dump(rec, open(out, "w"), indent=1)
            print(inst["id"], cid, rec["times"][key(cfg)].get("seconds"), flush=True)
elif a.stage == "build":
    from atlas.evolve.portfolio import fit_portfolio, geo_mean
    assert not os.path.exists(os.path.join(a.dir, "portfolio.json")), "portfolio already frozen; refusing to refit"
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "cross", "*.json")))]
    held = {r["id"] for r in instances("held_out")}
    assert not any(r["instance"]["id"] in held for r in recs), "held-out instance found in the tune matrix"
    T = {r["instance"]["id"]: {k: v["seconds"] for k, v in r["times"].items()} for r in recs}
    F = {r["instance"]["id"]: features(r["instance"], next(v["shape"] for v in r["times"].values() if "shape" in v)) for r in recs}
    P = fit_portfolio(T, F, max_routes=a.max_routes)
    P.update(frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tune_instances=sorted(T),
             candidates={k: json.loads(k) for k in sorted({c for t in T.values() for c in t})})
    body = json.dumps(P, sort_keys=True, default=str); P["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(P, open(os.path.join(a.dir, "portfolio.json"), "w"), indent=1, default=str)
    print(json.dumps({k: P[k] for k in ("members", "rule", "tune_geo_single", "tune_geo_dispatch", "tune_geo_oracle", "sha256")}, indent=1))
else:
    import jax
    assert jax.devices()[0].platform == "gpu" or os.environ.get("ATLAS_ALLOW_CPU") == "1", "not on a GPU"
    from atlas.evolve.portfolio import dispatch
    P = json.load(open(os.path.join(a.dir, "portfolio.json")))
    print("portfolio", P["sha256"][:16], "frozen", P["frozen_utc"], flush=True)
    os.makedirs(os.path.join(a.dir, "heldout"), exist_ok=True)
    for inst in sorted(instances("held_out"), key=lambda r: r["dataset"])[i::N]:
        out = os.path.join(a.dir, "heldout", inst["id"] + ".json")
        if os.path.exists(out):
            continue
        # A dataset can be absent from disk (covtype and gisette were removed after the 2026-09-11 run and
        # are not in the data manifest). load() raised FileNotFoundError here, OUTSIDE time_config's guard,
        # so the whole stage died on the first such instance and measured nothing at all. Skip and say so:
        # a held-out set that can only be partly measured must report which part, not crash.
        if not os.path.exists(os.path.join(ROOT, "data/enet_libsvm", inst["dataset"], "X.npy")):
            print(inst["id"], "SKIPPED: dataset not on disk", flush=True)
            continue
        X, _ = load(inst["dataset"])
        feat = features(inst, X.shape)
        routed = dispatch(P["rule"], feat)
        rec = {"instance": inst, "features": feat, "portfolio_sha256": P["sha256"], "routed": routed, "times": {}}
        for label, k in (("dispatched", routed), ("single_best", P["single_best"]), ("plain_fista", key(PLAIN))):
            if k not in [v.get("key") for v in rec["times"].values()]:
                rec["times"][label] = {"key": k, **time_config(inst, json.loads(k))}
            else:
                rec["times"][label] = {"same_as": next(l for l, v in rec["times"].items() if v.get("key") == k)}
        cmin, cmed, nrun = cuml_times(inst["id"])
        rec["cuml"] = {"min": cmin, "median": cmed, "runs": nrun}
        json.dump(rec, open(out, "w"), indent=1, default=str)
        print(inst["id"], "routed", routed, rec["times"]["dispatched"].get("seconds"), "cuML min", cmin, flush=True)
print("PORTFOLIO_STAGE_DONE", a.stage, a.shard)
