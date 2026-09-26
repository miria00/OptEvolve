"""STEP 3 analysis - can deployable features predict N for candidate formulations, and does the resulting cost model
choose formulations well on sizes and families it was not trained on?

Target: log N (censored runs use log of iterations run, a lower bound, and are flagged).
Decision: per instance choose the candidate minimizing T_hat = setup + N_hat * t_iter, where t_iter is the calibrated
per-iteration cost of that candidate (the t factor is measured, never predicted). Oracle uses measured N; a censored
candidate costs infinity. Report regret T(chosen)/T(oracle) and catastrophic picks (chosen never converged).

Feature sets are nested so the contribution of each kind of information is visible:
  norm      : log ||L||                                   (the proxy the earlier rule used)
  static    : spectrum + shape of the candidate operator, inexactness of its projection, no solve at all
  geometry  : static + probe estimates of the free face, curvature on it, coupling restricted to it
  decay     : empirical residual decay over the probe only (pure extrapolation baseline)
  all       : geometry + decay
"""
import argparse, glob, json, math, os, sys
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = []
for pat in a.files:
    for f in glob.glob(pat):
        for line in open(f):
            r = json.loads(line)
            if "error" in r or r.get("features") is None:
                continue
            rows.append(r)
print(f"{len(rows)} usable (instance, candidate) rows from {len(set(r['family']+str(r['size'])+str(r['seed'])+str(r['params']) for r in rows))} instances")


def inst_key(r):
    return (r["family"], r["size"], r["seed"], json.dumps(r["params"], sort_keys=True))


def dyk_k(cand):
    return int(cand.split("dykstra")[1]) if "dykstra" in cand else 0


BASE = {
    "norm": ["log_normL"],
    "static": ["log_normL", "log_gap_L", "rowmass_L", "rank_ratio", "nnz_ratio", "absorbed_frac", "muL_P", "log_n",
               "inexact"],
}
BASE["geometry"] = BASE["static"] + ["free_frac_T", "curv_free_min", "curv_free_mean", "log_coupling_free",
                                     "log_free_dim"]
BASE["decay"] = ["decay_rate", "inexact"]
BASE["all"] = BASE["geometry"] + ["decay_rate"]


def X_of(rs, cols):
    out = []
    for r in rs:
        f = dict(r["features"])
        k = dyk_k(r["cand"])
        f["inexact"] = 0.0 if k == 0 else 1.0 / math.log(1.0 + k)       # larger = fewer inner steps = less exact
        out.append([float(f.get(c, 0.0)) for c in cols])
    return np.nan_to_num(np.array(out), nan=0.0, posinf=50.0, neginf=-50.0)


def y_of(rs):
    return np.array([math.log(r["N"] if r["N"] else r["iters_run"]) for r in rs])


def total_time(r, N):
    return r["setup_s"] + N * (r["t_iter"] or 0.0)


def decide(test_rows, predN):
    """Per instance: chosen candidate by predicted T, oracle by measured T. Returns regret stats."""
    by = defaultdict(list)
    for r, n_hat in zip(test_rows, predN):
        by[inst_key(r)].append((r, n_hat))
    regrets, catastrophic, agree, n_inst = [], 0, 0, 0
    for key, lst in by.items():
        conv = [(r, total_time(r, r["N"])) for r, _ in lst if r["N"]]
        if not conv:
            continue
        n_inst += 1
        o_r, o_t = min(conv, key=lambda z: z[1])
        c_r, _ = min(lst, key=lambda z: total_time(z[0], z[1]))
        if not c_r["N"]:
            catastrophic += 1
            regrets.append(float("inf"))
        else:
            regrets.append(total_time(c_r, c_r["N"]) / o_t)
        agree += int(c_r["cand"] == o_r["cand"])
    fin = [x for x in regrets if math.isfinite(x)]
    return {"instances": n_inst, "agree_with_oracle": agree, "catastrophic": catastrophic,
            "geo_regret": float(np.exp(np.mean(np.log(fin)))) if fin else None,
            "worst_finite_regret": max(fin) if fin else None}


def fixed_policy(test_rows, choose):
    by = defaultdict(list)
    for r in test_rows:
        by[inst_key(r)].append(r)
    preds = []
    for r in test_rows:
        preds.append(None)
    # emulate predicted N: chosen candidate gets its true N (or cap) and the rest get +inf so the argmin lands on it
    chosen = {k: choose(v) for k, v in by.items()}
    return [(r["N"] if r["N"] else r["iters_run"]) if chosen[inst_key(r)] is r else 1e30 for r in test_rows]


def evaluate(train, test, label):
    res = {"split": label, "n_train": len(train), "n_test": len(test), "models": {}, "baselines": {}}
    ytr, yte = y_of(train), y_of(test)
    for fs, cols in BASE.items():
        for mname, model in (("gbr", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                                subsample=0.8, random_state=0)),
                             ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0)))):
            model.fit(X_of(train, cols), ytr)
            p = model.predict(X_of(test, cols))
            rho = spearmanr(p, yte).correlation if len(test) > 2 else None
            dec = decide(test, np.exp(p))
            res["models"][f"{fs}/{mname}"] = {"spearman_logN": rho, "mae_logN": float(np.mean(np.abs(p - yte))),
                                               **dec}
    std = lambda v: next((r for r in v if r["cand"] == "std_ruiz"), v[0])
    exact = lambda v: next((r for r in v if r["cand"] in ("reloc:sort", "reloc:bisect")), v[0])
    minnorm = lambda v: min(v, key=lambda r: r["features"]["log_normL"])
    oracleN = lambda v: min(v, key=lambda r: (r["N"] if r["N"] else 1e30))
    for bname, ch in (("never_relocate(std_ruiz)", std), ("always_relocate_exact", exact),
                      ("min_normL", minnorm), ("min_N_ignoring_t", oracleN)):
        res["baselines"][bname] = decide(test, np.array(fixed_policy(test, ch)))
    return res


def show(res):
    print(f"\n=== {res['split']}   train={res['n_train']} test={res['n_test']}")
    print("  %-26s %-9s %-8s %-7s %-6s %-10s %s" % ("policy", "spearman", "maeLogN", "agree", "catas", "geoRegret",
                                                     "worst"))
    for k, v in list(res["models"].items()) + list(res["baselines"].items()):
        print("  %-26s %-9s %-8s %-7s %-6s %-10s %s" % (
            k, ("%.3f" % v["spearman_logN"]) if v.get("spearman_logN") is not None else "-",
            ("%.2f" % v["mae_logN"]) if "mae_logN" in v else "-",
            f"{v['agree_with_oracle']}/{v['instances']}", v["catastrophic"],
            ("%.3f" % v["geo_regret"]) if v["geo_regret"] else "-",
            ("%.2f" % v["worst_finite_regret"]) if v["worst_finite_regret"] else "-"))


results = []
# IN-DISTRIBUTION SIZE TRANSFER within the small grid: same families, train on n <= 1000, test on n = 2000.
# This is the repeated-workload setting (many instances of known families), unlike leave-one-family-out.
tr_sz = [r for r in rows if r["grid"] == "small" and r["size"] <= 1000]
te_sz = [r for r in rows if r["grid"] == "small" and r["size"] == 2000]
if len(tr_sz) > 20 and len(te_sz) > 5:
    results.append(evaluate(tr_sz, te_sz, "SIZE TRANSFER within families: train n<=1000 -> test n=2000"))
    for fam in sorted(set(r["family"] for r in te_sz)):
        te_f = [r for r in te_sz if r["family"] == fam]
        results.append(evaluate(tr_sz, te_f, f"  ... same split, test family {fam} only"))
small = [r for r in rows if r["grid"] == "small"]
large = [r for r in rows if r["grid"] == "large"]
if small and large:
    results.append(evaluate(small, large, "SIZE TRANSFER: train small (<=2k) -> test large (8k-30k)"))
fams = sorted(set(r["family"] for r in rows))
for fam in fams:
    tr = [r for r in rows if r["family"] != fam]
    te = [r for r in rows if r["family"] == fam]
    if len(tr) > 20 and len(te) > 5:
        results.append(evaluate(tr, te, f"FAMILY HOLD-OUT: {fam}"))
for res in results:
    show(res)
json.dump(results, open(a.out, "w"), indent=1, default=str)
print("\nFIT_DONE")
