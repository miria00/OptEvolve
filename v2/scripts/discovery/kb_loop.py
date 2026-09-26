"""The KB creation loop on one real instance of a problem family: enumerate compositions
of knowledge-base ingredients on the family's base operators, typecheck each with the gate
rules, run every admitted recipe to the family's certificate, rank, and attach the knowledge
base's novelty verdict for the family's practice layer. Cost is counted in evaluations of the
base map; time is on this process's default device.

Families: logreg (--data DIR with X.npy, y.npy; layer glm), qp (--instance NAME from
data/maros_meszaros; layer qp), ot (--instance CLASS:RES:I1-I2 from DOTmark; layer lp)."""
import argparse, gc, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from atlas.evolve.kb_composer import enumerate_recipes, novelty, run_recipe, typecheck_recipe
from atlas.typing_ import TypeCheckError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--family", default="logreg", choices=("logreg", "qp", "qpgen", "ot"))
ap.add_argument("--data"); ap.add_argument("--instance")
ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--lam-ratio", type=float, default=0.1)
ap.add_argument("--bases", default=None, help="default per family: logreg fbs,fista; qp condat_vu; ot pdhg")
ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--max-evals", type=int, default=20000)
ap.add_argument("--K", type=int, default=64); ap.add_argument("--out", required=True)
ap.add_argument("--ruiz", type=int, default=0,
                help="qp/qpgen: Ruiz-equilibrate the problem with this many sweeps before the search, as the "
                     "measured arm does; without it Condat-Vu starts from a badly scaled instance and every "
                     "recipe inherits that, so the chain is asked to rescue a base that cannot converge")
ap.add_argument("--kappa", type=float, default=30.0, help="ot: primal weight omega0/kappa (kappa calibrated on a held-out pair)")
a = ap.parse_args()
if a.family == "logreg":
    from atlas.bench import logreg_cell as lc
    X = np.load(a.data + "/X.npy").astype(np.float64); y = np.load(a.data + "/y.npy").astype(np.float64)
    prob = lc.composite(X, y, lam=a.lam_ratio * lc.lam_max(X, y, a.alpha), alpha=a.alpha)
    desc, layer, bases = f"logreg n={X.shape[0]} p={X.shape[1]} alpha={a.alpha}", "glm", "fbs,fista"
elif a.family == "qpgen":
    # generated convex QP at a chosen size: --instance lasso:25000:0.001:0 (family:size:density:seed).
    # These are the sizes where a KKT factorisation dominates and the classical suite does not reach.
    from atlas.bench import qp_cell
    from atlas.bench.qp_gen import lasso, svm
    fam, size, dens, seed = a.instance.split(":")
    d = (lasso if fam == "lasso" else svm)(int(size), seed=int(seed), density=float(dens))
    prob = qp_cell.to_composite(d)
    desc = f"qpgen {fam} n={d['P'].shape[0]} m={d['A'].shape[0]} nnz={d['P'].nnz + d['A'].nnz} density={dens} seed={seed}"
    layer, bases = "qp", "condat_vu"
elif a.family == "qp":
    from atlas.bench import qp_cell
    prob = qp_cell.load(os.path.join(ROOT, "data", "maros_meszaros", a.instance + ".mat"))
    desc, layer, bases = f"qp {a.instance}", "qp", "condat_vu"
else:
    from atlas.bench import dotmark
    cls, res, pr = a.instance.split(":"); i1, i2 = map(int, pr.split("-"))
    prob = dotmark.composite(cls, int(res), i1, i2, operator="reduction")
    desc, layer, bases = f"ot {a.instance}", "lp", "pdhg"
bases = tuple((a.bases or bases).split(","))
ratio = None
if a.family == "ot":
    ratio = float(np.linalg.norm(np.asarray(prob.data["c"])) / np.linalg.norm(np.asarray(prob.data["b"]))) / a.kappa
    desc += f"; omega = omega0/{a.kappa:g} = {ratio:.4g}"
if a.ruiz and a.family in ("qp", "qpgen"):
    from atlas.schemes.rescale import rescale_qp_problem
    prob = rescale_qp_problem(prob, "ruiz", iters=a.ruiz)[0]
    desc += f"; ruiz={a.ruiz}"
print(f"device {jax.devices()[0]}; {desc}; bases {bases}; tol={a.tol:g}; novelty layer {layer}", flush=True)
admitted, rejected = [], []
from dataclasses import replace as _replace
for r in enumerate_recipes(bases=bases):
    if ratio is not None and r.base in ("pdhg", "condat_vu"):
        r = _replace(r, ratio=ratio)
    try:
        typecheck_recipe(r, prob); admitted.append(r)
    except TypeCheckError as e:
        rejected.append({"recipe": r.name(), "reason": str(e)[:200]})
print(f"{len(admitted)} admitted, {len(rejected)} rejected by the gates before any run", flush=True)
# Proof check: worst-case verdict of every fixed-coefficient recipe (admitted or rejected) over its operator
# class, and whether it agrees with the gate.
pep_verdict = {}
try:
    from atlas.evolve import pep_check
    from atlas.evolve.kb_composer import base_params
    from atlas.schemes.base_schemes import build_scheme
    if pep_check.available():
        allrec = admitted + [r for r in enumerate_recipes(bases=bases) if r.name() in {x["recipe"] for x in rejected}]
        groups = {}
        for r in allrec:
            groups.setdefault((r.base, r.step_frac, r.ratio), []).append(r)
        agree = disagree = 0
        adm = {r.name() for r in admitted}
        for (b, sf, ra), rs in groups.items():
            try:
                alpha = build_scheme(prob, b, base_params(prob, rs[0]), check=True).cert.averagedness
            except Exception:
                continue
            if not (alpha == alpha) or b == "fista":
                continue
            for r, pr in zip(rs, pep_check.pep_rank(rs, alpha_base=alpha)):
                pep_verdict[r.name()] = pr["verdict"]
                if pr["tau"] is not None and not pr["verdict"].startswith("PEP failed"):
                    ok = pr["verdict"].startswith("worst-case convergent")
                    if ok == (r.name() in adm): agree += 1
                    else: disagree += 1
        print(f"proof check: gate and worst-case analysis agree on {agree} of {agree + disagree} fixed-coefficient recipes", flush=True)
except Exception as ex:
    print("proof check skipped:", ex, flush=True)
rows = []
for r in admitted:
    out = run_recipe(prob, r, tol=a.tol, max_evals=a.max_evals, K=a.K)
    out.pop("x"); out.pop("u"); out["novelty"] = novelty(r, layer); out["pep"] = pep_verdict.get(r.name())
    jax.clear_caches(); gc.collect()        # drop this recipe's compiled program and buffers before the next
    rows.append(out)
    print(f"  {out['recipe']:62s} evals {out['evals']:6d} {out['wall_s']:8.3f}s gap {out['gap']:.1e} {'OK' if out['converged'] else 'CAP'}", flush=True)
    json.dump({"args": vars(a), "rejected": [{**x, "pep": pep_verdict.get(x["recipe"])} for x in rejected],
               "rows": rows}, open(a.out, "w"), indent=1)
ok = sorted([r for r in rows if r["converged"]], key=lambda r: r["wall_s"])
ref = next((r for r in rows if r["recipe"].startswith("fista")), None) or next(
    (r for r in rows if "+" not in r["recipe"] and r["converged"]), None)
print(f"\nRANKED (certified), reference: {ref['recipe'] if ref else 'none'}", flush=True)
for r in ok[:12]:
    sp = f"  {ref['wall_s'] / r['wall_s']:5.2f}x time, {ref['evals'] / max(r['evals'], 1):5.2f}x evals vs ref" if ref and ref["converged"] else ""
    print(f"  {r['wall_s']:8.3f}s {r['evals']:6d} evals  {r['recipe']:62s}{sp}  | {r['novelty']['verdict'][:80]}", flush=True)
print("LOOP_DONE", flush=True)
