"""Solve-time router: the `solve` action of the agent. Consult the vault, choose (formulation, operator, device) for
this instance, run it to the requested accuracy inside a wall-clock budget, verify on the original problem, record the
evidence, and return the tool's feedback.

The DEVICE is a candidate, not an argument (see "DEVICE AS A CANDIDATE" below): `device` names the MACHINE, and
`devices=("gpu", "cpu")` offers that machine's GPU and its CPU as two separately priced candidates.

Decision rule (lessons applied): iteration counts come from EVIDENCE for the formulation (never from a norm bound;
exact operators share N), per-iteration cost and setup come from CALIBRATION of the operator on this device (never
from padding). A candidate without evidence is EXPLORED: its run is charged to the budget, and if it reaches the
target it is itself the solve. Feedback names what binds (the formulation's N or the operator's t) and, when a faster
exact operator would change the decision, emits a SPEC with the in-solver per-call budget and the contract.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# baselines and the reference solver run ONLY in isolated envs (jaxenv2 must never be the fallback interpreter)
os.environ.setdefault("OPTEVOLVE_BASELINE_ENV_ROOT", "/home/miria/baseline_envs" if os.path.isdir("/home/miria/baseline_envs")
                      else "/workspace:/root/baseline_envs")
import numpy as np

# The tool's SHIPPED exploration order (a planner rule the agent may replace): structure-exploiting, row-scaled
# formulations first, the unscaled standard formulation last.
#
# Recognition-gated engines (`lasso:*`) come first, and that placement is not a benchmark prior. They are only
# OFFERED for an instance whose (P, q, A, l, u) blocks have been checked to be a lasso encoding, a test that costs one
# pass over the data and fires on none of the other four QP families. When it does fire it is the strongest evidence
# available before any solve: the instance is not a 2n+m-variable QP, it is an n-variable problem with a small active
# support, and every other candidate in this list will solve the larger problem. Exploring it first is what the
# measurement says (0.048 s against 17.6 s at n = 100,000); exploring it last would leave it untried, because the
# stopping rule ends exploration at twice the time to the first verified answer.
# The structural specialists (specialist_struct.py) sit in the SAME slot and for the same stated reason: each is
# offered only after the instance's own blocks have been checked to be that structure, and in each case recognition
# says the instance is a smaller or a differently shaped problem than its encoding (a network flow rather than a
# constraint matrix, an n-dimensional group-sparse regression rather than one cone per group, a ball around a handful
# of support points rather than one cone per sample). The placement is the shipped rule applied uniformly, not a
# per-family tuning: it is NOT adjusted using any held-out measurement, and where it costs a first visit (group lasso,
# where our own cone formulation is faster than the specialist) that cost is reported rather than ordered away.
DEFAULT_ORDER = ("engine:lasso:celer", "engine:lasso:sklearn", "engine:lasso:skglm",
                 "engine:meb:coreset", "engine:glasso:skglm",
                 "engine:flow:pot", "engine:flow:ortools", "engine:flow:networkx",
                 "reloc_rows", "reloc_cone", "engine:clarabel", "reloc_rows_prod", "std_ruiz", "reloc", "reloc_prod",
                 "reloc_cone_raw", "std_raw", "engine:highs", "engine:piqp", "engine:scs", "engine:ecos",
                 "engine:pdlp")

from uni import (make_instance, permute_instance, diagnose, plan_block, plan_product, build_formulation, reference,
                 measure_ladder, time_clean, LADDER)
from vault import Vault

HAND = {"component:bisect60": {"impl": "bisect"}, "component:breakpoint_hand": {"impl": "breakpoint"},
        "component:newton12_hand": {"impl": "newton", "iters": 12}}


# ------------------------------------------------------------------------------------------------------------------
# DEVICE AS A CANDIDATE
#
# T = setup + N(formulation) * t(operator, device). N belongs to the FORMULATION and carries between devices (the
# iteration is the same map); setup and t do not. So the same (formulation, operator) on this machine's GPU and on
# its CPU are two candidates, priced from one measured N and from each device's OWN calibration. The GPU keeps the
# machine's existing vault key ("rtx4090", "h100") so no stored calibration or evidence is reinterpreted, and the CPU
# of the same machine is a SEPARATE device with key "<machine>_cpu": a 20-thread i7 and a 128-thread Xeon are no more
# the same device than a 4090 and an H100 are, and neither one's calibration may be read as the other's.
#
# Our solver reaches the CPU through the SAME JAX code path (jax.default_device places the build and the traced
# iteration), so nothing numerical and nothing in the acceptance check changes. A device this process has no backend
# for is DECLINED by device_variants, never priced from a guess; a device with no calibration is CALIBRATED by the
# existing STEP B, which is a measurement, not an estimate.
CPU_SUFFIX = "_cpu"


# ------------------------------------------------------------------------------------------------------------------
# TILE AS A CANDIDATE
#
# A fused kernel runs T iterations of the map inside ONE launch; which (block geometry, T) TILE is fastest is a
# property of the DEVICE, not of the problem, and carrying one device's choice to another costs up to 4.82x
# (results/fusion_lp_20260919). So the tile belongs in the same cost model as everything else:
#     T = setup + N(formulation) * t(operator, device, TILE)
# N is unchanged by the tile in the strongest possible sense: admission shows one fused launch is BITWISE identical
# to T sequential unfused launches, so the iterates are the same numbers and the iteration count is literally the
# same. setup and t are not, so each admissible tile is a SEPARATE candidate priced by its own calibration.
#
# `tiles` is a provider: tiles(d, formulation, device_key) -> [tile, ...], where a tile is any hashable the builder
# understands. NO_TILE (do not fuse) is prepended unconditionally, so the ballot always contains the shipped
# behaviour and a provider that returns nothing leaves this file's decisions bit-for-bit what they were.
#
# A provider must return only tiles ADMISSIBLE on that device, which for a shared-memory fused kernel means the
# tile's halo fits the device's opt-in shared memory per block. That is a structural rule, checked without timing,
# and it is why the candidate SET already differs across devices before any measurement.
NO_TILE = None


def tile_variants(tiles, d, formulation, device_key):
    """[None] + whatever the provider offers. None is "do not fuse" and is never removed."""
    if tiles is None:
        return [NO_TILE]
    try:
        offered = list(tiles(d, formulation, device_key)) or []
    except Exception:
        offered = []                       # a failing provider must not break solving: fall back to not fusing
    return [NO_TILE] + [t for t in offered if t is not NO_TILE]


def _jax_device(kind):
    import jax
    try:
        return jax.devices("cpu" if kind == "cpu" else "cuda")[0]
    except (RuntimeError, IndexError):
        return None


def device_variants(device: str, devices=None):
    """[(vault device key, jax device)] this process can actually run.

    devices=None keeps the shipped behaviour exactly: one candidate device, the process default, under the caller's
    key. devices=("gpu", "cpu") makes the device a candidate axis, dropping any backend this process does not have.
    """
    if not devices:
        return [(device, None)]
    out = []
    for k in devices:
        jd = _jax_device(k)
        if jd is None:
            continue                       # DECLINED: no backend here, and a borrowed calibration is not evidence
        out.append((device if k == "gpu" else device + CPU_SUFFIX, jd))
    return out


def _on(jax_device):
    """Place everything built and run inside on this device. None = the process default (shipped behaviour)."""
    if jax_device is None:
        return contextlib.nullcontext()
    import jax
    return jax.default_device(jax_device)


def make_case(case: str):
    """'family:size[:rho]' (ladder encoding, seeds 0/7), 'H:family:size[:rho]' (held-out, seeds 3/11), 'MM:NAME'."""
    parts = case.split(":")
    if parts[0] == "MM":
        from realdata import load_mm
        return load_mm(parts[1])
    if parts[0] in ("RB", "RBCAP"):                     # 'RB:n_prompts[:budget_frac]' RouterBench routing LP
        from routerbench import make_routing, make_routing_capacitated
        n_p = int(parts[1])
        frac = float(parts[2]) if len(parts) > 2 else 0.5
        mk = make_routing_capacitated if parts[0] == "RBCAP" else make_routing
        d = mk(n_p, frac, seed=0)
        return permute_instance({k: d[k] for k in ("P", "q", "r", "A", "l", "u")}, 1000)
    if parts[0] == "NETLIB":
        from netlib import load_netlib
        return load_netlib(parts[1])
    if parts[0] in ("LP", "HLP"):                       # 'LP:family:size[:key=value...]'; HLP = held-out seed
        from lp_families import make_lp
        kw = {k: float(v) for k, v in (x.split("=") for x in parts[3:])}
        seed = 3 if parts[0] == "HLP" else 0
        d = make_lp(parts[1], int(parts[2]), seed=seed, **kw)
        return permute_instance({k: d[k] for k in ("P", "q", "r", "A", "l", "u")}, 1000 + seed)
    if parts[0] in ("SOCP", "HSOCP"):                   # 'SOCP:family:size'; HSOCP = held-out seed
        import socp
        seed = 3 if parts[0] == "HSOCP" else 0
        d, _ = socp.permute_socp(socp.make_socp(parts[1], int(parts[2]), seed=seed), 1000 + seed)
        return d
    if parts[0] == "CBLIB":
        import cblib
        from problem_adapters import normalize
        try:
            return normalize(cblib.load_cbf(cblib.instance_path(parts[1]), dense_f=True))
        except (ValueError, MemoryError):
            d = normalize(cblib.load_cbf(cblib.instance_path(parts[1]), dense_f=False))
            import scipy.sparse as _sp
            for c in d["cones"]:                         # 1-D sparse f -> the 1 x n sparse row socp.py expects
                if _sp.issparse(c["f"]) and len(c["f"].shape) == 1:
                    c["f"] = _sp.csr_matrix(_sp.coo_array(c["f"]).reshape((1, c["f"].shape[0])))
            return d
    held = parts[0] == "H"
    parts = parts[1:] if held else parts
    kw = {"rho": float(parts[2])} if len(parts) > 2 else {}
    seed, pseed = (3, 11) if held else (0, 7)
    return permute_instance(make_instance(parts[0], int(parts[1]), seed=seed, **kw), pseed)


def operator_kwargs(comp, vault=None) -> dict:
    if comp.id in HAND:
        return dict(HAND[comp.id])
    if comp.id == "component:product_bisect":
        return {"impl": "bisect"}                      # the library product projection; synthesized ones carry a file
    path = comp.payload["file"]
    return {"impl": "file:" + (vault.resolve(path) if vault is not None else path)}


def candidates(d, vault: Vault, include_uncertified=False, device: str = "", devices=None, engines=True,
               tiles=None):
    """Every (formulation or engine, operator, DEVICE, TILE) the vault can currently run on this instance.

    Our own formulations are crossed with device_variants(device, devices): one candidate per device this process can
    actually run, and then with tile_variants(...), one candidate per admissible tile ON THAT DEVICE plus the
    do-not-fuse candidate. Engines are external solvers in their own CPU interpreters, so they are not crossed with
    our devices or tiles; they carry the machine's key and no tile, as before.
    """
    import problem_adapters as PA
    k = PA.kind(d)
    pl = PA.plans(d)
    plan, pplan, cplan = pl["single"], pl["product"], pl["cone"]
    out = [("std_raw", None, None, {}), ("std_ruiz", None, None, {})]
    if k == "socp":
        if cplan is not None:
            out += [("reloc_cone", cplan, None, {}), ("reloc_cone_raw", cplan, None, {})]
    else:
        if plan is not None:
            set_type = "box_hyperplane" if plan["kind"] in ("box_hyperplane", "box_slab") else plan["kind"]
            comps = [c for c in vault.current("component")
                     if c.payload.get("set") == set_type
                     and (include_uncertified or (c.payload.get("certificate") or {}).get("certified") == 1.0)]
            if plan["kind"] == "simplex":
                out += [(v, plan, None, {"impl": "sort"}) for v in ("reloc", "reloc_rows")]
            for c in comps:
                for v in ("reloc", "reloc_rows"):
                    out.append((v, plan, c, operator_kwargs(c, vault)))
        if pplan is not None:
            # the ROW-SELECTION RULE is a candidate, not a global switch: on real MIPLIB relaxations selecting
            # absorbable rows by norm instead of width converged 10x faster on several instances, and neither
            # rule dominates, so the cost model compares them by measurement
            rules = ["width", "norm", "narrow"] + [e.payload.get("rule_name") for e in vault.query("transformation",
                     lambda p: p.get("name") == "product_rule" and p.get("admitted"))]
            seen = set()
            for rule in rules:
                if not rule or rule in seen:
                    continue
                seen.add(rule)
                pl = pplan if rule == "width" else plan_product(d, rule=_rule_fn(rule, vault))[0]
                if pl is None:
                    continue
                for c in vault.components("product_box_slab"):
                    for v in ("reloc", "reloc_rows"):
                        out.append((v, pl, c, operator_kwargs(c, vault)))
    devs = device_variants(device, devices)
    out = [(c, p, comp, kw, lab, tl) for (c, p, comp, kw) in out for lab, _ in devs
           for tl in tile_variants(tiles, d, _form_key(c, p), lab)]
    admitted_engines = {e.payload.get("engine") for e in vault.query("transformation",
                        lambda p: p.get("name") == "engine" and p.get("admitted", True))}
    if engines:
        for eng in tuple(PA.engines(k)) + tuple(PA.structural_engines(d)):
            if not admitted_engines or eng in admitted_engines:
                out.append((f"engine:{eng}", None, None, {"engine": eng}, devs[0][0], NO_TILE))
    return out, plan, pplan, devs


def _known_per_call(comp, device):
    """Calibrated per-call cost of a component on this device from the vault (inf if unknown): used only to pick the
    representative operator for a formulation's single N-measuring run."""
    if comp is None:
        return 0.0
    cal = (comp.payload.get("calibration") or {}).get(device) or {}
    return float(cal.get("per_call_us", float("inf")))


def _rule_fn(rule, vault):
    """A shipped rule name, or code admitted into the vault as a product_rule transformation."""
    if rule in ("width", "norm", "narrow", "norm_per_col", "eq_first"):
        return rule
    e = next((e for e in vault.query("transformation", lambda p: p.get("name") == "product_rule"
                                     and p.get("rule_name") == rule and p.get("admitted"))), None)
    if e is None:
        return "width"
    ns = {}
    exec(open(vault.resolve(e.payload["file"])).read(), ns)
    return ns["score"]


def _with_tile(row):
    """The builder kwargs for this candidate. A do-not-fuse candidate is built with exactly the kwargs it was built
    with before this axis existed, so the shipped path is untouched."""
    if row.get("tile") is NO_TILE:
        return row["kw"]
    return {**row["kw"], "tile": row["tile"]}


def _form_key(cand, plan):
    if plan is not None and plan.get("kind") == "product":
        rule = plan.get("rule", "width")
        return cand + "_prod" + ("" if rule == "width" else "@" + str(rule))
    return cand


def solve(case: str, vault: Vault, device: str, tol: float = 1e-3, budget_s: float = 30.0, record_evidence: bool = True,
          d=None, obj_ref=None, include_uncertified=False, selector=None, cost_term=None, devices=None,
          engines=True, tiles=None) -> dict:
    d = make_case(case) if d is None else d
    import problem_adapters as PA
    d = PA.normalize(d)
    if obj_ref is None:
        obj_ref, _ = PA.reference(d)        # the JUDGE's oracle, outside the policy's clock (it is not a policy cost)
    t_start = time.perf_counter()           # the policy's clock: structure detection, exploration, calibration, solve
    cands, plan, pplan, devs = candidates(d, vault, include_uncertified, device=device, devices=devices,
                                          engines=engines, tiles=tiles)
    DEVMAP = dict(devs)
    dev_keys = [lab for lab, _ in devs]
    explore_dev = dev_keys[0]                # the N-measuring runs stay on the machine's primary device
    key_tol = f"{tol:g}"

    # EVIDENCE. N is a property of the FORMULATION (exact operators share it); setup and per-iteration cost are
    # properties of the (formulation, operator) pair on this device. A failed run is evidence too: a formulation that
    # did not reach the target within C iterations is not re-explored unless the budget now allows more than C.
    def ev_rows(form_key, dev=None):
        if dev is not None:
            return vault.evidence(case=case, formulation=form_key, device=dev)
        return [r for dv in dev_keys for r in vault.evidence(case=case, formulation=form_key, device=dv)]

    def ev_N(form_key):
        # N is a property of the FORMULATION, so evidence from ANY candidate device of this machine counts. That is
        # exactly what lets a device be priced without re-measuring its iteration count.
        rows = ev_rows(form_key)
        ns = [r.payload["N_at"].get(key_tol) for r in rows if (r.payload.get("N_at") or {}).get(key_tol)]
        caps = [r.payload.get("iters_run", 0) for r in rows if "N_at" in r.payload and not r.payload["N_at"].get(key_tol)]
        return (min(ns) if ns else None), (max(caps) if caps else 0)

    def ev_t(form_key, comp_id, dev, tile=NO_TILE):
        # setup and per-iteration cost belong to the (operator, device, TILE) triple. A stored row with no tile
        # field is a pre-tile record and means "do not fuse", which is what NO_TILE asks for.
        rows = [r.payload for r in ev_rows(form_key, dev) if r.payload.get("component") == comp_id
                and r.payload.get("tile", NO_TILE) == tile
                and r.payload.get("per_iter_s") is not None and r.payload.get("calibrated")]
        return (rows[-1]["setup_s"], rows[-1]["per_iter_s"]) if rows else None

    def record(rec, src):
        if record_evidence:
            vault.add("evidence", rec, provenance={"source": src})

    slice_s = [budget_s]

    def ladder_run(row, cap_iters):
        if row["cand"].startswith("engine:"):
            er = PA.run_engine(d, row["kw"]["engine"], obj_ref, tol, slice_s[0])
            rec = {"case": case, "device": row["device"], "formulation": row["formulation"], "component": None,
                   "N_at": {key_tol: 1 if er["accepted"] else None}, "setup_s": er.get("t_solver_s") or 0.0,
                   "per_iter_s": 0.0, "wall_s": er["wall_s"], "iters_run": 1, "tol": tol, "accepted": er["accepted"],
                   "calibrated": bool(er["accepted"]), "engine": er}
            record(rec, "router.engine_run")
            return rec
        with _on(DEVMAP.get(row["device"])):            # the SAME code path, placed on the candidate's device
            form = PA.build(d, row["cand"], row["plan"], _with_tile(row))
            t0 = time.perf_counter()
            m = PA.ladder(form, d, obj_ref, tuple(sorted({1e-2, tol}, reverse=True)), cap_iters, 50,
                          time.perf_counter() + slice_s[0])              # stop at the target, not the whole ladder
            wall_s = time.perf_counter() - t0
        rec = {"case": case, "device": row["device"], "formulation": row["formulation"], "component": row["component"],
               "tile": row["tile"],
               "N_at": m["N_at"], "setup_s": m["setup_s"], "per_iter_s": m["t_iter"], "wall_s": wall_s,
               "iters_run": m["iters_run"], "rv": m["row_violation"], "gap": m["obj_gap"], "tol": tol,
               "accepted": m["N_at"].get(key_tol) is not None, "calibrated": False}
        record(rec, "router.ladder_run")
        return rec

    def calibrate(row, iters=300):
        # The per-iteration cost is the MEDIAN of several short timed blocks, not one block: this machine shares its
        # GPU with a resident inference server and one block picks up its stalls. The spread that the median survived
        # is recorded with the answer so the noise stays visible in the record.
        with _on(DEVMAP.get(row["device"])):
            form = PA.build(d, row["cand"], row["plan"], _with_tile(row))
            c = time_clean(form, iters, K=50)
        rec = {"case": case, "device": row["device"], "formulation": row["formulation"], "component": row["component"],
               "tile": row["tile"],
               "setup_s": c["setup_s"], "per_iter_s": c["per_iter_s"], "calibrated": True, "cal_iters": iters}
        for k in ("per_iter_min_s", "per_iter_max_s", "per_iter_first_s", "cal_blocks", "cal_K", "cal_chunks_per_block", "cal_spread"):
            if k in c:
                rec[k] = c[k]
        rec["cal_iters"] = c.get("cal_iters", iters)
        record(rec, "router.calibrate")
        return rec

    table = []
    for cand, pl, comp, kw, dv, tl in cands:
        fk = _form_key(cand, pl)
        table.append({"formulation": fk, "component": comp.id if comp is not None else kw.get("impl"), "cand": cand,
                      "plan": pl, "kw": kw, "device": dv, "tile": tl,
                      "per_call_known": _known_per_call(comp, dv)})
    explored = 0
    elapsed = lambda: time.perf_counter() - t_start
    solve_run = {}
    first_solution_s = None
    # STEP A: one ladder run per formulation whose N is unknown, with its fastest known operator (so it may already
    # be the solve). Order: cheapest-looking formulations first.
    allowed = None
    if selector is not None:
        from rules import instance_features, neighbors
        try:
            allowed = list(selector(instance_features(case, d, plan, pplan), neighbors(vault, case, device, tol)))
        except Exception:
            allowed = None                                     # a failing rule must not break solving: explore all
    keys = {r["formulation"] for r in table}
    base_of = lambda f: f.split("@")[0]
    order = [f for d0 in DEFAULT_ORDER for f in sorted(k for k in keys if base_of(k) == d0)]
    if allowed is not None:
        order = [f for f in (allowed if isinstance(allowed, list) else order) if f in order] + \
                [f for f in order if f not in allowed and ev_N(f)[0] is not None]
    to_explore = [f for f in order if ev_N(f)[0] is None]
    for i, fk in enumerate(order):
        # SHIPPED STOPPING RULE (replaceable by an admitted planner rule): once a verified answer exists, keep exploring
        # only while exploration has cost less than twice the time to that answer
        if first_solution_s is not None and elapsed() > 2.0 * first_solution_s:
            break
        N, cap_seen = ev_N(fk)
        if N is None and fk not in to_explore:
            continue
        rows = [r for r in table if r["formulation"] == fk and r["device"] == explore_dev
                and r["tile"] is NO_TILE] or \
               [r for r in table if r["formulation"] == fk and r["tile"] is NO_TILE] or \
               [r for r in table if r["formulation"] == fk]
        rep = min(rows, key=lambda r: r["per_call_known"])         # N is measured WITHOUT fusing: the tile cannot
        # change it (the fused launch is bitwise identical), and the unfused path is the one with a known cost
        remaining = max(1, sum(1 for f in to_explore[to_explore.index(fk):] if ev_N(f)[0] is None)) if fk in to_explore else 1
        left = (budget_s - elapsed()) / remaining                 # fair slice: one stalled formulation cannot eat it all
        cap_iters = int(min(200000, max(1000, left / 2e-5)))   # wall cap assuming >= 20 us/iter until calibrated
        est = ev_t(fk, rep["component"], rep["device"], rep["tile"])
        if est and est[1] > 0:
            cap_iters = int(max(1000, left / est[1]))
        if N is None and cap_iters > cap_seen and left > 0:
            slice_s[0] = left
            rec = ladder_run(rep, cap_iters)
            explored += 1
            if rec["accepted"]:
                solve_run[(fk, rep["component"], rep["device"], rep["tile"])] = rec
                if first_solution_s is None:
                    first_solution_s = elapsed()
    # INCUMBENT: the fastest exploration run that already reached the target is a valid solve; nothing below may lose it
    incumbent = None
    for (fk, cid, dv, tl), rec in solve_run.items():
        t_rec = rec["setup_s"] + rec["N_at"][key_tol] * rec["per_iter_s"]
        if incumbent is None or t_rec < incumbent[0]:
            incumbent = (t_rec, fk, cid, rec, dv, tl)
    # STEP B: calibrate only candidates the vault's component calibration says could beat the incumbent by >10%
    # (default cost-model term: per-iteration = incumbent per-iteration - incumbent per-call + candidate per-call)
    inc_row = None if incumbent is None else next((r for r in table if r["formulation"] == incumbent[1]
                                                    and r["component"] == incumbent[2]
                                                    and r["device"] == incumbent[4]
                                                    and r["tile"] == incumbent[5]), None)
    for r in table:
        N, _ = ev_N(r["formulation"])
        r["N_hat"] = N
        r["setup_hat"] = r["per_iter_hat"] = r["T_hat"] = None
        if N is None:
            continue
        tt = ev_t(r["formulation"], r["component"], r["device"], r["tile"])
        if tt is None and incumbent is not None and r is not inc_row:
            inc_call = (inc_row or {}).get("per_call_known", float("inf"))
            cand_call = r["per_call_known"]
            # the prediction below reads the incumbent's MEASURED per-iteration cost and swaps one operator's
            # per-call cost for another's. Both numbers belong to one device, so it is only valid on the incumbent's
            # device: a candidate on the OTHER device is calibrated instead of guessed.
            # ... and only within one TILE: the per-call swap reads a measured per-iteration cost that already
            # contains the incumbent's launch structure, and a different tile changes that structure, so a tile
            # with no calibration is MEASURED rather than predicted.
            if math.isfinite(inc_call) and math.isfinite(cand_call) and r["formulation"] == incumbent[1] \
                    and r["device"] == incumbent[4] and r["tile"] == incumbent[5]:
                pred = incumbent[3]["per_iter_s"] + 1e-6 * (cand_call - inc_call)
                if pred * N > 0.9 * incumbent[3]["per_iter_s"] * N:
                    continue                                   # predicted not to beat the incumbent: skip calibration
            elif r["formulation"] != incumbent[1] and not math.isfinite(cand_call):
                pass      # unknown cost on a DIFFERENT formulation: calibrate it rather than skip, or a component
                          # with no calibration (the product projection on a new device) can never be priced, never
                          # chosen, and never turned into a spec (measured on MIPLIB: 4 instances converged with
                          # T_hat = None)
        if tt is None and cost_term is not None:
            # an admitted cost-model term predicts this pair's per-iteration cost from evidence on OTHER instances,
            # which is how a first visit avoids paying for a calibration run
            try:
                from rules import instance_features, neighbors, candidate_summaries
                ph = cost_term(instance_features(case, d, plan, pplan), {"formulation": r["formulation"],
                               "component": r["component"], "device": r["device"], "tile": r["tile"]},
                               candidate_summaries(vault, r["device"]), neighbors(vault, case, r["device"], tol))
                if ph and ph > 0:
                    tt = (incumbent[3]["setup_s"] if incumbent else 0.0, float(ph))
            except Exception:
                tt = None
        if tt is None and budget_s - elapsed() > 1.0:
            c = calibrate(r)
            tt = (c["setup_s"], c["per_iter_s"])
            explored += 1
        if tt is not None:
            r["setup_hat"], r["per_iter_hat"] = tt
            r["T_hat"] = tt[0] + N * tt[1]
    if incumbent is not None and inc_row is not None and inc_row["T_hat"] is None:
        inc_row.update({"setup_hat": incumbent[3]["setup_s"], "per_iter_hat": incumbent[3]["per_iter_s"],
                        "T_hat": incumbent[0]})
    known = [r for r in table if r["T_hat"] is not None]
    choice = min(known, key=lambda r: r["T_hat"]) if known else None
    solved, final = False, None
    if choice is not None:
        final = solve_run.get((choice["formulation"], choice["component"], choice["device"], choice["tile"]))
        if final is None and budget_s - elapsed() > choice["T_hat"] * 1.5:
            slice_s[0] = budget_s - elapsed()
            final = ladder_run(choice, int(max(1000, 2 * choice["N_hat"])))
            if final["accepted"] and first_solution_s is None:
                first_solution_s = elapsed()
        if (final is None or not final["accepted"]) and incumbent is not None:
            choice, final = inc_row, incumbent[3]                # fall back to the verified incumbent
        solved = bool(final and final["accepted"])
    feedback = feedback_for(table, choice, d, plan, pplan, vault, case)
    explore_s = sum(e.payload.get("wall_s", 0.0) for dv in dev_keys for e in vault.evidence(case=case, device=dv)
                    if e.provenance.get("source") == "router.ladder_run") if explored else 0.0
    feedback["exploration"] = {"first_visit_wall_s": round(time.perf_counter() - t_start, 2), "runs": explored,
                               "note": "exploration and calibration on a NEW instance; repeat visits route from evidence. "
                                       "A planner rule that explores fewer formulations lowers this."}
    return {"case": case, "device": device, "tol": tol, "solved": solved, "first_solution_s": first_solution_s,
            "time_to_solution_s": None if not solved else choice["setup_hat"] + final["N_at"][key_tol] * choice["per_iter_hat"],
            "wall_total_s": time.perf_counter() - t_start, "choice": choice and {k: choice[k] for k in
            ("formulation", "component", "device", "tile", "N_hat", "T_hat")}, "explored": explored,
            "devices": dev_keys, "device_chosen": choice and choice["device"],
            "tiles_offered": sorted({str(r["tile"]) for r in table}),
            "tile_chosen": choice and choice["tile"],
            "table": [{k: r[k] for k in ("formulation", "component", "device", "tile", "N_hat", "T_hat")}
                      for r in table],
            "feedback": feedback}


def feedback_for(table, choice, d, plan, pplan, vault, case=None):
    """What binds, and the spec that would change the decision."""
    fb = {"binding": None, "spec": None, "uncovered": []}
    if choice is not None and not any(r["formulation"].startswith("reloc") and r["N_hat"] and r["per_iter_hat"]
                                      for r in table):
        tried = [r["formulation"] for r in table if r["formulation"].startswith("reloc")]
        fb["binding"] = ("the chosen route is " + str(choice["formulation"]) + "; no relocated formulation was both "
                         "measured and calibrated here" + (f" (candidates present: {sorted(set(tried))})" if tried else
                         " (no relocatable structure was detected)") + ", so there is no operator spec to meet: the "
                         "lever is a planner rule or a transformation, or more budget for exploration")
    if choice is None:
        # Two different failures used to report the same cause, and the second report was simply false: on a 64x64
        # transport instance structure detection consumed a 120 s budget before ANY formulation was tried, and the
        # router still answered "the formulation's N binds". Nothing had been measured. Separate them, and name the
        # detection cost when that is what ran out, because the lever there is the planner, not a transformation.
        # (The second string literal below was also a dead statement, so the message was truncated mid-sentence.)
        if not any(r["N_hat"] is not None or r["T_hat"] is not None for r in table):
            fb["binding"] = ("no formulation was measured at all inside the budget: structure detection and setup "
                             "consumed it before the first ladder run, so the lever is cheaper detection or a larger "
                             "budget. This is NOT evidence about N.")
        else:
            fb["binding"] = ("nothing reached the target inside the budget: the formulation's N binds (no formulation "
                             "fast enough), so a new TRANSFORMATION or planner rule is the lever, not a faster operator")
        return fb
    rel = [r for r in table if r["formulation"].startswith("reloc") and r["N_hat"] and r["per_iter_hat"]]
    if not choice["formulation"].startswith("reloc") and rel:
        r = min(rel, key=lambda r: r["T_hat"])
        # budget for a NEW operator assumes the setup of the best relocated candidate it would replace
        per_call_budget = (choice["T_hat"] - r["setup_hat"]) / r["N_hat"]
        fb["binding"] = (f"operator cost t: relocation needs {r['N_hat']} iterations vs the chosen formulation, but "
                         f"its per-iteration cost {1e6 * r['per_iter_hat']:.1f} us is too high")
        if per_call_budget > 0:
            pl = r["plan"] or {}
            fb["spec"] = {"kind": "decision_flip", "case": case, "formulation": r["cand"], "replace_component": r["component"],
                          "set": "product_box_slab" if pl.get("kind") == "product" else "box_hyperplane",
                          "n_x": pl.get("n_x"), "current_per_iteration_us": round(1e6 * r["per_iter_hat"], 2),
                          "per_iteration_budget_us": round(1e6 * per_call_budget, 2),
                          "note": "in-solver per-iteration budget of the relocated formulation (subtract the null-map "
                                  "base to get the per-call projection budget)"}
        else:
            fb["binding"] += "; even a free projection loses (setup plus base iteration exceed the budget)"
    elif choice["formulation"].startswith("reloc"):
        fb["binding"] = "chosen relocated formulation; a faster exact operator lowers T linearly in N"
        pl = choice["plan"] or {}
        fb["spec"] = {"kind": "improvement", "case": case, "formulation": choice["cand"],
                      "replace_component": choice["component"],
                      "set": "product_box_slab" if pl.get("kind") == "product" else "box_hyperplane",
                      "n_x": pl.get("n_x"), "current_per_iteration_us": round(1e6 * choice["per_iter_hat"], 2),
                      "per_iteration_budget_us": round(1e6 * choice["per_iter_hat"], 2),
                      "note": "improvement spec: any certified operator with lower in-solver per-iteration cost than "
                              "the current choice lowers this instance's time (and every instance with this structure)"}
    if pplan is not None and not vault.components("product_box_slab"):
        fb["uncovered"].append("a product of box-slab sets is absorbable here but the vault has no certified operator")
    return fb
