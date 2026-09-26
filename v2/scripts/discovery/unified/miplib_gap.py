"""Gap analysis: route real MIPLIB 2017 LP relaxations through router.solve and record WHAT BINDS.

One instance per subprocess. A first-order run on a real LP can exhaust the GPU, diverge to NaN or take the process
down with it; the driver must still finish the sweep and report the failure as a finding, so --one is the unit of work
and the driver only collects JSON lines.

The acceptance reference is HiGHS in its isolated interpreter rather than router.solve's own Clarabel call. Both are
outside the policy's clock, but on these relaxations Clarabel is the slower and less reliable of the two (netlib.py
records five Netlib problems where uni.reference's Clarabel call is wrong), and the reference is the JUDGE: a bad one
turns an accuracy question into a routing conclusion. Its value is q'x, matching uni.reference and uni.accept, which
both leave out the objective constant r.

The vault is a COPY of the shipped one (same components, same transformations, same per-device calibrations), so every
instance is a genuine FIRST VISIT and the sweep never mutates the shipped evidence store.

Usage
  python miplib_gap.py --names a,b,c --out gaps.jsonl --vault vault_copy.jsonl [--budget 60] [--tol 1e-3]
  python miplib_gap.py --one NAME  --vault ... --budget 60        # the unit the driver spawns
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 8% of the H100's 80 GB is about 6.5 GB, which holds every formulation in this study and leaves the vLLM server
# that owns the other 70 GB of this box untouched. Preallocation off for the same reason.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.08")
os.environ.setdefault("OPTEVOLVE_BASELINE_ENV_ROOT", "/workspace:/root/baseline_envs")


def prepare_vault(src, dst, components_dir):
    """Copy the shipped vault and repoint every component's operator file at THIS machine's components directory.

    The vault records each synthesized operator as an ABSOLUTE path on the machine that wrote it
    ('/home/miria/OptEvolve/v2/vault/components/spec_r2.py'). The files themselves travel with the repo, so on any
    other machine uni.load_operator raises FileNotFoundError from inside the traced iteration: every certified
    box_hyperplane component except the three hand-written ones is silently unusable, and the failure surfaces as a
    solver crash rather than as a missing component. Only the basename is portable, so that is what is rewritten;
    the copy keeps everything else, including calibrations, byte for byte.
    """
    import json as _json

    n_fixed = 0
    with open(src) as fh, open(dst, "w") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = _json.loads(line)
            f = (e.get("payload") or {}).get("file")
            if e.get("kind") == "component" and f:
                local = os.path.join(components_dir, os.path.basename(f))
                if os.path.exists(local) and local != f:
                    e["payload"]["file"] = local
                    n_fixed += 1
            out.write(_json.dumps(e) + "\n")
    return n_fixed


def reference_obj(d, timeout_s=900.0, solvers=("highs", "clarabel")):
    """(q'x, record) from the first isolated baseline that returns a point, judged by uni.accept's own feasibility."""
    from atlas.bench.baseline_proc import run_baseline_proc
    from atlas.certify.residuals import qp_accuracy_judge

    tried = []
    for s in solvers:
        t0 = time.perf_counter()
        rb = run_baseline_proc(d, s, 1e-9, timeout_s=timeout_s, scratch=os.environ.get("OPTEVOLVE_SCRATCH"))
        rec = {"solver": s, "wall_s": round(time.perf_counter() - t0, 2)}
        if "error" in rb or rb.get("x") is None:
            rec["error"] = str(rb.get("error"))[:300]
            tried.append(rec)
            continue
        x = np.asarray(rb["x"])
        j = qp_accuracy_judge(x, d["P"], d["q"], d["r"], d["A"], d["l"], d["u"], ref_obj=None, feas_tol=1e-6)
        rec.update(status=rb.get("status"), t_s=rb.get("t_s"), row_violation=float(j["row_violation"]),
                   obj=float(np.asarray(d["q"]) @ x))
        tried.append(rec)
        if np.isfinite(rec["obj"]) and rec["row_violation"] <= 1e-5:
            return rec["obj"], {"chosen": s, "tried": tried}
    return None, {"chosen": None, "tried": tried}


def classify(res):
    """The five gap types of the study, as non-exclusive flags plus one primary bucket in the study's own order.

    They are not disjoint on real data: an instance can route to a CPU engine AND carry an operator-cost spec saying
    which projection budget would move it onto the GPU, which is exactly the interesting case.
    """
    fb = res.get("feedback") or {}
    choice = res.get("choice") or {}
    form = (choice.get("formulation") or "") if choice else ""
    binding = fb.get("binding") or ""
    flags = {
        "a_cpu_engine": bool(res.get("solved") and form.startswith("engine:")),
        "b_relocated": bool(res.get("solved") and form.startswith("reloc")),
        "c_operator_cost_t": "operator cost t" in binding,
        "d_iteration_count_N": (not res.get("solved")) or binding.startswith("nothing reached the target"),
        "e_uncovered": bool(fb.get("uncovered")),
    }
    for k in ("a_cpu_engine", "b_relocated", "c_operator_cost_t", "d_iteration_count_N", "e_uncovered"):
        if flags[k]:
            return k, flags
    return "f_no_diagnosis", flags


def run_one(name, vault_path, device="h100", tol=1e-3, budget_s=60.0, ref_timeout=900.0, record_evidence=True):
    import miplib
    import router
    from vault import Vault

    rec = {"name": name, "case": f"MIPLIB:{name}", "device": device, "tol": tol, "budget_s": budget_s}
    t0 = time.perf_counter()
    try:
        d = miplib.load_miplib(name)
    except Exception as e:
        rec["load_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return rec
    A = sp.csc_matrix(d["A"])
    rec.update(n=int(A.shape[1]), m=int(A.shape[0]), nnz=int(A.nnz), load_s=round(time.perf_counter() - t0, 2),
               n_integer_relaxed=d["meta"]["n_integer_relaxed"])
    obj_ref, ref = reference_obj(d, timeout_s=ref_timeout)
    rec["reference"] = ref
    if obj_ref is None:
        rec["error"] = "no usable reference"
        return rec
    rec["obj_ref"] = obj_ref
    try:
        import jax
        rec["jax_device"] = str(jax.devices()[0])
    except Exception as e:
        rec["jax_device"] = f"error: {e}"
    t0 = time.perf_counter()
    try:
        res = router.solve(rec["case"], Vault(vault_path), device, tol=tol, budget_s=budget_s,
                           record_evidence=record_evidence, d=d, obj_ref=obj_ref)
    except Exception as e:
        import traceback
        rec["solve_error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["traceback"] = traceback.format_exc()[-1200:]
        rec["solve_wall_s"] = round(time.perf_counter() - t0, 2)
        return rec
    rec["solve_wall_s"] = round(time.perf_counter() - t0, 2)
    for k in ("solved", "first_solution_s", "time_to_solution_s", "choice", "explored", "table", "feedback",
              "wall_total_s"):
        rec[k] = res.get(k)
    rec["gap_class"], rec["gap_flags"] = classify(res)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--vault", required=True)
    ap.add_argument("--device", default="h100")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--ref-timeout", type=float, default=900.0)
    ap.add_argument("--proc-timeout", type=float, default=1800.0)
    ap.add_argument("--prepare-vault-from", default="",
                    help="copy this vault to --vault, repointing component files at --components-dir")
    ap.add_argument("--components-dir", default="/workspace/optevolve/vault/components")
    args = ap.parse_args()

    if args.prepare_vault_from:
        n = prepare_vault(args.prepare_vault_from, args.vault, args.components_dir)
        print(f"vault copied to {args.vault}: {n} component file paths repointed at {args.components_dir}")
        if not args.names and not args.one:
            return

    if args.one:
        rec = run_one(args.one, args.vault, args.device, args.tol, args.budget, args.ref_timeout)
        print("@@REC@@" + json.dumps(rec, default=str), flush=True)
        return

    names = [x for x in args.names.split(",") if x]
    script = os.path.abspath(__file__)
    with open(args.out, "w") as fh:
        for i, name in enumerate(names, 1):
            cmd = [sys.executable, script, "--one", name, "--vault", args.vault, "--device", args.device,
                   "--tol", repr(args.tol), "--budget", repr(args.budget), "--ref-timeout", repr(args.ref_timeout)]
            t0 = time.perf_counter()
            try:
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=args.proc_timeout,
                                    env=dict(os.environ))
                line = next((l for l in reversed(pr.stdout.splitlines()) if l.startswith("@@REC@@")), None)
                if line is None:
                    rec = {"name": name, "driver_error": "no record emitted", "returncode": pr.returncode,
                           "stderr": (pr.stderr or "")[-1500:], "stdout_tail": (pr.stdout or "")[-800:]}
                else:
                    rec = json.loads(line[len("@@REC@@"):])
            except subprocess.TimeoutExpired:
                rec = {"name": name, "driver_error": f"process timeout after {args.proc_timeout}s"}
            rec["driver_wall_s"] = round(time.perf_counter() - t0, 2)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[{i}/{len(names)}] {name} class={rec.get('gap_class')} solved={rec.get('solved')} "
                  f"choice={(rec.get('choice') or {}).get('formulation')} {rec['driver_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
