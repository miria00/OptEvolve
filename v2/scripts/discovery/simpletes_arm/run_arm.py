"""Launch SimpleTES (arXiv 2604.19341, pinned clone in ./SimpleTES) as a program-search arm on the exact projection
onto {lo <= x <= hi, a'x = b}, at a budget of N LLM-generated candidates so it can be compared with the SkyDiscover arms.

Budget: --candidates N is SimpleTES's --max-generations, the number of generation requests. A candidate that fails code
extraction still consumes one unit (SimpleTES does not retry), so N equals the number of generation LLM calls. N must
be a multiple of chains * k, so the allocation is exact: L = N / (chains * k) refinement steps per trajectory.
Reflection is OFF by default because each reflection is an extra LLM call per committed step outside N; turn it on with
--reflection on and count the calls from llm_calls.jsonl. The seed is evaluated once and does not count toward N.

Examples (run with any Python; the script re-executes itself under /home/miria/simpletesenv):
  CPU plumbing check:  run_arm.py --eval stub --candidates 3 --chains 1 --k 1 --no-thinking --max-tokens 6000
  in-solver arm:       run_arm.py --eval insolver --spec spec.json --device "RTX 4090" --candidates 96
  end-to-end arm:      run_arm.py --eval blackbox_multi --e2e-cases A,B,C --e2e-cache cache.json --candidates 96
  re-summarize a run:  run_arm.py --summarize runs/<run_dir>
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIMPLETES_ROOT = os.path.join(HERE, "SimpleTES")
UNIFIED = os.path.normpath(os.path.join(HERE, "..", "unified"))
ARM_PY = "/home/miria/simpletesenv/bin/python"
DEFAULT_LOCK = "/tmp/claude-1000/opcheck.lock"  # the lock the SkyDiscover evaluators share, so timings never overlap


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarize", metavar="RUN_DIR", help="only (re)write summary.json for an existing run")
    p.add_argument("--eval", choices=("blackbox_multi", "insolver", "stub"), help="evaluator semantics")
    p.add_argument("--candidates", type=int, help="N: LLM-generated candidates (generation requests)")
    p.add_argument("--chains", type=int, default=4, help="C: independent trajectories")
    p.add_argument("--k", type=int, default=4, help="K: candidates per prompt, best one committed")
    p.add_argument("--selector", default="rpucg", help="SimpleTES history policy (paper default: rpucg)")
    p.add_argument("--num-inspirations", type=int, default=5, help="history nodes per prompt (SimpleTES default 5)")
    p.add_argument("--reflection", choices=("off", "on"), default="off")
    p.add_argument("--seed-program", default=os.path.join(UNIFIED, "seed_bisect.py"))
    p.add_argument("--instruction", help="instruction file; default depends on --eval (see build_instruction)")
    p.add_argument("--spec", help="insolver: router spec JSON (case, formulation, per_iteration_budget_us, n_x, ...)")
    p.add_argument("--device", default="the local GPU", help="insolver: device name shown in the prompt")
    p.add_argument("--e2e-cases", help="blackbox_multi: comma-separated suite (else inherits E2E_CASES)")
    p.add_argument("--e2e-cache", help="blackbox_multi: standard-formulation timing cache (else E2E_CACHE)")
    p.add_argument("--e2e-tol", help="blackbox_multi: acceptance tolerance (else E2E_TOL or 1e-3)")
    p.add_argument("--lock", default=os.environ.get("SPEC_LOCK", DEFAULT_LOCK))
    p.add_argument("--failure-as-error", choices=("on", "off"), default="on")
    p.add_argument("--marker-fallback", choices=("on", "off"), default="on")
    # LLM: defaults mirror the SkyDiscover arm config written by unified/evolve.py.
    p.add_argument("--api-base", default="http://127.0.0.1:8002/v1")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--llm-timeout", type=float, default=2400)
    p.add_argument("--no-thinking", action="store_true", help="disable Qwen thinking (cheap dry runs only)")
    p.add_argument("--gen-concurrency", type=int, default=8)
    p.add_argument("--eval-concurrency", type=int, default=1)
    p.add_argument("--eval-timeout", type=float, default=3600)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--run-root", default=os.path.join(HERE, "runs"))
    p.add_argument("--tag", default="")
    p.add_argument("--print-only", action="store_true", help="prepare the run dir and print the command, do not run")
    args = p.parse_args(argv)
    if args.summarize:
        return args
    if not args.eval or not args.candidates:
        p.error("--eval and --candidates are required")
    per_round = args.chains * args.k
    if args.candidates % per_round:
        lo = (args.candidates // per_round) * per_round
        p.error(f"--candidates must be a multiple of chains*k = {per_round} so the budget is exact "
                f"(nearest: {lo} or {lo + per_round})")
    if args.eval == "insolver" and not args.spec:
        p.error("--eval insolver needs --spec")
    return args


def _prompt_constant(name):
    # Parse, do not import: evolve.py pulls in the vault and jaxenv2 dependencies. Parsing keeps the SkyDiscover arm's
    # prompt text byte-identical without them.
    tree = ast.parse(open(os.path.join(UNIFIED, "evolve.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def build_instruction(args, spec):
    if args.instruction:
        base = open(args.instruction).read().rstrip()
    elif args.eval == "insolver":
        if spec.get("set", "box_hyperplane") != "box_hyperplane":
            raise SystemExit("only the box_hyperplane set is wired (seed and certificate have that signature)")
        base = _prompt_constant("PROMPT").format(
            device=args.device, case=spec["case"], n_x=spec.get("n_x", "?"),
            budget_us=spec["per_iteration_budget_us"], current_us=spec.get("current_per_iteration_us", 0.0)).rstrip()
    else:
        base = open(os.path.join(HERE, "instruction_blackbox_multi.txt")).read().rstrip()
    # SimpleTES's task guide asks every instruction to name the evolved region and the per-evaluation time limit.
    return (f"{base}\n\nDo this by evolving the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.\n"
            f"The time limit for each program evaluation is {int(args.eval_timeout)} seconds.\n")


def arm_environment(args, run_dir, spec):
    env = {
        "SIMPLETES_ARM_EVAL": args.eval,
        "SIMPLETES_ARM_FAILURE_AS_ERROR": "1" if args.failure_as_error == "on" else "0",
        "SIMPLETES_ARM_MARKER_FALLBACK": "1" if args.marker_fallback == "on" else "0",
        "SIMPLETES_ARM_TOKEN_LOG": os.path.join(run_dir, "llm_calls.jsonl"),
        "SIMPLETES_ARM_TOP_P": repr(args.top_p),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "SPEC_LOCK": args.lock,
    }
    if args.no_thinking:
        env["SIMPLETES_ARM_EXTRA_BODY"] = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
    if args.eval == "insolver":
        env.update({"SPEC_CASE": spec["case"], "SPEC_FORMULATION": spec["formulation"],
                    "SPEC_BUDGET_PER_ITER": str(spec["per_iteration_budget_us"] * 1e-6),
                    "CERT_SCRIPT": "opcheck.py", "SPEC_PLAN": "single"})
    if args.eval == "blackbox_multi":
        for flag, key in ((args.e2e_cases, "E2E_CASES"), (args.e2e_cache, "E2E_CACHE"), (args.e2e_tol, "E2E_TOL")):
            if flag:
                env[key] = flag
            elif os.environ.get(key):
                env[key] = os.environ[key]
        missing = [k for k in ("E2E_CASES", "E2E_CACHE") if not env.get(k)]
        if missing:
            raise SystemExit(f"--eval blackbox_multi needs {missing} (flags or environment)")
    return env


def simpletes_argv(args, run_dir):
    per_round = args.chains * args.k
    argv = [
        "--init-program", os.path.join(run_dir, "init_program.py"),
        "--evaluator", os.path.join(HERE, "evaluator.py"),
        "--instruction", os.path.join(run_dir, "instruction.txt"),
        "--model", "openai/" + args.model, "--api-base", args.api_base,
        "--selector", args.selector, "--num-chains", str(args.chains), "--k-candidates", str(args.k),
        "--max-generations", str(args.candidates), "--num-inspirations", str(args.num_inspirations),
        "--temperature", str(args.temperature), "--max-tokens", str(args.max_tokens),
        "--timeout", str(args.llm_timeout), "--retry", "0",
        "--gen-concurrency", str(args.gen_concurrency), "--eval-concurrency", str(args.eval_concurrency),
        "--eval-timeout", str(args.eval_timeout), "--init-eval-repeats", "1",
        "--log-interval", str(per_round), "--db-show-interval", str(per_round),
        "--output-path", os.path.join(run_dir, "simpletes"), "--save-llm-io",
    ]
    if args.reflection == "off":
        argv.append("--disable-reflection")
    if args.skip_preflight:
        argv.append("--skip-preflight")
    return argv


def _git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def summarize(run_dir):
    states = sorted(glob.glob(os.path.join(run_dir, "simpletes", "*", "instance-*", "db_state_*")), key=os.path.getmtime)
    out = {"run_dir": run_dir, "checkpoint": states[-1] if states else None}
    if states:
        meta = json.load(open(os.path.join(states[-1], "metadata.json")))
        nodes = json.load(open(os.path.join(states[-1], "nodes.json")))
        out.update({k: meta.get(k) for k in ("best_score", "completed_evaluations", "generation_attempts",
                                              "generation_failures", "evaluation_failures")})
        ranked = sorted((n for n in nodes if n.get("score") is not None and math.isfinite(n["score"])),
                        key=lambda n: -n["score"])
        top_dir = os.path.join(run_dir, "top")
        os.makedirs(top_dir, exist_ok=True)
        out["top"] = []
        for rank, n in enumerate(ranked[:3]):
            path = os.path.join(top_dir, f"rank{rank}.py")
            open(path, "w").write(n["code"])
            out["top"].append({"file": path, "score": n["score"], "is_seed": not n.get("parent_ids"),
                               "chain": n.get("chain_idx"), "feedback": (n.get("metrics") or {}).get("feedback")})
        out["nodes"] = len(nodes)
        failures = os.path.join(states[-1], "failure.json")
        out["failure_records"] = len(json.load(open(failures))) if os.path.exists(failures) else 0
    calls = []
    log = os.path.join(run_dir, "llm_calls.jsonl")
    if os.path.exists(log):
        calls = [json.loads(l) for l in open(log) if l.strip()]
    by_kind = {}
    for c in calls:
        agg = by_kind.setdefault(c["kind"], {"calls": 0, "ok": 0, "prompt_tokens": 0, "completion_tokens": 0})
        agg["calls"] += 1
        agg["ok"] += int(bool(c.get("ok")))
        agg["prompt_tokens"] += int(c.get("prompt_tokens") or 0)
        agg["completion_tokens"] += int(c.get("completion_tokens") or 0)
    out["llm"] = by_kind
    json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    return out


def main(argv):
    args = parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize(os.path.abspath(args.summarize)), indent=2))
        return
    spec = json.load(open(args.spec)) if args.spec else {}
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{args.eval}_N{args.candidates}_C{args.chains}_K{args.k}" + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(os.path.abspath(args.run_root), name)
    os.makedirs(run_dir, exist_ok=True)

    seed_text = open(args.seed_program).read()
    if "# EVOLVE-BLOCK-START" not in seed_text or "# EVOLVE-BLOCK-END" not in seed_text:
        raise SystemExit(f"seed program lacks EVOLVE-BLOCK markers: {args.seed_program}")
    shutil.copy(args.seed_program, os.path.join(run_dir, "init_program.py"))
    open(os.path.join(run_dir, "instruction.txt"), "w").write(build_instruction(args, spec))
    env = arm_environment(args, run_dir, spec)
    argv_st = simpletes_argv(args, run_dir)
    config = {"args": vars(args), "spec": spec, "env": env, "simpletes_argv": argv_st,
              "simpletes_commit": _git_head(SIMPLETES_ROOT), "python": sys.executable,
              "allocation": {"N": args.candidates, "C": args.chains, "K": args.k,
                             "L": args.candidates // (args.chains * args.k)}}
    json.dump(config, open(os.path.join(run_dir, "arm_config.json"), "w"), indent=2)
    print(f"[simpletes-arm] run dir {run_dir}")
    print(f"[simpletes-arm] allocation {config['allocation']}")
    if args.print_only:
        print("[simpletes-arm] env: " + " ".join(f"{k}={v!r}" for k, v in env.items()))
        print("[simpletes-arm] simpletes main.py " + " ".join(argv_st))
        return

    os.environ.update(env)
    # The SkyDiscover evaluators strip this themselves; stripping it here also keeps the CPU stub off a stale CUDA path.
    os.environ.pop("LD_LIBRARY_PATH", None)
    random.seed(args.rng_seed)
    sys.path.insert(0, SIMPLETES_ROOT)
    import arm_patches
    arm_patches.install()
    spec_main = importlib.util.spec_from_file_location("simpletes_main", os.path.join(SIMPLETES_ROOT, "main.py"))
    st_main = importlib.util.module_from_spec(spec_main)
    spec_main.loader.exec_module(st_main)
    sys.argv = ["main.py"] + argv_st
    t0 = time.time()
    try:
        st_main.main()
    finally:
        summary = summarize(run_dir)
        summary["wall_s"] = round(time.time() - t0, 1)
        json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
        print("[simpletes-arm] summary " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    if os.path.realpath(sys.executable) != os.path.realpath(ARM_PY) and os.path.exists(ARM_PY):
        os.execv(ARM_PY, [ARM_PY, os.path.abspath(__file__)] + sys.argv[1:])
    main(sys.argv[1:])
