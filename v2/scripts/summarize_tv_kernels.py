"""Training-selected native/XLA controls and synchronized Nsight range summaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.bench.factorial import score, select_catalog


def trace_ranges(path):
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    out = {}
    try:
        ranges = connection.execute("select start,end,text from NVTX_EVENTS where text like 'pdhg/%' or text like 'condat_vu/%'").fetchall()
        for start, end, name in ranges:
            rows = connection.execute('''select s.value,count(*),sum(k.end-k.start)
                from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id
                where k.start>=? and k.end<=? group by s.value order by sum(k.end-k.start) desc''', (start, end)).fetchall()
            total = sum(r[2] for r in rows)
            out[name] = {"range_ns": end-start, "kernel_count": sum(r[1] for r in rows),
                "kernel_time_ns": total, "kernels": [{"name": n,"count": c,"total_ns": t,"share": t/total}
                                                    for n,c,t in rows]}
    finally:
        connection.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, protocols, inputs = [], [], []
    for path in map(Path, args.runs):
        protocol = json.loads((path/"protocol.json").read_text())
        result = json.loads((path/"factorial.json").read_text())
        data = json.loads((path/"data.json").read_text())
        if result["status"] != "complete":
            raise ValueError(f"unfinished catalog: {path}")
        protocols.append(protocol)
        inputs.append(data)
        summary = {"run": str(path), "device": protocol["device"],
                   "catalog_s": result["campaign_s"], "tolerances": {}}
        for tol, block in result["tolerances"].items():
            choices = select_catalog(block["rows"], *protocol["baseline"])
            assert choices == {k:tuple(v) for k,v in block["selections"].items()}
            by_key = {(r["algorithm"],r["policy"]):r for r in block["rows"]}
            per_algorithm = {}
            for alg in protocol["algorithms"]:
                rows = [r for r in block["rows"] if r["algorithm"] == alg]
                xla = min((r for r in rows if r["genome"]["backend"].get("kernel", "jax") == "jax"),
                          key=lambda r:(score(r["train"]),r["policy"]))
                native = min((r for r in rows if r["genome"]["backend"].get("kernel", "jax") != "jax"),
                             key=lambda r:(score(r["train"]),r["policy"]))
                cadence = native["genome"]["backend"]["K"]
                matched = by_key[alg,f"f64_K{cadence}"]
                per_algorithm[alg] = {"xla_policy": xla["policy"], "native_policy": native["policy"],
                    "xla": xla["validation"], "native": native["validation"],
                    "matched_f64": matched["validation"],
                    "xla_over_native": xla["validation"]["penalized_sgm1_s"] / native["validation"]["penalized_sgm1_s"],
                    "matched_f64_over_native": matched["validation"]["penalized_sgm1_s"] / native["validation"]["penalized_sgm1_s"]}
            summary["tolerances"][tol] = {"algorithms": per_algorithm,
                "selections": {arm:{"recipe": list(key), "validation":by_key[key]["validation"]}
                               for arm,key in choices.items()}}
        summaries.append(summary)
    for p,d in zip(protocols[1:],inputs[1:]):
        for key in ("source_sha256","algorithms","policies","baseline","tolerances"):
            assert p[key] == protocols[0][key], f"protocol mismatch: {key}"
        assert {k:v for k,v in p["args"].items() if k != "out"} == {k:v for k,v in protocols[0]["args"].items() if k != "out"}
        assert d == inputs[0], "input or partition mismatch"
        assert p["jax"] == protocols[0]["jax"]
    (out/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    lines = ["**TV FFI kernel comparison**", "",
        "Selections use training measurements only. Six training and six development-validation OCT images, 256x256, three warm repetitions, 5,000-iteration cap.",
        "The XLA control can choose f64/K64, f64/K256 or refinement/K256. Native choices are three FP64 stencil variants at K64 or K256.",
        "A fixed 120-second penalty is applied to non-solves. Scores containing failures are not actual mean solve times.",
        "All device catalogs have matching numerical source, inputs, candidate definitions, budgets and JAX versions.", "",
        "| Device | Tolerance | Algorithm | Selected XLA | Selected native | Solves XLA/native | XLA score (s) | Native score (s) | XLA/native | Same-K f64/native |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        for tol,b in s["tolerances"].items():
            for alg,v in b["algorithms"].items():
                lines.append(f"| {s['device']} | {tol} | {alg} | {v['xla_policy']} | {v['native_policy']} | "
                    f"{v['xla']['n_solved']}/{v['native']['n_solved']} of 6 | {v['xla']['penalized_sgm1_s']:.6f} | "
                    f"{v['native']['penalized_sgm1_s']:.6f} | {v['xla_over_native']:.3f}x | {v['matched_f64_over_native']:.3f}x |")
    lines += ["", "A ratio above 1 favors the native kernel; below 1 favors XLA.", "", "**Whole-catalog choices**", ""]
    for s in summaries:
        lines.append(f"- {s['device']}: catalog loop {s['catalog_s']:.1f} seconds, including warmups and repeats, excluding initial loading/profiling.")
        for tol,b in s["tolerances"].items():
            lines.append(f"  Tolerance {tol}: " + "; ".join(f"{arm} = {'/'.join(v['recipe'])}" for arm,v in b["selections"].items()) + ".")
    lines += ["", "**Limits**", "",
        "- This is a finite-catalog development diagnostic, not a budget-matched evolutionary search experiment or final holdout evaluation.",
        "- Existing GPU desktop/server allocations remain active; snapshots are in protocol.json. Hardware timing differences require confirmation under controlled allocations.",
        "- Native precision is FP64 only. The best-XLA comparison includes mixed precision; the matched comparison holds cadence and FP64 fixed.",
        "- Numerical admission and reference-trajectory agreement are not formal floating-point proofs. The reciprocal variant deliberately changes rounding.",
        "- Both schemes and every tested native variant remain visible in the raw catalogs, including failures and slower recipes.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))
    for path in args.traces:
        (out/(Path(path).stem+"_ranges.json")).write_text(json.dumps(trace_ranges(path),indent=2)+"\n")
    print(out/"REPORT.md")


if __name__ == "__main__":
    main()
