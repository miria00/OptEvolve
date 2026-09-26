"""Exact network-simplex reference for WMD transport instances, run in POT's OWN venv.

POT must not be relied on from any shared environment, so this file is executed by OPTEVOLVE_POT_PYTHON
(default /home/miria/baseline_envs/venv_pot/bin/python, /root/baseline_envs/venv_pot/bin/python on the H100) as a
subprocess and prints JSON lines. It imports wmd.py, which needs only numpy and scipy, never jax.

Network simplex is exact in the sense that matters for a reference: it terminates at a VERTEX of the transportation
polytope with an optimality certificate, so its objective is not a tolerance-limited number the way an
interior-point or first-order reference is. It is also a completely independent implementation of the reduction
(POT builds its own graph from (a, b, C)), which is what makes it a check on wmd.make_wmd rather than a restatement
of it.

Subcommands
  pair I J [p]                          one instance: objective, plan feasibility, timing
  spec SPEC [p]                         any run_wmd instance spec ('pair:I:J' or 'concat:TARGET_WORDS'), which is
                                        how the large concatenated instances get an exact reference objective
  matrix ROWS.npy COLS.npy OUT.npy [p]  the full WMD distance matrix between two sets of documents (the k-NN
                                        workload: one LP per entry, and the count is the headline)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import wmd


def solve_pair(i, j, p=1.0, log=True):
    ids_a, a = wmd.doc_nbow(i)
    ids_b, b = wmd.doc_nbow(j)
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b * (a.sum() / b.sum()))      # POT rejects marginals whose sums differ
    C = np.ascontiguousarray(wmd.cost_matrix(ids_a, ids_b, p))
    import ot
    t0 = time.perf_counter()
    if log:
        G, lg = ot.emd(a, b, C, numItermax=10_000_000, log=True)
        val = float((G * C).sum())
        warn = lg.get("warning")
    else:
        val = float(ot.emd2(a, b, C, numItermax=10_000_000))
        G, warn = None, None
    dt = time.perf_counter() - t0
    rec = {"solver": "pot_network_simplex", "pot_version": ot.__version__, "doc_a": int(i), "doc_b": int(j),
           "m": int(a.size), "n": int(b.size), "n_var": int(a.size * b.size), "obj": val,
           "t_solve_s": dt, "warning": warn}
    if G is not None:
        # feasibility of the transport plan itself, in the metric uni.accept uses on the LP's rows
        rec["plan_row_err"] = float(np.max(np.abs(G.sum(1) - a)))
        rec["plan_col_err"] = float(np.max(np.abs(G.sum(0) - b)))
        rec["plan_min"] = float(G.min())
        rec["plan_mass"] = float(G.sum())
    return rec


def solve_spec(spec, p=1.0):
    """Exact optimum of any run_wmd instance spec. Marginals are rebuilt the same deterministic way, so the LP this
    references is the LP run_wmd builds."""
    parts = spec.split(":")
    if parts[0] == "pair":
        ids_a, a = wmd.doc_nbow(int(parts[1]))
        ids_b, b = wmd.doc_nbow(int(parts[2]))
    elif parts[0] == "concat":
        g = wmd.concat_groups(int(parts[1]), n_groups=2, seed=0)
        ids_a, a = wmd.concat_nbow(g[0])
        ids_b, b = wmd.concat_nbow(g[1])
    else:
        raise ValueError(f"bad spec {spec!r}")
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b * (a.sum() / b.sum()))
    C = np.ascontiguousarray(wmd.cost_matrix(ids_a, ids_b, p))
    import ot
    t0 = time.perf_counter()
    val, lg = ot.emd2(a, b, C, numItermax=100_000_000, log=True,
                      numThreads=int(os.environ.get("OMP_NUM_THREADS", 1)))
    return {"solver": "pot_network_simplex", "pot_version": ot.__version__, "label": spec, "obj_ref": float(val),
            "m": int(a.size), "n": int(b.size), "n_var": int(a.size * b.size),
            "t_solve_s": time.perf_counter() - t0, "warning": lg.get("warning")}


def main():
    cmd = sys.argv[1]
    if cmd == "spec":
        print(json.dumps(solve_spec(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)))
        return
    if cmd == "pair":
        i, j = int(sys.argv[2]), int(sys.argv[3])
        p = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        print(json.dumps(solve_pair(i, j, p)))
        return
    if cmd == "pairs":
        # exactly the listed pairs, not a cross product: this is the batch workload's true cost
        P = np.load(sys.argv[2])
        p = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        import ot
        V = wmd.load()["V"]
        vals = np.zeros(P.shape[0])
        t0 = time.perf_counter()
        for k, (i, j) in enumerate(P):
            ids_a, a = wmd.doc_nbow(int(i))
            ids_b, b = wmd.doc_nbow(int(j))
            A, B = V[ids_a].astype(np.float64), V[ids_b].astype(np.float64)
            d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
            np.maximum(d2, 0.0, out=d2)
            C = np.sqrt(d2)
            if p != 1.0:
                C = C ** p
            a = np.ascontiguousarray(a)
            vals[k] = ot.emd2(a, np.ascontiguousarray(b * (a.sum() / b.sum())), np.ascontiguousarray(C),
                              numItermax=10_000_000)
        dt = time.perf_counter() - t0
        print(json.dumps({"done": True, "n_pairs": int(P.shape[0]), "wall_s": dt,
                          "us_per_lp": 1e6 * dt / max(P.shape[0], 1), "pot_version": ot.__version__,
                          "obj_mean": float(vals.mean())}), flush=True)
        return
    if cmd == "matrix":
        rows = np.load(sys.argv[2])
        cols = np.load(sys.argv[3])
        out_path = sys.argv[4]
        p = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        import ot
        nb = [wmd.doc_nbow(int(j)) for j in cols]
        Cs = [np.ascontiguousarray(wmd.load()["V"][ids].astype(np.float64)) for ids, _ in nb]
        D = np.zeros((rows.size, cols.size))
        t0 = time.perf_counter()
        solved = 0
        for r, i in enumerate(rows):
            ids_a, a = wmd.doc_nbow(int(i))
            a = np.ascontiguousarray(a)
            A = wmd.load()["V"][ids_a].astype(np.float64)
            aa = (A * A).sum(1)
            for c, (ids_b, b) in enumerate(nb):
                B = Cs[c]
                d2 = aa[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
                np.maximum(d2, 0.0, out=d2)
                C = np.sqrt(d2)
                if p != 1.0:
                    C = C ** p
                bb = np.ascontiguousarray(b * (a.sum() / b.sum()))
                D[r, c] = ot.emd2(a, np.ascontiguousarray(bb), np.ascontiguousarray(C), numItermax=10_000_000)
                solved += 1
            if (r + 1) % 25 == 0:
                print(json.dumps({"progress": int(r + 1), "of": int(rows.size), "lps": solved,
                                  "elapsed_s": time.perf_counter() - t0}), flush=True)
        np.save(out_path, D)
        print(json.dumps({"done": True, "lps_solved": solved, "wall_s": time.perf_counter() - t0,
                          "pot_version": ot.__version__, "out": out_path,
                          "us_per_lp": 1e6 * (time.perf_counter() - t0) / max(solved, 1)}), flush=True)
        return
    raise SystemExit(f"unknown subcommand {cmd!r}")


if __name__ == "__main__":
    main()
