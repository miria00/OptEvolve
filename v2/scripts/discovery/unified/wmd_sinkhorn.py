"""Entropic (Sinkhorn) approximation of WMD, batched over document pairs, for the accuracy comparison.

WHY THIS FILE EXISTS. The point of the comparison is "does the cheap approximation give the same downstream answer",
and answering it needs the SAME test-by-train distance matrix the exact run produces: 1,000 x 10,970 pairs here.
A straightforward per-pair numpy Sinkhorn costs about 8 s per query row at 100 iterations, i.e. days for one
regularization setting. The whole cost is elementwise work on tiny m x n arrays, so it batches essentially perfectly;
this file does that.

METHOD. Log-domain Sinkhorn, not the u/v scaling form: the kernel exp(-C/eps) underflows to zero in float32 as soon
as eps falls below about cost_max/80, which is exactly the regime where the entropic distance approaches the LP
optimum, and the scaling form then returns NaN. The log-sum-exp form has no such limit.

    f <- eps (log a - LSE_j (-C_ij/eps + g_j/eps))
    g <- eps (log b - LSE_i (-C_ij/eps + f_i/eps))

BUCKETING. Documents are padded to a power-of-two length and grouped, so one jitted call handles every column
document of the same padded length at once and only a handful of shapes are ever compiled. Padded words carry
log-weight -BIG, so their rows and columns contribute exp(-BIG) = 0 to every reduction and 0 to the returned cost:
the padding changes no number, it only changes the shape.

REPORTED QUANTITY. The transport cost <C, Pi> of the entropic plan, NOT the regularized objective. That is what a
practitioner substitutes for WMD, and it is the quantity that can be compared with the LP optimum on equal terms.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import wmd

BIG = 1e20                      # log-weight of a padded word; exp(-BIG/eps) is 0 in every reachable eps
BUCKETS = (8, 16, 32, 64, 128, 256, 512)


def _bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"document of {n} words exceeds the largest bucket {BUCKETS[-1]}")


def _pack(docs, V):
    """Group documents by padded length. Returns {padded_len: (indices, embeddings, log weights)}."""
    by = {}
    for pos, i in enumerate(docs):
        ids, w = wmd.doc_nbow(int(i))
        by.setdefault(_bucket(ids.size), []).append((pos, ids, w))
    out = {}
    for L, items in by.items():
        B = len(items)
        E = np.zeros((B, L, V.shape[1]), dtype=np.float32)
        lw = np.full((B, L), -BIG, dtype=np.float32)
        pos = np.empty(B, dtype=np.int64)
        for k, (p, ids, w) in enumerate(items):
            E[k, :ids.size] = V[ids]
            lw[k, :ids.size] = np.log(np.maximum(w, 1e-300))
            pos[k] = p
        out[L] = (pos, E, lw)
    return out


def sinkhorn_matrix(rows, cols, p=1.0, eps_rel=0.1, n_iter=200, mem_fraction=None):
    """Entropic transport cost for every (row document, column document) pair. Shape (len(rows), len(cols)).

    eps is set per pair as eps_rel * mean(C) so one setting is comparable across pairs with different vocabulary.
    """
    if mem_fraction:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    import jax
    import jax.numpy as jnp

    V = wmd.load()["V"]
    packed = _pack(np.asarray(cols), V)
    D = np.zeros((len(rows), len(cols)), dtype=np.float64)
    # the column side is the SAME for every query row, so it is moved to the device once: transferring it inside the
    # loop costs about 1 GB of host-to-device traffic per query row and dominates everything else


    @jax.jit
    def run(A, la, E, lb):
        """A (M,300) query words, la (M,) log weights; E (B,N,300) column words, lb (B,N) log weights."""
        d2 = ((A * A).sum(-1)[None, :, None] + (E * E).sum(-1)[:, None, :]
              - 2.0 * jnp.einsum("md,bnd->bmn", A, E))
        C = jnp.sqrt(jnp.maximum(d2, 0.0))
        if p != 1.0:
            C = C ** p
        live = (la[None, :, None] > -BIG / 2) & (lb[:, None, :] > -BIG / 2)
        C = jnp.where(live, C, 0.0)
        denom = jnp.maximum(live.sum(axis=(1, 2)), 1)
        eps = jnp.maximum(eps_rel * C.sum(axis=(1, 2)) / denom, 1e-12)      # eps_rel * mean over REAL entries
        M = -C / eps[:, None, None]
        M = jnp.where(live, M, -BIG)

        def body(_, fg):
            f, g = fg
            f = eps[:, None] * (la[None, :] - jax.nn.logsumexp(M + (g / eps[:, None])[:, None, :], axis=2))
            g = eps[:, None] * (lb - jax.nn.logsumexp(M + (f / eps[:, None])[:, :, None], axis=1))
            return f, g

        B, Mn, Nn = M.shape
        f, g = jax.lax.fori_loop(0, n_iter, body, (jnp.zeros((B, Mn)), jnp.zeros((B, Nn))))
        Pi = jnp.exp(M + (f / eps[:, None])[:, :, None] + (g / eps[:, None])[:, None, :])
        Pi = jnp.where(live, Pi, 0.0)
        return (Pi * C).sum(axis=(1, 2)), Pi.sum(axis=(1, 2))

    packed_dev = {L: (pos, jnp.asarray(E), jnp.asarray(lb)) for L, (pos, E, lb) in packed.items()}
    t0 = time.perf_counter()
    mass_min = 1.0
    for r, i in enumerate(rows):
        ids_a, a = wmd.doc_nbow(int(i))
        L = _bucket(ids_a.size)
        A = np.zeros((L, V.shape[1]), dtype=np.float32)
        A[:ids_a.size] = V[ids_a]
        la = np.full(L, -BIG, dtype=np.float32)
        la[:ids_a.size] = np.log(np.maximum(a, 1e-300))
        Aj, laj = jnp.asarray(A), jnp.asarray(la)
        for _, (pos, E, lb) in packed_dev.items():
            cost, mass = run(Aj, laj, E, lb)
            D[r, pos] = np.asarray(cost, dtype=np.float64)
            mass_min = min(mass_min, float(np.asarray(mass).min()))
    return D, time.perf_counter() - t0, mass_min


def main():
    """CLI: sinkhorn ROWS.npy COLS.npy OUT.npy EPS_REL N_ITER [p]"""
    import json
    rows = np.load(sys.argv[1])
    cols = np.load(sys.argv[2])
    out = sys.argv[3]
    eps_rel, n_iter = float(sys.argv[4]), int(sys.argv[5])
    pp = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    D, dt, mass = sinkhorn_matrix(rows, cols, pp, eps_rel, n_iter)
    np.save(out, D)
    print(json.dumps({"done": True, "eps_rel": eps_rel, "n_iter": n_iter, "p": pp, "wall_s": dt,
                      "pairs": int(rows.size * cols.size), "min_plan_mass": mass,
                      "us_per_pair": 1e6 * dt / max(rows.size * cols.size, 1), "out": out}), flush=True)


if __name__ == "__main__":
    main()
