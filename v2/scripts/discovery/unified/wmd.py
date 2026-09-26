"""Word Mover's Distance on 20 Newsgroups as OSQP-form transport LPs (2026-09-15).

Kusner, Sun, Kolkin, Weinberger (ICML 2015) define the distance between two documents as the optimal transport cost
between their normalized bags of words, with the ground cost the Euclidean distance between the words' word2vec
embeddings. Every document PAIR is therefore an optimal-transport LP, and a k-NN classification run over 20
Newsgroups is a workload of MILLIONS of such LPs. That is the sense in which this text dataset is secretly an
optimization problem: nobody who runs WMD thinks of themselves as running an LP solver, but that is the whole cost.

With x = vec(Pi) in R^{mn}, m = unique words of document A, n = unique words of document B, the LP is

    min <c, x>   s.t.   sum_j x_ij = dA_i,   sum_i x_ij = dB_j,   x >= 0

and it lands in the pipeline's OSQP dict (P = 0, general rows then identity bound rows) exactly as
lp_families.transport and dotmark_ot.make_ot do, so uni.plan_product, uni.plan_block, uni.diagnose,
lp_families.lp_formulation and router.solve all read it unchanged. The row layout is the same convention:

    source rows   sum_j x_ij = dA_i       m rows, nnz n each      kron(I_m, 1_n^T)     row_family 0
    sink rows     sum_i x_ij = dB_j       n rows, nnz m each      kron(1_m^T, I_n)     row_family 1
    bound rows    x_ij >= 0               mn rows, nnz 1 each     I                    row_family 2

WHAT IS STRUCTURALLY NEW HERE, RELATIVE TO DOTMARK
--------------------------------------------------
DOTmark pairs two images at the SAME resolution, so m = n always and the two marginal families are
INDISTINGUISHABLE to uni.plan_product's greedy: it sorts absorbable rows by (-nnz, row index), every general row has
the same nnz, the sort degenerates to row index, and under a randomized encoding which family gets absorbed is a
coin flip.

A WMD pair is RECTANGULAR: m and n are the unique-word counts of two different documents and are equal only by
accident (measured below: 2.6% of sampled pairs). So the greedy is DECIDED, and decided by a rule worth naming:
source rows carry nnz = n and sink rows carry nnz = m, so sorting by -nnz absorbs the family belonging to the
SHORTER document, i.e. it takes FEWER, LARGER blocks (K = min(m, n) blocks of size max(m, n)). Both families are
absorbable and both cover every variable, so the choice is free and the greedy's tie-break is doing real work. The
rows-versus-columns experiment this makes possible is the point of `plan_side`, which is reused from dotmark_ot
because the row_family convention is identical.

GROUND COST. Euclidean, c_ij = ||v_i - v_j||_2, NOT squared: that is Kusner et al.'s definition and the one the
published k-NN number is computed with. `p` raises it to a power for conditioning studies; p = 1.0 is the default
and is what `knn` and the verification use.

PREPROCESSING, matching the paper (its Section 5.1) so the reproduced k-NN number is comparable:
  * "bydate" train/test split (11,314 / 7,532), which is what SetFit/20_newsgroups ships.
  * remove every word in the SMART stop word list (Salton and Buckley 1971). We use the copy in Kusner's OWN
    repository, mkusner/wmd/stop_words.txt, so the list is the one the published number used.
  * 20NEWS only: additionally drop words occurring fewer than 5 times across the corpus.
  * drop words with no word2vec embedding (the paper drops them for WMD and keeps them for the baselines).
  * cap each document at its 500 most frequent words (the paper does this to make WMD tractable at all, which is
    itself evidence for the track: the authors had to shrink the LPs to afford them).
  * weights are the normalized counts of the surviving words, so sum(dA) = sum(dB) = 1 and mass balance is exact.

SIZES. The paper's Table 1 reports 72 unique words per 20NEWS document on average, so the TYPICAL instance is about
72 x 72 = 5,184 variables. This module therefore also builds CONCATENATED super-documents (`make_wmd_concat`): the
bag of a set of documents from one newsgroup, summed and renormalized, which is a genuine nBOW of a genuine text and
reaches n = 1e5 .. 1e6 without inventing data. Every reported large instance says how many documents were merged.
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import numpy as np
import scipy.sparse as sp

# dotmark_ot's plan_side / plan_side_of / plans_agree / permute_ot / solver_dict read ONLY meta["row_family"] and the
# OSQP keys, so they are the shared side-selectable planner for any transport instance in this convention. Reusing
# them (rather than copying) is also the check that a WMD dict really is in the same convention.
import dotmark_ot
from dotmark_ot import permute_ot, plan_side, plan_side_of, plans_agree, solver_dict  # noqa: F401

def _data_root() -> str:
    """Where the prepared corpus lives. The H100 copy sits beside its own checkout, so this is not hard-coded."""
    r = os.environ.get("OPTEVOLVE_WMD_DATA")
    if r:
        return r
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "data", "wmd")
    for c in (here, "/workspace/optevolve/data/wmd", "/home/miria/OptEvolve/v2/data/wmd"):
        if os.path.isdir(c):
            return c
    return here


DATA = _data_root()
SNAPSHOT = os.environ.get(
    "OPTEVOLVE_20NEWS_SNAPSHOT",
    "/home/miria/.cache/huggingface/hub/datasets--SetFit--20_newsgroups/snapshots/"
    "f1b91292074e7cfb69be58b642d583ec262f30ed")
W2V_BIN = os.environ.get(
    "OPTEVOLVE_W2V_BIN",
    "/home/miria/.cache/huggingface/hub/models--NathaNn1111--word2vec-google-news-negative-300-bin/snapshots/"
    "78856d4586b3a938134c9833d92139f2e056e369/GoogleNews-vectors-negative300.bin")
CACHE = os.path.join(DATA, "corpus_cache.npz")
STOPWORDS_FILE = os.path.join(DATA, "stop_words.txt")

MIN_DF = 5              # 20NEWS-only rule from the paper
CAP_WORDS = 500         # the paper's per-document cap
_TOKEN = re.compile(r"[a-z]+")


# ----------------------------------------------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------------------------------------------
def stopwords() -> set:
    with open(STOPWORDS_FILE) as fh:
        return {w.strip() for w in fh if w.strip()}


def tokenize(text: str) -> list:
    """Lowercase alphabetic tokens of length >= 2. Digits and punctuation are dropped, as in the WMD pipeline."""
    return [w for w in _TOKEN.findall(text.lower()) if len(w) >= 2]


def read_split(split: str):
    texts, labels = [], []
    with open(os.path.join(SNAPSHOT, f"{split}.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["label"])
    return texts, np.asarray(labels, dtype=np.int32)


def _read_w2v(words_wanted: set, path: str = W2V_BIN):
    """Stream the word2vec C binary and keep only the wanted words.

    Format: an ASCII header "<n_words> <dim>\\n", then per entry the word's bytes up to a space, then dim float32.
    Reading the whole 3.6 GB file into a dict would cost ~11 GB of RAM for 3M x 300 float32 plus Python objects; the
    corpus vocabulary is ~5e4 words, so filtering during the scan keeps the working set at a few tens of MB.

    Lookup is lowercase-first with a capitalized fallback: GoogleNews is case sensitive and stores many proper nouns
    only in capitalized form, so 'clinton' misses while 'Clinton' hits. The fallback is counted and reported rather
    than hidden.
    """
    want_lower = {w for w in words_wanted}
    want_cap = {w.capitalize(): w for w in words_wanted}
    want_upper = {w.upper(): w for w in words_wanted}
    got = {}
    src = {}
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"\n"):
            header += fh.read(1)
        n_words, dim = (int(x) for x in header.split())
        vec_bytes = dim * 4
        for _ in range(n_words):
            wb = bytearray()
            while True:
                ch = fh.read(1)
                if ch == b" " or ch == b"":
                    break
                if ch != b"\n":                       # the C tool writes a newline between records
                    wb += ch
            raw = fh.read(vec_bytes)
            w = wb.decode("utf-8", errors="ignore")
            key = None
            if w in want_lower:
                key, how = w, "lower"
            elif w in want_cap and want_cap[w] not in got:
                key, how = want_cap[w], "capitalized"
            elif w in want_upper and want_upper[w] not in got:
                key, how = want_upper[w], "upper"
            if key is None:
                continue
            if key in got and how != "lower":
                continue                              # an exact lowercase hit always wins over a fallback
            got[key] = np.frombuffer(raw, dtype=np.float32, count=dim).copy()
            src[key] = how
    return got, src, int(dim), int(n_words)


def build_cache(force: bool = False) -> str:
    """One pass over the corpus and one pass over word2vec; writes CACHE. Idempotent.

    The cache holds the filtered vocabulary, its embedding matrix, and every document as (word ids, counts) in CSR
    form, so no later step re-reads the 3.6 GB binary.
    """
    if os.path.exists(CACHE) and not force:
        return CACHE
    stop = stopwords()
    tr_txt, tr_y = read_split("train")
    te_txt, te_y = read_split("test")
    texts = tr_txt + te_txt
    labels = np.concatenate([tr_y, te_y])
    split = np.concatenate([np.zeros(len(tr_txt), dtype=np.int8), np.ones(len(te_txt), dtype=np.int8)])

    toks = [[w for w in tokenize(t) if w not in stop] for t in texts]
    df = {}
    for tk in toks:
        for w in set(tk):
            df[w] = df.get(w, 0) + 1
    # the paper's 20NEWS-only rule is a CORPUS frequency cut, not a document-frequency cut; we apply it on total
    # occurrences, which is what "appear less than 5 times across all documents" says
    tf = {}
    for tk in toks:
        for w in tk:
            tf[w] = tf.get(w, 0) + 1
    kept = {w for w, c in tf.items() if c >= MIN_DF}
    emb, src, dim, n_w2v = _read_w2v(kept)
    vocab = sorted(emb)
    index = {w: i for i, w in enumerate(vocab)}
    V = np.stack([emb[w] for w in vocab]).astype(np.float32)

    indptr = [0]
    indices, counts = [], []
    for tk in toks:
        c = {}
        for w in tk:
            j = index.get(w)
            if j is not None:
                c[j] = c.get(j, 0) + 1
        if len(c) > CAP_WORDS:                        # keep the CAP_WORDS most frequent words of this document
            top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:CAP_WORDS]
            c = dict(top)
        ids = np.array(sorted(c), dtype=np.int32)
        indices.append(ids)
        counts.append(np.array([c[i] for i in ids], dtype=np.float64))
        indptr.append(indptr[-1] + ids.size)

    meta = {"n_docs": len(texts), "n_train": len(tr_txt), "n_test": len(te_txt), "vocab_size": len(vocab),
            "dim": dim, "w2v_vocab": n_w2v, "min_tf": MIN_DF, "cap_words": CAP_WORDS,
            "tokens_after_stop": int(sum(len(t) for t in toks)),
            "distinct_after_stop": len(tf), "distinct_after_tf_cut": len(kept),
            "in_w2v": len(vocab), "oov_after_tf_cut": len(kept) - len(vocab),
            "lookup_lower": sum(1 for v in src.values() if v == "lower"),
            "lookup_capitalized": sum(1 for v in src.values() if v == "capitalized"),
            "lookup_upper": sum(1 for v in src.values() if v == "upper"),
            "snapshot": SNAPSHOT, "w2v": W2V_BIN, "stopwords": len(stop)}
    np.savez_compressed(CACHE, V=V, indptr=np.asarray(indptr, dtype=np.int64),
                        indices=np.concatenate(indices).astype(np.int32),
                        counts=np.concatenate(counts), labels=labels, split=split,
                        vocab=np.asarray(vocab, dtype=object), meta=np.asarray(json.dumps(meta), dtype=object))
    return CACHE


_CACHE = None


def load():
    """The cached corpus: V (vocab x dim embeddings), per-document (ids, counts), labels, split, meta."""
    global _CACHE
    if _CACHE is None:
        if not os.path.exists(CACHE):
            build_cache()
        z = np.load(CACHE, allow_pickle=True)
        _CACHE = {"V": z["V"], "indptr": z["indptr"], "indices": z["indices"], "counts": z["counts"],
                  "labels": z["labels"], "split": z["split"], "vocab": z["vocab"],
                  "meta": json.loads(str(z["meta"]))}
    return _CACHE


def doc_nbow(i: int):
    """(word ids, normalized weights) of document i. Weights sum to 1 exactly up to float rounding."""
    c = load()
    s, e = int(c["indptr"][i]), int(c["indptr"][i + 1])
    ids = c["indices"][s:e]
    w = c["counts"][s:e]
    return ids, w / w.sum() if w.size else w


def doc_sizes() -> np.ndarray:
    """Unique-word count of every document AFTER the full filter; this is the m (or n) of an instance."""
    c = load()
    return np.diff(c["indptr"])


def concat_nbow(docs) -> tuple:
    """nBOW of the concatenation of several documents: counts summed over the union, then normalized.

    This is the bag of words of a genuine text (the documents pasted together), which is what makes a large instance
    here real data rather than a synthetic blow-up.
    """
    c = load()
    tot = {}
    for i in docs:
        s, e = int(c["indptr"][i]), int(c["indptr"][i + 1])
        for j, v in zip(c["indices"][s:e], c["counts"][s:e]):
            tot[int(j)] = tot.get(int(j), 0.0) + float(v)
    ids = np.array(sorted(tot), dtype=np.int32)
    w = np.array([tot[int(j)] for j in ids], dtype=np.float64)
    return ids, w / w.sum()


# ----------------------------------------------------------------------------------------------------------------
# The LP
# ----------------------------------------------------------------------------------------------------------------
def cost_matrix(ids_a, ids_b, p: float = 1.0) -> np.ndarray:
    """Ground cost C[i, j] = ||v_i - v_j||_2^p between the two documents' words. p = 1 is Kusner et al.'s metric.

    Computed from the Gram expansion rather than a broadcast difference: at m = n = 1000 the broadcast form would
    materialize a 1000 x 1000 x 300 float array (2.4 GB) while this touches 8 MB.
    """
    V = load()["V"]
    A = V[np.asarray(ids_a)].astype(np.float64)
    B = V[np.asarray(ids_b)].astype(np.float64)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)                        # the expansion can go a few ulp negative on equal vectors
    D = np.sqrt(d2)
    return D if p == 1.0 else D ** p


def _assemble(ids_a, w_a, ids_b, w_b, p, name, meta_extra, with_feas=True, permute=True, seed=0) -> dict:
    """The OSQP dict for one transport LP, in dotmark_ot's exact row convention."""
    m, n = int(ids_a.size), int(ids_b.size)
    N = m * n
    Rm = sp.csr_matrix((np.ones(N), (np.repeat(np.arange(m), n), np.arange(N))), shape=(m, N))
    Cm = sp.csr_matrix((np.ones(N), (np.tile(np.arange(n), m), np.arange(N))), shape=(n, N))
    x_feas = (np.asarray(w_a)[:, None] * np.asarray(w_b)[None, :]).ravel()   # product coupling, exactly feasible
    a = Rm @ x_feas
    b = Cm @ x_feas                                    # marginals READ OFF x_feas, so mass balance is exact
    q = cost_matrix(ids_a, ids_b, p).ravel()

    A = sp.vstack([sp.vstack([Rm, Cm], format="csr"), sp.identity(N, format="csr", dtype=np.float64)], format="csc")
    rhs = np.concatenate([a, b])
    l = np.concatenate([rhs, np.zeros(N)])
    u = np.concatenate([rhs, np.full(N, np.inf)])
    fam = np.concatenate([np.zeros(m, dtype=np.int8), np.ones(n, dtype=np.int8), np.full(N, 2, dtype=np.int8)])
    meta = {"dataset": "SetFit/20_newsgroups", "embedding": "GoogleNews-vectors-negative300", "p": float(p),
            "m": m, "n": n, "n_var": int(N), "rows": int(A.shape[0]), "nnz": int(A.nnz),
            "mass_a": float(a.sum()), "mass_b": float(b.sum()), "cost_max": float(q.max()),
            "cost_min": float(q.min()), "row_family": fam}
    meta.update(meta_extra)
    d = {"P": sp.csc_matrix((N, N)), "q": q, "r": 0.0, "A": A, "l": l, "u": u, "name": name, "meta": meta}
    if with_feas:
        d["x_feas"] = x_feas
    return permute_ot(d, 1000 + seed) if permute else d


def make_wmd(pair, p: float = 1.0, permute: bool = True, seed: int = 0, with_feas: bool = True) -> dict:
    """The WMD transport LP for a document pair. `pair` is (doc_a, doc_b), two integer corpus indices."""
    i, j = (int(pair[0]), int(pair[1]))
    ids_a, w_a = doc_nbow(i)
    ids_b, w_b = doc_nbow(j)
    if ids_a.size == 0 or ids_b.size == 0:
        raise ValueError(f"document {i if ids_a.size == 0 else j} has no in-vocabulary word")
    c = load()
    extra = {"doc_a": i, "doc_b": j, "label_a": int(c["labels"][i]), "label_b": int(c["labels"][j]),
             "split_a": int(c["split"][i]), "split_b": int(c["split"][j]), "kind": "pair"}
    return _assemble(ids_a, w_a, ids_b, w_b, p, f"wmd_{i}v{j}", extra, with_feas, permute, seed)


def make_wmd_concat(docs_a, docs_b, p: float = 1.0, permute: bool = True, seed: int = 0,
                    with_feas: bool = True) -> dict:
    """A WMD LP between two CONCATENATED super-documents; the route to n = 1e5 .. 1e6.

    Both sides are real 20 Newsgroups text; only the document boundaries are erased. meta records how many documents
    each side merged so no size in a table can be mistaken for a natural single pair.
    """
    ids_a, w_a = concat_nbow(docs_a)
    ids_b, w_b = concat_nbow(docs_b)
    extra = {"docs_a": [int(x) for x in docs_a], "docs_b": [int(x) for x in docs_b],
             "n_docs_a": len(docs_a), "n_docs_b": len(docs_b), "kind": "concat"}
    name = f"wmdcat_{len(docs_a)}x{len(docs_b)}_{ids_a.size}x{ids_b.size}"
    return _assemble(ids_a, w_a, ids_b, w_b, p, name, extra, with_feas, permute, seed)


def feasibility(d: dict) -> dict:
    """Mass balance and the acceptance metric's row violation at x_feas (dotmark_ot.feasibility, same convention)."""
    return dotmark_ot.feasibility(d)


# ----------------------------------------------------------------------------------------------------------------
# Listings
# ----------------------------------------------------------------------------------------------------------------
def sample_pairs(n_pairs: int = 4000, seed: int = 0, split: int | None = None) -> np.ndarray:
    """Uniformly random document pairs (i != j), optionally restricted to one split (0 train, 1 test)."""
    c = load()
    pool = np.flatnonzero((doc_sizes() > 0) if split is None else ((doc_sizes() > 0) & (c["split"] == split)))
    rng = np.random.default_rng(seed)
    out = np.empty((n_pairs, 2), dtype=np.int64)
    k = 0
    while k < n_pairs:
        cand = rng.choice(pool, size=(n_pairs - k, 2))
        cand = cand[cand[:, 0] != cand[:, 1]]
        out[k:k + cand.shape[0]] = cand
        k += cand.shape[0]
    return out


def pair_table(pairs) -> dict:
    """Instance size n = m * n over a set of pairs, with the quantiles the report needs."""
    s = doc_sizes()
    pairs = np.asarray(pairs)
    m, n = s[pairs[:, 0]], s[pairs[:, 1]]
    nv = m.astype(np.int64) * n.astype(np.int64)
    qs = [0, 5, 25, 50, 75, 90, 95, 99, 100]
    return {"n_pairs": int(pairs.shape[0]),
            "m_quantiles": {f"p{q}": float(np.percentile(m, q)) for q in qs},
            "n_var_quantiles": {f"p{q}": float(np.percentile(nv, q)) for q in qs},
            "n_var_mean": float(nv.mean()), "m_mean": float(m.mean()),
            "square_frac": float((m == n).mean()),
            "frac_ge_1e4": float((nv >= 1e4).mean()), "frac_ge_1e5": float((nv >= 1e5).mean()),
            "frac_ge_1e6": float((nv >= 1e6).mean())}


def largest_pairs(k: int = 20, split: int | None = None) -> list:
    """The k largest natural (single-document) instances: pair the longest documents with each other."""
    c = load()
    s = doc_sizes()
    ok = np.flatnonzero(s > 0 if split is None else ((s > 0) & (c["split"] == split)))
    order = ok[np.argsort(-s[ok])]
    top = order[:max(2, int(np.ceil(np.sqrt(2 * k))) + 1)]
    out = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            out.append((int(top[a]), int(top[b]), int(s[top[a]]) * int(s[top[b]])))
    out.sort(key=lambda t: -t[2])
    return out[:k]


def concat_groups(target_words: int, n_groups: int = 2, seed: int = 0, label: int | None = None) -> list:
    """Document index lists whose merged bag has about `target_words` unique words.

    Documents are drawn from one newsgroup (so the merged text is topically coherent, which keeps the transport
    problem meaningful rather than a bag of unrelated vocabulary) and added until the union stops growing enough.
    """
    c = load()
    labels = c["labels"]
    s = doc_sizes()
    rng = np.random.default_rng(seed)
    groups = []
    for g in range(n_groups):
        lab = int(label) if label is not None else int(g % 20)
        pool = np.flatnonzero((labels == lab) & (s > 0))
        rng.shuffle(pool)
        seen, chosen = set(), []
        for i in pool:
            st, e = int(c["indptr"][i]), int(c["indptr"][i + 1])
            seen.update(c["indices"][st:e].tolist())
            chosen.append(int(i))
            if len(seen) >= target_words:
                break
        groups.append(chosen)
    return groups
