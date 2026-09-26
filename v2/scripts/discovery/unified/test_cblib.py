"""Certify cblib.load_cbf against a hand-written CBF instance that uses every supported cone (F, L+, L-, L=, Q, QR)
under both VAR and CON, with MAX sense, constants, comments and a trailing CHANGE block. Expected matrices are written
out by hand from the CBF semantics, not recomputed through the loader. Also checks that the QR rotation accepts
exactly the points the rotated cone does, and that integer, PSD, EXP and POW instances are refused.
Runs as a plain script (exit code 1 on failure) or under pytest."""
import gzip, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
from cblib import load_cbf, objective, violation

S2 = np.sqrt(0.5)

TINY = """# hand-written instance covering every supported cone
VER
3

OBJSENSE
MAX

VAR
11 6
F 1
L+ 1
L- 1
L= 1
Q 3
QR 4

CON
11 7
F 1
L+ 1
L- 1
L= 1
Q 3
QR 3
Q 1

OBJACOORD
2
0 1.5
7 -2.0

OBJBCOORD
3.0

# rows: 0 free, 1 L+, 2 L-, 3 L=, 4-6 Q, 7-9 QR, 10 Q of dimension 1
ACOORD
14
0 0 9.0
1 0 1.0
1 1 2.0
2 2 3.0
3 3 1.0
3 0 -1.0
4 4 2.0
5 5 1.0
6 6 1.0
6 0 0.5
7 7 1.0
8 8 1.0
9 9 4.0
10 10 1.0

BCOORD
7
1 -1.0
2 4.0
3 0.5
4 1.0
6 -2.0
8 2.0
10 -3.0

CHANGE
OBJACOORD
1
0 100.0
"""


_TMP = tempfile.TemporaryDirectory()                                   # removed at interpreter exit


def _write(text, gz=False):
    fd, path = tempfile.mkstemp(suffix=".cbf.gz" if gz else ".cbf", dir=_TMP.name)
    os.close(fd)
    with (gzip.open(path, "wt") if gz else open(path, "w")) as fh:
        fh.write(text)
    return path


def _rows(entries, n):
    """Dense matrix from a list of {col: value} rows."""
    M = np.zeros((len(entries), n))
    for i, row in enumerate(entries):
        for j, v in row.items():
            M[i, j] = v
    return M


def _unit(j, n, scale=1.0):
    e = np.zeros(n)
    e[j] = scale
    return e


def test_tiny_mapping():
    d = load_cbf(_write(TINY))
    n, inf = 11, np.inf
    q = np.zeros(n); q[0], q[7] = -1.5, 2.0                       # MAX negated; CHANGE block ignored
    assert np.array_equal(d["q"], q) and d["r"] == -3.0 and d["obj_sign"] == -1.0 and d["sense"] == "MAX"
    assert d["P"].shape == (n, n) and d["P"].nnz == 0
    # linear rows: CON L+, L-, L=, Q1 (as L+), then VAR L+, L-, L=; the free CON row 0 and free VAR x0 give nothing
    A = _rows([{0: 1.0, 1: 2.0}, {2: 3.0}, {3: 1.0, 0: -1.0}, {10: 1.0}, {1: 1.0}, {2: 1.0}, {3: 1.0}], n)
    assert np.array_equal(d["A"].toarray(), A)
    assert np.array_equal(d["l"], [1.0, -inf, -0.5, 3.0, 0.0, -inf, 0.0])
    assert np.array_equal(d["u"], [inf, -4.0, -0.5, inf, inf, 0.0, 0.0])
    assert len(d["soc"]) == 4
    expected = [
        # CON Q on rows 4-6: t = 2 x4 + 1, rest = (x5, 0.5 x0 + x6 - 2)
        (_rows([{5: 1.0}, {0: 0.5, 6: 1.0}], n), [0.0, -2.0], _unit(4, n, 2.0), 1.0),
        # CON QR on rows 7-9: t1 = x7, t2 = x8 + 2, w = 4 x9; u = (t1+t2)/sqrt2, v = (t1-t2)/sqrt2
        (_rows([{7: S2, 8: -S2}, {9: 4.0}], n), [-2.0 * S2, 0.0], _unit(7, n, S2) + _unit(8, n, S2), 2.0 * S2),
        # VAR Q on x4..x6
        (_rows([{5: 1.0}, {6: 1.0}], n), [0.0, 0.0], _unit(4, n), 0.0),
        # VAR QR on x7..x10
        (_rows([{7: S2, 8: -S2}, {9: 1.0}, {10: 1.0}], n), [0.0, 0.0, 0.0], _unit(7, n, S2) + _unit(8, n, S2), 0.0),
    ]
    for c, (F, g, f, h) in zip(d["soc"], expected):
        assert sp.isspmatrix_csc(c["F"]) and c["F"].shape == F.shape
        assert np.allclose(c["F"].toarray(), F, rtol=0, atol=1e-15) and np.allclose(c["g"], g, rtol=0, atol=1e-15)
        assert isinstance(c["f"], np.ndarray) and c["f"].shape == (n,)
        assert np.allclose(c["f"], f, rtol=0, atol=1e-15) and abs(c["h"] - h) < 1e-15 and isinstance(c["h"], float)
    assert d["cbf_cones"] == {"F": 2, "L+": 2, "L-": 2, "L=": 2, "Q": 3, "QR": 2}


def test_gzip_matches_plain():
    a, b = load_cbf(_write(TINY)), load_cbf(_write(TINY, gz=True))
    assert np.array_equal(a["A"].toarray(), b["A"].toarray()) and np.array_equal(a["q"], b["q"])
    assert all(np.array_equal(x["F"].toarray(), y["F"].toarray()) for x, y in zip(a["soc"], b["soc"]))


def test_sparse_f_matches_dense():
    a, b = load_cbf(_write(TINY)), load_cbf(_write(TINY), dense_f=False)
    for x, y in zip(a["soc"], b["soc"]):
        assert sp.issparse(y["f"]) and y["f"].shape == (11,) and np.array_equal(y["f"].toarray(), x["f"])
    v = np.random.default_rng(1).standard_normal(11)
    assert all(abs(x["f"] @ v - y["f"] @ v) < 1e-15 for x, y in zip(a["soc"], b["soc"]))


def test_clarabel_dump_stacks_cones():
    """The runner's stacked [f'; F] must reproduce every cone row, for dense and sparse f alike."""
    from cblib_clarabel_runner import dump_conic, _csc_load
    for dense in (True, False):
        d = load_cbf(_write(TINY), dense_f=dense)
        path = _write("")
        dump_conic(d, path + ".npz")
        z = np.load(path + ".npz")
        G, hg, dims = _csc_load(z, "G").toarray(), z["hg"], z["dims"]
        off = 0
        for c, k in zip(d["soc"], dims):
            f = c["f"].toarray() if sp.issparse(c["f"]) else c["f"]
            assert np.array_equal(G[off], f) and np.array_equal(G[off + 1:off + k], c["F"].toarray())
            assert hg[off] == c["h"] and np.array_equal(hg[off + 1:off + k], c["g"])
            off += k
        assert off == G.shape[0]


def test_objective_and_violation():
    d = load_cbf(_write(TINY))
    # hand-built feasible point: VAR L= fixes x3 = 0, so CON L= (x3 - x0 = -0.5) gives x0 = 0.5 and CON L+ (x0 + 2 x1
    # >= 1) needs x1 = 0.25; x2 = -2 (3 x2 <= -4); x10 = 3 (Q1 row); x4, x5, x6 = 2, 1, 1 (VAR Q: 2 >= sqrt 2, CON Q:
    # 5 >= ||(1, 0.25 + 1 - 2)||); x7, x8, x9 = 5, 1, 0.5 (CON QR: 2 * 5 * 3 >= 4, VAR QR: 2 * 5 * 1 >= 0.25 + 9)
    x = np.array([0.5, 0.25, -2.0, 0.0, 2.0, 1.0, 1.0, 5.0, 1.0, 0.5, 3.0])
    lin, soc = violation(d, x)
    assert lin == 0.0 and soc == 0.0
    assert objective(d, x) == 1.5 * 0.5 - 2.0 * 5.0 + 3.0           # file's own (MAX) sense
    x_bad = x.copy(); x_bad[7] = 1.0                                   # VAR QR now 2 < 9.25
    assert violation(d, x_bad)[1] > 0


def test_qr_rotation_is_exact():
    cbf = "VER\n3\nOBJSENSE\nMIN\nVAR\n4 1\nQR 4\n"
    c = load_cbf(_write(cbf))["soc"][0]
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(20000):
        x = rng.standard_normal(4) * rng.choice([0.1, 1.0, 10.0])
        in_qr = x[0] >= 0 and x[1] >= 0 and 2 * x[0] * x[1] >= x[2] ** 2 + x[3] ** 2
        slack = c["f"] @ x + c["h"] - np.linalg.norm(c["F"] @ x + c["g"])
        if abs(slack) < 1e-9 * (1 + np.abs(x).max() ** 2):
            continue                                                   # on the boundary, rounding decides
        assert (slack >= 0) == in_qr, x
        agree += 1
    assert agree > 19000


def test_q1_becomes_linear_row():
    d = load_cbf(_write("VER\n3\nOBJSENSE\nMIN\nVAR\n2 2\nQ 1\nF 1\n"))
    assert d["soc"] == [] and np.array_equal(d["A"].toarray(), [[1.0, 0.0]])
    assert np.array_equal(d["l"], [0.0]) and np.array_equal(d["u"], [np.inf])


def test_refuses_unsupported():
    head = "VER\n3\nOBJSENSE\nMIN\n"
    for body, why in (("VAR\n3 1\nQ 3\nINT\n1\n0\n", "integer"),
                      ("PSDVAR\n1\n2\nVAR\n1 1\nF 1\n", "PSD"),
                      ("VAR\n3 1\nF 3\nCON\n3 1\nEXP 3\n", "EXP"),
                      ("POWCONES\n1 2\n2\n1.0\n1.0\nVAR\n3 1\n@0:POW 3\n", "POW")):
        try:
            load_cbf(_write(head + body))
        except ValueError as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"{why} instance was accepted")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print("ok  ", name)
        except Exception as e:                                          # report every failure, not only the first
            failed += 1
            print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
