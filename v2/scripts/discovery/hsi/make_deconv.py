"""Immune cell-type deconvolution as a simplex-constrained least squares.

    min_x  1/2||A x - b_i||^2   s.t.  x >= 0,  1^T x = 1     for each sample i

A is a signature matrix: LM22, the 547-gene by 22-immune-cell-type matrix that
CIBERSORT introduced (Newman et al., Nature Methods 2015), or quanTIseq's
TIL10.  x is the cell-type composition of a bulk tumour sample.  This is the
SAME problem the hyperspectral class solves, on the same certified operator,
which is why the same arm runs on it unchanged: a spectral library and a gene
signature matrix are the same object with different units.

The medical reading matters for this line of work.  An immune composition
drives immunotherapy decisions, and a clinician has good reason not to accept
a number from a black box.  Every number our arm returns carries a recomputed
optimality certificate on the original problem.

Mixtures are generated in silico from the signature matrix with known ground
truth, which is the validation design the deconvolution literature uses,
because real bulk samples have no ground-truth composition.
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sig", default="/home/miria/data/deconv/LM22.txt")
ap.add_argument("--samples", type=int, default=10000)
ap.add_argument("--noise", type=float, default=0.05)
ap.add_argument("--out", default="/home/miria/data/deconv/deconv_lm22.npz")
a = ap.parse_args()

rows = open(a.sig).read().strip().split("\n")
header = rows[0].split("\t")
genes, vals = [], []
for r in rows[1:]:
    p = r.split("\t")
    genes.append(p[0]); vals.append([float(v) for v in p[1:]])
A = np.asarray(vals)                      # genes x cell types
rng = np.random.default_rng(0)
# Dirichlet compositions: sparse and realistic, most cell types near zero
Xtrue = rng.dirichlet(np.full(A.shape[1], 0.5), size=a.samples).T   # types x samples
B = A @ Xtrue
B = B * np.exp(a.noise * rng.standard_normal(B.shape))              # multiplicative
np.savez(a.out, A=A, B=B, Xtrue=Xtrue, genes=np.array(genes),
         celltypes=np.array(header[1:]))
print("signature %s: %d genes x %d cell types" % (a.sig.split('/')[-1], *A.shape))
print("  cond(A) = %.3e   samples = %d   noise = %.0f%% multiplicative"
      % (np.linalg.cond(A), a.samples, a.noise * 100))
print("  wrote %s" % a.out)
