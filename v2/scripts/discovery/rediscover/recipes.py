"""Give the proposer the context that actually produced the discovery.

An earlier version of this file scored against a move menu I wrote while
already knowing the answers, which is circular.  What actually produced each
recipe was the problem's MEASURED properties plus a FAILED attempt with its
numbers.  On the unmixing class the first arm built was FISTA; watching it
need more than 10^5 iterations is what produced the observation that the
pixels share one Hessian and that a method able to factorize it once would
win.  On the Poisson class the trigger was that the data term's gradient is
not Lipschitz, so the split used for the Gaussian case cannot apply.

Two conditions, so the result can distinguish the ingredients:

  cold      the problem statement and its measured properties, nothing else
  failure   the same, plus the record of what was tried and what it measured

If `failure` names the move and `cold` does not, the measured failure is the
active ingredient, which is the paper's own mechanism: the fitness makes the
absence visible.  If both do, the move is obvious from the structure alone.

The lasso problem is deliberately NOT posed as a discovery test here: that
recipe came from reading the published derivation, so scoring a model on it
would measure recall, not discovery.  It is included as a recall control.
"""
from __future__ import annotations

FRAMEWORK = """Context. Solvers here are built from a composite convex form
  minimize f(x) + g(x) + h(Lx)
where f is smooth, g and h are proximable, and L is a linear map. A design
decision is: which term each piece of the problem goes into, which algorithm
family runs the resulting split, what state is carried between subproblems,
and what runs on which device."""

PROBLEMS = {
 "cuprite_unmixing": {
  "cold": """Design a solver. For each of 47,750 pixels i independently:
    minimize (1/2)||D x - y_i||^2   subject to  x >= 0,  sum(x) = 1
Measured properties of this instance:
  - D is 188 x 498. It is THE SAME matrix for every one of the 47,750 pixels;
    only the right-hand side y_i changes.
  - D has condition number 7.5e8 and full row rank 188.
  - 498 > 188, so each pixel's problem is underdetermined.
  - Target: a certified relative optimality gap of 1e-4 on the WORST pixel.
  - Hardware available: one GPU with fast float64.
What solver do you build, and why?""",
  "failure": """[cold]
What has been tried, and what it measured:
  - FISTA (accelerated projected gradient) with the exact Euclidean
    projection onto the simplex, step 1/L with L = sigma_max(D)^2 = 29873.
    On a 2,000-pixel subset it ran 46,801 iterations in 70 s and reached a
    worst-pixel gap of only 2.1e-4; the MEDIAN pixel was at 3.5e-5 but the
    worst would not come down. Adaptive restart helped by about 2x and did
    not change the picture.
  - Extrapolating, the full 47,750 pixels would need well over 10^5
    iterations to certify 1e-4.
Given that, what do you change?""",
  "keys": {
   "share one factorization across pixels": [
     "shared", "share the", "same for every pixel", "once for all pixels",
     "one factorization", "reuse the factor", "precompute the inverse",
     "factorize once", "amortize", "cache the factor", "single cholesky"],
   "switch family to ADMM / splitting with an exact prox": [
     "admm", "douglas-rachford", "douglas rachford", "alternating direction",
     "operator splitting", "proximal splitting"],
   "batch the pixels on the device": [
     "batch", "batched", "all pixels at once", "vectorize", "simultaneously"],
  }},

 "poisson_deconv": {
  "cold": """Design a solver.
    minimize  sum_i ( (Ax)_i - y_i log((Ax)_i) )  +  lambda TV(x)
    subject to x >= 0
Measured properties:
  - A is a known Gaussian blur applied by FFT; ||A||^2 = 1.
  - y are Poisson photon counts, peak about 100 photons, min 4.
  - TV is isotropic, on a 2D image; the gradient operator has ||D||^2 <= 8.
  - Target: a certified relative gap of 1e-6.
What solver do you build, and why?""",
  "failure": """[cold]
What has been tried, and what it measured:
  - The Gaussian analogue of this problem was solved by Condat-Vu with
    f = (1/2)||Ax-y||^2 as the SMOOTH term (gradient Lipschitz constant
    ||A||^2 = 1), g = the box indicator, and h = lambda||.||_{2,1} composed
    with the gradient. That reached 1e-8 in seconds.
  - Putting the Poisson term in f the same way fails: its gradient is
    A^T(1 - y/(Ax)), which is unbounded as Ax approaches 0, so there is no
    Lipschitz constant to set a step from, and the iteration diverges.
Given that, what do you change?""",
  "keys": {
   "move the data term behind the linear map": [
     "behind the linear map", "compose", "into h", "put it in h", "relocate",
     "move the data term", "as h(ax)", "through a"],
   "use its conjugate prox rather than its gradient": [
     "conjugate", "prox of the", "dual", "moreau", "legendre", "fenchel",
     "proximal operator of the poisson", "closed form"],
   "stack the two operators": [
     "stack", "concatenate", "[a; d]", "both operators", "block operator",
     "k = ", "combined operator"],
  }},

 "lasso_path_recall": {
  "cold": """Solve the lasso regularization path: for a decreasing grid of
K = 100 penalties, minimize (1/2n)||y - Xw||^2 + lambda||w||_1. X is
6000 x 5000 and dense, and is the same for every penalty on the grid. A
certified duality gap of 1e-10 is required at every grid point.
What solver do you build, and why?""",
  "failure": """[cold]
What has been tried, and what it measured:
  - Coordinate descent with warm starts along the grid certifies 1e-4
    quickly, but the time to certify grows sharply as the target tightens:
    roughly 3x from 1e-4 to 1e-8, and it fails to certify 1e-12 at all on
    some designs.
Given that, what do you change?""",
  "keys": {
   "exact path / homotopy": [
     "homotopy", "exact path", "piecewise linear", "lars", "breakpoint",
     "active set path", "least angle"],
   "maintain the factorization along the path": [
     "cholesky update", "downdate", "rank-one", "update the factor",
     "incremental", "maintain the factor", "low-rank update"],
   "screening": [
     "screen", "screening", "safe rule", "strong rule", "gap-safe",
     "discard", "working set"],
  }},
}

SYS = ("You are designing a numerical solver for a convex optimization "
       "problem. Give a concrete recipe: which algorithm family, what each "
       "term of the problem becomes, what is carried between subproblems, "
       "and what runs where. Be specific and brief. Do not write code.")
