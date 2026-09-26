"""Executable component space; no library of winning solver implementations.

This is a program-synthesis experiment, NOT the theorem-certified genome grammar.
Static checks restrict the API. Independent numerical certificates judge outputs.
The AST filter is an experiment guardrail, not a security sandbox.
"""
from __future__ import annotations
import ast
import hashlib
import json

HOOKS = {
    1: "formulate", 2: "advance", 3: "initialize", 4: "restrict",
    5: "handoff", 6: "contract", 7: "admit", 8: "width",
}
ALLOWED_IMPORTS = {"numpy", "jax", "scipy", "math", "functools"}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "open", "input", "__import__", "getattr",
                   "setattr", "delattr", "globals", "locals", "vars", "breakpoint"}
FORBIDDEN_ATTRS = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
                  "fromfile", "tofile", "memmap", "ctypeslib", "f2py", "callback",
                  "io_callback", "pure_callback", "debug", "config", "optimize"}

def validate_source(source):
    if not isinstance(source, str) or len(source) > 100_000:
        raise ValueError("source must be a string of at most 100000 characters")
    tree = ast.parse(source)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = set(HOOKS.values()) | {"solution"}
    if missing - names:
        raise ValueError("missing hooks: " + ", ".join(sorted(missing - names)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = ([a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import)
                     else [(node.module or '').split('.')[0]])
            if any(r not in ALLOWED_IMPORTS for r in roots):
                raise ValueError("only numpy, jax, scipy, math, functools imports are permitted")
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module or ''])
            if any(m.startswith('scipy.') and not m.startswith(('scipy.linalg', 'scipy.special'))
                   for m in modules):
                raise ValueError('only scipy.linalg and scipy.special are permitted')
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("relative imports are unavailable")
            if (isinstance(node,ast.ImportFrom) and node.module=='scipy'
                    and any(a.name not in ('linalg','special') for a in node.names)):
                raise ValueError('only scipy.linalg and scipy.special are permitted')
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_CALLS or node.id.startswith('__')):
            raise ValueError("forbidden name: " + node.id)
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN_ATTRS):
            raise ValueError("forbidden attribute: " + node.attr)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx,(ast.Store,ast.Del)):
            raise ValueError('attribute mutation is unavailable; use explicit array/dictionary state')
    return tree

def source_id(source):
    """AST hash ignores whitespace and comments, but deliberately not semantics."""
    return hashlib.sha256(ast.dump(validate_source(source), include_attributes=False).encode()).hexdigest()

def layer_ids(source):
    tree = validate_source(source)
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    def dependency_hash(hook):
        seen, pending = set(), [hook]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            pending.extend(n.id for n in ast.walk(functions[name])
                           if isinstance(n,ast.Name) and n.id in functions and n.id not in seen)
        payload = '\n'.join(ast.dump(functions[n]) for n in sorted(seen))
        return hashlib.sha256(payload.encode()).hexdigest()
    return {str(k):dependency_hash(hook) for k,hook in HOOKS.items()}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def apply_function_patch(parent, patch):
    """Replace/add executable functions, preserving unchanged tested primitives."""
    if not isinstance(patch, dict) or not patch or len(patch) > 32:
        raise ValueError('replacements must be a nonempty mapping with at most 32 functions')
    tree = validate_source(parent)
    replacements = {}
    for name, source in patch.items():
        part = ast.parse(source)
        if len(part.body) != 1 or not isinstance(part.body[0], ast.FunctionDef):
            raise ValueError('each replacement must contain exactly one function (decorators allowed)')
        if part.body[0].name != name:
            raise ValueError('replacement key and function name differ')
        replacements[name] = part.body[0]
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in replacements:
            body.append(replacements.pop(node.name))
        else:
            body.append(node)
    body.extend(replacements.values())
    tree.body = body
    code = ast.unparse(tree) + '\n'
    validate_source(code)
    return code

PATCH_API = '''Output a FUNCTION PATCH, not an entire rewritten module.
Return JSON {"rationale":"short mathematical reason and shape analysis",
"replacements":{"function_name":"complete new def, including any decorators",...}}.
Each value must parse as exactly one function. You can replace any existing hook
or helper, and add helper functions. Standard imported aliases np, jax, jnp and
partial are already available. Put additional allowed imports inside a function
if needed. Unchanged functions are retained byte-for-byte in mathematical meaning.
Prefer reusing the existing working project(v) routine over rewriting it.
The smallest complete structural change is preferable to many unrelated changes.
Do not leave placeholders. Check shapes: D=(n,p), Y=(n,m), coefficients=(p,m),
Gram=(p,p), one scalar per RHS=(m,). Keep output to the functions you changed.
'''

API = '''Write a complete Python numerical solver module with these executable hooks.
Imports available: numpy as np, jax, jax.numpy as jnp, scipy.linalg, math, functools.
No files, network, installed solvers, project code, environment changes, callbacks,
or evaluator access. JAX float64 is enabled by the runner. Use ordinary JAX/NumPy
operations and your own algorithms. Do not call scipy.optimize or solvers from a package.

Layer 1: formulate(D, Y, options) -> problem (any dictionary).
  D and Y are host float64 arrays. All mathematical setup belongs here.
Layer 2: advance(problem, state, steps, options) -> state.
  Perform your chosen numerical iteration(s); steps is the requested work chunk.
  JIT kernels belong in the module, or can be built during formulate.
Layer 3: initialize(problem, options) -> state (any dictionary).
  Allocate carried algorithm state. Cache whatever your algorithm needs.
Layer 4: restrict(problem, state, options) -> (problem, state).
  Optional restriction of active work. A no-op is legal. Returned solutions must
  still include EVERY original column; the evaluator always checks all columns.
Layer 5: handoff(problem, state, feedback, options) -> (problem, state).
  Optional change of engine, stopping policy or state. No-op is legal.
  feedback has elapsed_s, iteration, worst_gap, feasibility; only this run's data.
Layer 6: contract() -> {"dtype": "float64" or "float32"}.
  Chooses internal arithmetic. The independent judge ALWAYS uses float64 and the
  SAME target and feasibility threshold. This hook cannot weaken acceptance.
Layer 7: admit(meta) -> "gpu" or "cpu".
  This determines the actual jax.default_device context. Respect options['device'].
Layer 8: width(meta) -> {"chunk": integer 1..8192, "unroll": integer 1..16}.
  The driver passes chunk to advance and unroll through options. Use unroll in
  your compiled loop if useful. Choose width from work and hardware, not data IDs.
solution(problem, state) -> full coefficient array of shape (D.shape[1], Y.shape[1]).

options contains tol, budget_s, dtype, device, chunk, unroll. meta contains n_rows,
n_features, n_rhs, gpu_name. The driver calls formulate, initialize, then repeatedly
advance, solution, an independent certificate, restrict, handoff. It stops on
success or a fixed time limit. Time includes setup, JIT, execution, transfer and
every certificate check; Python/JAX runtime startup is separately recorded.
Never return NaNs or an incomplete batch. Default arguments on helpers are fine.
Return JSON {"rationale": "short mechanism and expected benefit", "code": "full module"}.
'''

TASK = '''For each column y of Y solve
    minimize 0.5 * ||D x - y||_2^2 subject to x >= 0 and sum(x) = 1.
D is shared by all columns. The important real design is 188 by 498, severely
ill-conditioned and underdetermined. Training includes real spectral-unmixing
pixels and synthetic cases. The output is one full feasible coefficient matrix.
The fixed acceptance bar on EVERY column is the Frank-Wolfe gap
 (gradient(x).T @ x - min(gradient(x))) / (1 + abs(objective(x))) <= 1e-4,
with nonnegativity and sum-to-one error <= 1e-9, recomputed independently in float64.
Optimize measured time to this accepted solution on an A100 80GB. An iteration
reduction that loses wall time is a loss. One hard column cannot be discarded.
You may invent a different algorithm, formulation, state, restriction, transition,
precision policy, device policy or compiled loop. Supply executable code, not advice.
'''

# This is a DIFFERENT, explicitly labelled card ablation from the paper's lasso D3-D6.
# Withheld and independent-resampling arms never receive these strings.
HSI_CARDS = '''Four expert cards for this arm only:
1. Variable splitting permits an exact quadratic proximal solve plus simplex projection.
2. All right-hand sides can reuse one factorization of the shifted Gram matrix.
3. Retire certified columns while retaining their full output and rechecking every column.
4. Group iterations in a compiled device loop and measure certificate cadence as part of cost.
'''

REFERENCES = '''Standard-method teaching notes (not a task-specific solver):
Proximal gradient for f+g: x_next=prox_(t*g)(x-t*grad(f)(x)), with a valid step.
Accelerated proximal gradient carries a momentum point and an extrapolation scalar;
acceleration may help iteration counts but should be measured to the requested bar.
For min f(x)+g(z), x=z, scaled ADMM alternates
 x=prox_(f/rho)(z-u), z=prox_(g/rho)(x+u), u=u+x-z.
Douglas-Rachford composes reflected proximal maps. Derive the proximal operators
for the given objective, rather than assuming gradients are always the right primitive.
For a constrained quadratic, active-set methods solve a smaller equality-constrained
quadratic and check excluded-coordinate optimality; singular systems and changing
active sets require care. Newton-type methods can use factorizations and safeguards.
Repeated linear systems may reuse symbolic or numerical state when their matrices
permit it. Hardware choices must be assessed including setup and transfers.
References: Parikh & Boyd, Proximal Algorithms (2014),
https://web.stanford.edu/~boyd/papers/prox_algs.html ; Boyd et al., ADMM (2011),
https://web.stanford.edu/~boyd/papers/admm_distr_stats.html .
These are starting methods to learn from. Do not call a solver library.
'''
