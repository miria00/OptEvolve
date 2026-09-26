"""Random search over the VALID genome region (the mandatory control).

Proposals are drawn from per-scheme log-uniform stepsize ranges shaped by
the side conditions - now including avg blends, finite-prefix switch_k
genomes, metric (rescaling) genes on LP cells, dys (on problems without
a linear map) and Halpern-anchor restarts - then rejection-sampled
against the Stage-0 gate (atlas.genome.validate): a draw that violates
its side condition is discarded and counted (stage0_rejections), NEVER
clamped. This is the "does the LLM add anything" ablation baseline over
the identical space.

MENU NOTE: the residual-window "switch" mix is DEPRECATED and no longer
proposed (switch_k replaces it); it stays validated/runnable for old
genomes and its regression tests.
"""
from __future__ import annotations

from typing import List

import numpy as np

from atlas.genome import (
    L_SMOOTH_DUAL,
    L_SMOOTH_PRIMAL,
    MAX_SWITCH_K,
    METRIC_PROBLEM_KINDS,
    OPNORM,
    TV_SCHEMES,
    Backend,
    Genome,
    GenomeParams,
    Metric,
    Mix,
    SubScheme,
    Wrapper,
    validate,
)

_RESTART_KS = (25, 50, 100, 200, 400)
_SWITCH_KS = (25, 100, 300, 1000, 3000)  # finite-prefix lengths (<= MAX_SWITCH_K)
_RUIZ_ITERS = (5, 10, 20)
_BACKEND_KS = (10, 25, 50, 100, 250)

assert max(_SWITCH_KS) <= MAX_SWITCH_K


class RandomBackend:
    """SearchBackend: seeded random proposals over the valid region."""

    def __init__(
        self,
        seed: int = 0,
        schemes=None,
        problem_kind: str = "tv_denoise",
        p_halpern: float = 0.15,
        p_rho: float = 0.25,
        p_restart: float = 0.3,  # P(restart | halpern)
        p_alpha: float = 0.2,
        p_mix: float = 0.10,  # avg blends
        p_switch: float = 0.08,  # finite-prefix switch_k genomes
        p_backend: float = 0.10,  # execution-policy gene (base/avg genomes)
        p_metric: float = 0.15,  # metric (rescaling) gene; LP cells only
        p_outer: float = 0.0,  # registered outer wrapper (knowledge base); 0 keeps legacy draws
        max_tries: int = 1000,
    ):
        self.rng = np.random.default_rng(seed)
        self.problem_kind = problem_kind
        if schemes is None:
            if problem_kind == "tv_denoise":
                schemes = TV_SCHEMES
            elif problem_kind == "lp_netflow":
                schemes = ("pdhg", "condat_vu")
            else:  # three_composite
                schemes = ("dys",)
        self.schemes = tuple(schemes)
        self.p_halpern = p_halpern
        self.p_rho = p_rho
        self.p_restart = p_restart
        self.p_alpha = p_alpha
        self.p_mix = p_mix
        self.p_switch = p_switch
        self.p_backend = p_backend
        self.p_metric = p_metric
        self.p_outer = p_outer
        self.max_tries = max_tries
        self.stage0_rejections = 0
        self.history: list[tuple[Genome, float, dict]] = []

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    # -- sampling ----------------------------------------------------------

    def _log_uniform(self, lo: float, hi: float) -> float:
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def _draw_steps(self, scheme: str, aggressive: bool = False):
        """(tau, sigma) for one scheme; `aggressive` widens the ranges past
        the side conditions (used only inside switch genomes, where the
        aggressive branch is exempt by design)."""
        scale = 20.0 if aggressive else 1.0
        sigma = None
        if scheme == "fbs":
            tau = self._log_uniform(1e-4, scale * 0.98 * 2.0 / L_SMOOTH_DUAL)
        elif scheme == "drs":
            tau = self._log_uniform(1e-3, 20.0)  # any a > 0 is valid anyway
        elif scheme == "dys":
            tau = self._log_uniform(1e-3, scale * 0.98 * 2.0 / L_SMOOTH_PRIMAL)
        elif scheme == "pdhg":
            tau = self._log_uniform(1e-3, 2.0)
            s_max = scale * 0.98 / (OPNORM**2 * tau)
            sigma = self._log_uniform(s_max * 1e-4, s_max)
        else:  # condat_vu: t*L_f/2 + t*s*||L||^2 < 1
            tau = self._log_uniform(1e-3, 0.98 * 2.0 / L_SMOOTH_PRIMAL)
            rem = max(1.0 - tau * L_SMOOTH_PRIMAL / 2.0, 1e-3)
            s_max = scale * 0.98 * rem / (OPNORM**2 * tau)
            sigma = self._log_uniform(s_max * 1e-4, s_max)
        return tau, sigma

    def _draw_sub(self, scheme: str, aggressive: bool = False) -> SubScheme:
        tau, sigma = self._draw_steps(scheme, aggressive=aggressive)
        return SubScheme(scheme=scheme, tau=tau, sigma=sigma)

    def _draw_wrapper_genes(self):
        """(alpha, rho, wrapper) shared by base and avg genomes."""
        alpha = (
            float(self.rng.uniform(0.0, 1.0))
            if self.rng.uniform() < self.p_alpha
            else None
        )
        rho = None
        halpern = False
        u = self.rng.uniform()
        if u < self.p_halpern:
            halpern = True
        elif u < self.p_halpern + self.p_rho:
            # deliberately allows rho*averagedness >= 1 draws; the gate
            # rejects them (rejection sampling, not clamping)
            rho = float(self.rng.uniform(0.05, 1.95))
        if halpern and self.rng.uniform() < self.p_restart:
            wrapper = Wrapper(
                halpern=True,
                restart="fixed_k",
                restart_k=int(self.rng.choice(_RESTART_KS)),
            )
        else:
            wrapper = Wrapper(halpern=halpern)
        return alpha, rho, wrapper

    def _draw_backend(self):
        """Execution-policy gene, or None (the P0 f64 path). theta drawn in
        [1e-3, 1e-1]: safely above the 1e-4 tolerance tier; the run-time
        theta > tol check catches tighter budgets."""
        if self.rng.uniform() >= self.p_backend:
            return None
        precision = str(self.rng.choice(("f64", "f32_cert64", "schedule")))
        K = int(self.rng.choice(_BACKEND_KS))
        theta = self._log_uniform(1e-3, 1e-1) if precision == "schedule" else None
        return Backend(precision=precision, K=K, theta=theta)

    def _draw_avg(self) -> Genome:
        alpha, rho, wrapper = self._draw_wrapper_genes()
        lam = float(self.rng.uniform(0.1, 0.9))
        saddle = [s for s in self.schemes if s in ("pdhg", "condat_vu")]
        if len(saddle) == 2 and self.rng.uniform() < 0.5:
            # the pdhg x condat_vu pair: identical (tau, sigma), common metric
            tau, sigma = self._draw_steps("condat_vu")  # stricter condition
            a = SubScheme("pdhg", tau, sigma)
            b = SubScheme("condat_vu", tau, sigma)
        else:
            pool = [s for s in self.schemes if s not in ("pdhg", "condat_vu")]
            scheme = str(self.rng.choice(pool or list(self.schemes)))
            if scheme in ("pdhg", "condat_vu"):
                tau, sigma = self._draw_steps(scheme)
                a = SubScheme(scheme, tau, sigma)
                b = SubScheme(scheme, tau, sigma)
            else:
                a = self._draw_sub(scheme)
                b = self._draw_sub(scheme)
        return Genome(
            scheme="mix",
            params=GenomeParams(tau=1.0, alpha=alpha, rho=rho),
            wrapper=wrapper,
            mix=Mix(kind="avg", lam=lam, a=a, b=b),
            backend=self._draw_backend(),
        )

    def _draw_switch_k(self) -> Genome:
        """Finite-prefix switch_k: certified tail + exempt aggressive
        prefix (the old residual-window "switch" is deprecated and never
        drawn)."""
        tail_scheme = str(self.rng.choice(self.schemes))
        tail = self._draw_sub(tail_scheme)
        agg_scheme = tail_scheme
        if tail_scheme in ("pdhg", "condat_vu"):
            agg_scheme = str(self.rng.choice(("pdhg", "condat_vu")))
        agg = self._draw_sub(agg_scheme, aggressive=True)
        return Genome(
            scheme="mix",
            params=GenomeParams(tau=1.0),
            wrapper=Wrapper(),
            mix=Mix(
                kind="switch_k",
                aggressive=agg,
                tail=tail,
                K=int(self.rng.choice(_SWITCH_KS)),
            ),
        )

    def _draw_metric(self):
        """Metric (rescaling) gene, or None. Only on problem kinds whose
        composite is closed under diagonal rescaling (LP cells)."""
        if self.problem_kind not in METRIC_PROBLEM_KINDS:
            return None
        if self.rng.uniform() >= self.p_metric:
            return None
        if self.rng.uniform() < 0.7:
            return Metric(kind="ruiz", iters=int(self.rng.choice(_RUIZ_ITERS)))
        return Metric(
            kind="pock_chambolle", alpha=float(self.rng.uniform(0.0, 2.0))
        )

    def _draw(self) -> Genome:
        u = self.rng.uniform()
        if u < self.p_mix:
            return self._draw_avg()
        if u < self.p_mix + self.p_switch:
            return self._draw_switch_k()
        scheme = str(self.rng.choice(self.schemes))
        tau, sigma = self._draw_steps(scheme)
        alpha, rho, wrapper = self._draw_wrapper_genes()
        if self.p_outer > 0.0:
            # consumes randomness only when enabled, so legacy seeds reproduce
            from atlas.wrappers.registry import admitted_for
            kinds = admitted_for(scheme)
            if kinds and self.rng.uniform() < self.p_outer:
                wrapper, rho = Wrapper(kind=str(self.rng.choice(kinds))), None
        return Genome(
            scheme=scheme,
            params=GenomeParams(tau=tau, sigma=sigma, alpha=alpha, rho=rho),
            wrapper=wrapper,
            backend=self._draw_backend(),
            metric=self._draw_metric(),
        )

    # -- SearchBackend -----------------------------------------------------

    def ask(self, k: int) -> List[Genome]:
        out: List[Genome] = []
        for _ in range(k):
            for _try in range(self.max_tries):
                g = self._draw()
                if validate(g, problem_kind=self.problem_kind).ok:
                    out.append(g)
                    break
                self.stage0_rejections += 1
            else:
                raise RuntimeError(
                    f"RandomBackend: no valid genome in {self.max_tries} tries"
                )
        return out

    def tell(self, genome: Genome, fitness: float, metrics: dict) -> None:
        self.history.append((genome, float(fitness), dict(metrics)))
