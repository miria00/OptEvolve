"""Promotion rules for a league of independently measured solver programs.

This is search control, not weight training. Opponents cannot change the judge,
and resampled timing intervals measure repeatability on search cases only.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
import statistics


def compare(challenger, incumbent, *, cap_s, min_speedup=1.05,
            max_case_regression=1.20, bootstrap_samples=2000, seed=4711):
    """Compare paired rows keyed by (case,repetition); require complete solutions.

    Same-case ratios remove much of the scale difference across problems. A
    fixed cap is the incumbent's penalty for failure. A challenger must pass
    every repeated independent certificate, including cases the incumbent fails.
    """
    def index(rows):
        result={}
        for row in rows:
            key=(row['case'],row['repetition'])
            if key in result:
                raise ValueError('duplicate paired measurement')
            result[key]=row
        return result
    candidate,old=index(challenger),index(incumbent)
    if not candidate or set(candidate)!=set(old):
        raise ValueError('opponents must have exactly the same paired measurements')
    if not all(r.get('accepted') for r in candidate.values()):
        return {'promote':False,'reason':'challenger failed an independent certificate'}
    by_case={}
    for key,r in candidate.items():
        o=old[key]
        t=float(r['elapsed_s'])
        baseline=float(o['elapsed_s']) if o.get('accepted') else cap_s
        if not (0<t<=cap_s and 0<baseline<=cap_s):
            raise ValueError('invalid accepted time or fixed failure penalty')
        by_case.setdefault(key[0],[]).append(math.log(baseline/t))
    if any(len(v)<3 for v in by_case.values()):
        raise ValueError('promotion requires at least three paired repetitions per case')
    means=[statistics.mean(v) for v in by_case.values()]
    log_speedup=statistics.mean(means)
    rng=random.Random(seed)
    samples=sorted(statistics.mean(statistics.mean(rng.choices(v,k=len(v)))
                                  for v in by_case.values()) for _ in range(bootstrap_samples))
    lower=math.exp(samples[int(.025*len(samples))])
    upper=math.exp(samples[min(len(samples)-1,int(.975*len(samples)))])
    case_speedup={k:math.exp(statistics.mean(v)) for k,v in by_case.items()}
    regression=any(s<1/max_case_regression for s in case_speedup.values())
    promote=lower>min_speedup and not regression
    return {'promote':promote,
            'reason':'confirmed improvement' if promote else
                ('per-case regression exceeds limit' if regression else 'improvement not repeatable above margin'),
            'speedup_geomean':math.exp(log_speedup),'bootstrap_interval':[lower,upper],
            'per_case_speedup':case_speedup,'log_reward':log_speedup,
            'scope':'timing repeatability on these fixed search cases; not generalization confidence'}


@dataclass
class StopRule:
    max_evaluations:int
    max_seconds:float
    patience:int
    target_speedup:float|None=None

    def reason(self, *, evaluations, elapsed_s, since_promotion, anchor_speedup=None):
        if evaluations>=self.max_evaluations:return 'fresh candidate budget exhausted'
        if elapsed_s>=self.max_seconds:return 'wall-time budget exhausted'
        if since_promotion>=self.patience:return 'no confirmed promotion within patience budget'
        if (self.target_speedup is not None and anchor_speedup is not None
                and anchor_speedup>=self.target_speedup):return 'fixed-anchor target reached'
        return None
