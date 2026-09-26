"""Choose the placement gene by MEASURING both devices, not by reading a flag or a spec sheet.

The goal requires CPU versus GPU to be a decision the framework makes up front, as part of building a
hardware specific algorithm for a solver family. Two measurements from 2026-09-13 say why it cannot
be a lookup:

  device     f32 TFLOPS   f64 TFLOPS   f64/f32
  RTX 4090        79.3        1.357      0.017
  H100 80GB      362.2       52.5        0.145

The 4090 is 4.6x behind the H100 at f32 but 39x behind at f64, so the right device depends on the
precision gene. And peak FLOPS is the wrong predictor anyway for these solvers, which are memory
bound: on portfolio(2000) an H100 run took 1.606 s while Clarabel on CPU took 0.009 s. Only a timed
trial on the actual problem settles it, which is what this module does.

A candidate that raises is recorded as unavailable and excluded. It is never silently replaced by
another device, because a timing attributed to the wrong device is worse than no timing.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

PLATFORM = {"cpu": "cpu", "gpu": "cuda"}


def available_devices(candidates: Sequence[str] = ("cpu", "gpu")) -> list:
    """Which placement genes this process can actually honour, in the given order."""
    import jax

    out = []
    for name in candidates:
        try:
            if jax.devices(PLATFORM[name]):
                out.append(name)
        except RuntimeError:
            pass
    return out


def choose_device(trial: Callable[[str], float], candidates: Optional[Sequence[str]] = None,
                  reps: int = 1, warmup: bool = True, log: Optional[Callable[[str], None]] = None) -> dict:
    """Time `trial(device)` on each available device and return the winner with its evidence.

    trial: runs the work on `device` and returns seconds. It should include whatever setup that
           device pays (transfer, compile), because that is the like-for-like contract used for
           baselines: each arm pays its own setup.
    reps:  timed repetitions; the median is used. A warm-up call is untimed when `warmup`.

    Returns {"device", "timings", "errors", "ratio", "candidates"}. "device" is None when no
    candidate ran, which callers must treat as "no decision", never as a default.
    """
    names = list(candidates) if candidates is not None else available_devices()
    timings, errors = {}, {}
    for name in names:
        try:
            if warmup:
                trial(name)
            ts = []
            for _ in range(max(1, reps)):
                t0 = time.perf_counter()
                trial(name)
                ts.append(time.perf_counter() - t0)
            ts.sort()
            timings[name] = ts[len(ts) // 2]
            if log:
                log(f"  device {name}: {timings[name]:.4f}s (median of {len(ts)})")
        except Exception as e:                      # unavailable or failing, never silently swapped
            errors[name] = f"{type(e).__name__}: {str(e)[:200]}"
            if log:
                log(f"  device {name}: unavailable ({errors[name][:90]})")
    if not timings:
        return {"device": None, "timings": {}, "errors": errors, "ratio": None, "candidates": names}
    best = min(timings, key=timings.get)
    others = [v for k, v in timings.items() if k != best]
    return {"device": best, "timings": timings, "errors": errors,
            "ratio": (min(others) / timings[best]) if others else None,
            "candidates": names}


def device_trial(build_and_run: Callable[[object], None]):
    """Wrap a callable that takes a jax device into the `trial(name) -> seconds` shape.

    The placement is applied with jax.default_device, and the device is resolved through the same
    refusing resolver the genome uses, so an unavailable device raises here too.
    """
    import jax
    from atlas.schemes.base_schemes import _resolve_device

    def trial(name: str) -> float:
        dev = _resolve_device(name)
        t0 = time.perf_counter()
        with jax.default_device(dev):
            build_and_run(dev)
        return time.perf_counter() - t0

    return trial
