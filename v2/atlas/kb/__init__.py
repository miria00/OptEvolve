"""Knowledge base of classical fixed-point and splitting schemes.

The 2021-2025 GPU first-order wave (PDLP, restarts, Halpern, reflection,
Anderson acceleration) instantiates ideas published 1950-1999 for abstract
nonexpansive or monotone operators. Discovery runs through three layers: the
original paper, its general primal-dual / monotone-inclusion instantiation,
and practice on a concrete problem class. The farmable frontier is a scheme
whose general instantiation exists but whose practice on a class was not found.

Every mined or unmined claim carries its evidence: a mined claim cites where it
was found; an unmined claim records the searches that failed to find it and the
date, because "not found" is only as strong as the search behind it.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

_PATH = Path(__file__).with_name("schemes.json")
LAYERS = ("general_pd", "lp", "qp", "glm")
MINED_VALUES = (True, False, "partial", "unchecked")
REQUIRED = ("id", "name", "original", "update", "side_condition",
            "guarantee", "genome", "implementation", "gate_rule", "status", "checked")


def load() -> dict:
    return json.loads(_PATH.read_text())


def schemes() -> list[dict]:
    return load()["schemes"]


def by_id(scheme_id: str) -> dict:
    for s in schemes():
        if s["id"] == scheme_id:
            return s
    raise KeyError(scheme_id)


def implemented() -> list[str]:
    return [s["id"] for s in schemes() if s["implementation"]]


def frontier(problem_class: str) -> list[str]:
    """Schemes with a general primal-dual instantiation but no practice found
    on `problem_class` ("lp", "qp" or "glm", sparse generalized-linear-model practice)."""
    if problem_class not in LAYERS[1:]:
        raise ValueError(f"problem_class must be one of {LAYERS[1:]}")
    return [s["id"] for s in schemes()
            if s["status"]["general_pd"]["mined"] is True
            and s["status"][problem_class]["mined"] is False]


def validate() -> list[str]:
    """Structural and evidential checks. Returns a list of problems."""
    errs, seen = [], set()
    for s in schemes():
        sid = s.get("id", "?")
        for k in REQUIRED:
            if k not in s:
                errs.append(f"{sid}: missing {k}")
        if sid in seen:
            errs.append(f"{sid}: duplicate id")
        seen.add(sid)
        for layer in LAYERS:
            st = s.get("status", {}).get(layer)
            if st is None:
                errs.append(f"{sid}: status.{layer} missing"); continue
            m = st.get("mined")
            if m not in MINED_VALUES:
                errs.append(f"{sid}: status.{layer}.mined={m!r} not in {MINED_VALUES}")
            if m in (True, "partial") and not st.get("evidence"):
                errs.append(f"{sid}: status.{layer} claims mined={m!r} without evidence")
            if m is False and not st.get("queries"):
                errs.append(f"{sid}: status.{layer} claims unmined without recorded queries")
        if s.get("implementation"):
            mod, _, fn = s["implementation"].partition(":")
            if not (Path(__file__).resolve().parents[2] / mod).exists():
                errs.append(f"{sid}: implementation {mod} does not exist")
        if s.get("gate_rule"):
            mod, _, fn = s["gate_rule"].partition(":")
            try:
                m = importlib.import_module(mod.replace("/", ".").removesuffix(".py"))
                if not hasattr(m, fn):
                    errs.append(f"{sid}: gate rule {fn} not found in {mod}")
            except Exception as e:
                errs.append(f"{sid}: cannot import gate module {mod}: {e}")
    return errs
