"""The optimization vault: the tool's growing, structured memory (joint idea with M. Pilanci's OptiVault).

An append-only JSONL store of five entry kinds. The agent consults it to decide what to build, and the router consults
it at solve time to decide which formulation, operator and device to run.

  component       an implementation of an exact operator, with its CONTRACT (the set and the input conditions it must
                  handle), its CERTIFICATE result, per-device CALIBRATION (setup and per-iteration cost), provenance
                  (hand-written, library, or synthesized in which round against which spec) and the regimes where it
                  LOSES or FAILS
  transformation  a formulation rewrite (relocation, product relocation, row equilibration, Ruiz, exact finishing,
                  device routing) with the structure detector that decides applicability
  evidence        one measurement: (instance, formulation, operator, device, accuracy target) -> iterations, setup,
                  per-iteration cost, total time, accepted
  lesson          a statement learned from a FAILURE, with the evidence that forced it and the rule it imposes; lessons
                  are checkable (a named check re-run on new evidence), not prose
  spec            a binding gap the cost model found: the budget and contract an operator (or rule, or model term)
                  must meet for a decision to flip, with the instance and device it came from and its status

Entries are never edited in place: a revision is a new entry whose `supersedes` names the old id, so the history of
what the tool believed, and why it changed, is itself part of the record.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional

KINDS = ("component", "transformation", "evidence", "lesson", "spec")


@dataclass
class Entry:
    kind: str
    id: str
    payload: dict
    provenance: dict = field(default_factory=dict)
    supersedes: Optional[str] = None
    created: float = field(default_factory=time.time)


class Vault:
    def __init__(self, path: str):
        self.path = path
        self._entries: list[Entry] = []
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line:
                    self._entries.append(Entry(**json.loads(line)))

    # ------------------------------------------------------------------ writing
    def add(self, kind: str, payload: dict, *, id: Optional[str] = None, provenance: Optional[dict] = None,
            supersedes: Optional[str] = None) -> Entry:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        e = Entry(kind=kind, id=id or f"{kind}:{uuid.uuid4().hex[:10]}", payload=payload,
                  provenance=provenance or {}, supersedes=supersedes)
        if any(x.id == e.id for x in self._entries):
            raise ValueError(f"duplicate id {e.id!r}; revise with supersedes= and a new id")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(asdict(e), default=_jsonable) + "\n")
        self._entries.append(e)
        return e

    # ------------------------------------------------------------------ reading
    def current(self, kind: Optional[str] = None) -> list[Entry]:
        """Entries not superseded by a later entry."""
        dead = {e.supersedes for e in self._entries if e.supersedes}
        return [e for e in self._entries if e.id not in dead and (kind is None or e.kind == kind)]

    def resolve(self, path: Optional[str]) -> Optional[str]:
        """Component files are stored relative to the vault directory when possible: an absolute path from another
        machine made every synthesized component raise FileNotFoundError inside the traced iteration on the H100."""
        if not path:
            return path
        if os.path.isabs(path) and os.path.exists(path):
            return path
        base = os.path.dirname(os.path.abspath(self.path))
        cand = os.path.join(base, os.path.basename(os.path.dirname(path)), os.path.basename(path)) \
            if os.path.dirname(path) else os.path.join(base, path)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(base, "components", os.path.basename(path))
        return cand2 if os.path.exists(cand2) else path

    def get(self, id: str) -> Optional[Entry]:
        return next((e for e in self._entries if e.id == id), None)

    def query(self, kind: str, where: Optional[Callable[[dict], bool]] = None) -> list[Entry]:
        return [e for e in self.current(kind) if where is None or where(e.payload)]

    def components(self, set_type: Optional[str] = None, certified_only: bool = True) -> list[Entry]:
        def ok(p):
            return ((set_type is None or p.get("set") == set_type)
                    and (not certified_only or (p.get("certificate") or {}).get("certified") == 1.0))
        return self.query("component", ok)

    def evidence(self, **match) -> list[Entry]:
        return self.query("evidence", lambda p: all(p.get(k) == v for k, v in match.items()))

    def open_specs(self) -> list[Entry]:
        return self.query("spec", lambda p: p.get("status") == "open")

    def summary(self) -> dict:
        cur = self.current()
        out = {k: sum(e.kind == k for e in cur) for k in KINDS}
        out["superseded"] = len(self._entries) - len(cur)
        return out

    def describe(self, kinds: Iterable[str] = ("component", "transformation", "lesson", "spec"), limit: int = 40) -> str:
        """Compact text view for an agent's context window."""
        lines = []
        for k in kinds:
            for e in self.current(k)[:limit]:
                p = e.payload
                if k == "component":
                    cal = p.get("calibration") or {}
                    cal_txt = "; ".join(f"{dev}: {v.get('per_call_us', '?')} us/call" for dev, v in cal.items())
                    lines.append(f"[component {e.id}] set={p.get('set')} certified="
                                 f"{(p.get('certificate') or {}).get('certified')} {cal_txt} loses={p.get('loses', [])}")
                elif k == "transformation":
                    lines.append(f"[transformation {e.id}] {p.get('name')}: {p.get('summary', '')}")
                elif k == "lesson":
                    lines.append(f"[lesson {e.id}] {p.get('statement')} RULE: {p.get('rule')}")
                elif k == "spec":
                    lines.append(f"[spec {e.id}] status={p.get('status')} {p.get('summary', '')}")
        return "\n".join(lines)


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)
