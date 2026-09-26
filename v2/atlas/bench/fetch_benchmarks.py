"""Fetchers for the real-LP benchmark suites + MANIFEST builder.

Sources:
  (a) Mittelmann LP benchmark instances, https://plato.asu.edu/ftp/lptestset/
      (top level + network/ + fctp/ subdirectories). Only *.mps.bz2 files
      are fetched; plain *.bz2 files there are MPC-compressed (need the
      netlib EMPS utility, see 00README) and are logged as skips, not
      failures. Downloads are re-compressed bz2 -> gz so highspy can read
      the stored file directly.
  (b) MIPLIB 2017 benchmark-set LP relaxations, https://miplib.zib.de
      (the 240-instance benchmark set listed on tag_benchmark.html;
      per-instance files at WebData/instances/<name>.mps.gz). Selection:
      nonzero count (from the site's table) below --miplib-max-nnz
      (default 2e6, the SMALL/MEDIUM tier), smallest first, up to
      --max-miplib instances. The integrality section is IGNORED at load
      time by atlas.bench.mps_cell (= LP relaxation).

Every download: 60 s per-file timeout (configurable), size caps per file
and in total (~2 GB default), sha256 recorded, every skip/failure logged
to <data_root>/fetch_log.json and stdout. Existing files are not
re-downloaded (delete to force).

MANIFEST.json (schema): {"created", "instances": [{name, file, rows,
cols, nnz, source, url, sha256, bytes, holdout_tier, usable,
skip_reason}]} where rows/cols/nnz describe the ORIGINAL model as read
by highspy (not the standard-form conversion, whose shape depends on the
transforms; atlas.bench.mps_cell reports that at load time), and
holdout_tier is "small" (nnz <= 1e5), "medium" (<= miplib-max-nnz cap),
or "large" (above the cap; stored but excluded from default builds).

Usage:
    python -m atlas.bench.fetch_benchmarks [--suite both|mittelmann|miplib]
        [--manifest-only] [--max-miplib N] [--timeout S] ...
    (or scripts/fetch_lp_suites.sh from the v2 root.)
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

PLATO_BASE = "https://plato.asu.edu/ftp/lptestset/"
PLATO_SUBDIRS = ("", "network/", "fctp/")
MIPLIB_TABLE_URL = "https://miplib.zib.de/tag_benchmark.html"
MIPLIB_INSTANCE_URL = "https://miplib.zib.de/WebData/instances/{name}.mps.gz"

_V2_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_ROOT = os.path.join(_V2_ROOT, "data", "lp_suites")

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_FILE_CAP = 25 * 1024 * 1024  # compressed bytes per Mittelmann file
DEFAULT_TOTAL_CAP = 2 * 1024 * 1024 * 1024  # ~2 GB across both suites
SMALL_TIER_NNZ = 100_000
DEFAULT_MIPLIB_MAX_NNZ = 2_000_000
DEFAULT_MAX_MIPLIB = 60

_SIZE_RE = re.compile(
    r'<a href="(?P<name>[^"?/][^"]*)">[^<]*</a></td><td[^>]*>[^<]*</td>'
    r'<td align="right">\s*(?P<size>[0-9.]+[KMG]?)'
)


def _human_to_bytes(s: str) -> int:
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(float(s))


def _download(url: str, dest: str, timeout_s: float) -> int:
    """Stream url -> dest with a wall-clock timeout covering the WHOLE
    transfer (not just connect). Returns byte count; raises on failure
    (partial files are removed)."""
    t0 = time.monotonic()
    part = dest + ".part"
    n = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "atlas-bench/0.1"})
        with urllib.request.urlopen(req, timeout=min(timeout_s, 30.0)) as resp, open(
            part, "wb"
        ) as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                n += len(chunk)
                if time.monotonic() - t0 > timeout_s:
                    raise TimeoutError(
                        f"transfer exceeded {timeout_s:.0f}s ({n} bytes in)"
                    )
        os.replace(part, dest)
        return n
    except BaseException:
        if os.path.exists(part):
            os.unlink(part)
        raise


def _bz2_to_gz(src_bz2: str, dest_gz: str) -> None:
    with bz2.open(src_bz2, "rb") as fin, gzip.open(dest_gz, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FetchLog:
    def __init__(self):
        self.events: list[dict] = []

    def log(self, kind: str, **kw):
        rec = {"kind": kind, **kw}
        self.events.append(rec)
        detail = " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[{kind}] {detail}", flush=True)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"created": datetime.now(timezone.utc).isoformat(), "events": self.events},
                f,
                indent=1,
            )


class Budget:
    """Total-download-byte budget shared across both suites."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.used = 0

    def allows(self, nbytes: int) -> bool:
        return self.used + nbytes <= self.cap

    def charge(self, nbytes: int):
        self.used += nbytes


# ---------------------------------------------------------------------------
# (a) Mittelmann / plato.asu.edu
# ---------------------------------------------------------------------------


def list_plato(timeout_s: float, log: FetchLog) -> list[dict]:
    """Scrape the lptestset directory listings. Returns candidate dicts
    {name, url, subdir, approx_bytes, mps_format} for every *.bz2 file."""
    out = []
    for sub in PLATO_SUBDIRS:
        url = PLATO_BASE + sub
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "atlas-bench/0.1"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                html = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log.log("listing_failed", url=url, error=repr(e))
            continue
        for m in _SIZE_RE.finditer(html):
            name = m.group("name")
            if not name.endswith(".bz2"):
                continue
            out.append(
                {
                    "name": name,
                    "url": url + name,
                    "subdir": sub.strip("/"),
                    "approx_bytes": _human_to_bytes(m.group("size")),
                    "mps_format": name.endswith(".mps.bz2"),
                }
            )
    return out


def fetch_mittelmann(
    dest_dir: str,
    budget: Budget,
    log: FetchLog,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    file_cap: int = DEFAULT_FILE_CAP,
    max_instances: Optional[int] = None,
) -> list[dict]:
    """Fetch reachable Mittelmann *.mps.bz2 instances; store as .mps.gz.
    Returns [{name, file, url, source}] for files present on disk."""
    os.makedirs(dest_dir, exist_ok=True)
    got: list[dict] = []
    cands = list_plato(timeout_s, log)
    cands.sort(key=lambda c: c["approx_bytes"])
    for c in cands:
        stem = c["name"][: -len(".mps.bz2")] if c["mps_format"] else c["name"][:-4]
        rec = {
            "name": stem,
            "file": os.path.join("mittelmann", stem + ".mps.gz"),
            "url": c["url"],
            "source": "mittelmann",
        }
        dest = os.path.join(dest_dir, stem + ".mps.gz")
        if not c["mps_format"]:
            log.log(
                "skip", suite="mittelmann", name=c["name"],
                reason="mpc-format (plain .bz2 needs netlib EMPS; see 00README)",
            )
            continue
        if os.path.exists(dest):
            log.log("cached", suite="mittelmann", name=stem)
            got.append(rec)
            continue
        if c["approx_bytes"] > file_cap:
            log.log(
                "skip", suite="mittelmann", name=c["name"],
                reason=f"over per-file cap ({c['approx_bytes']} > {file_cap} bytes)",
            )
            continue
        if not budget.allows(c["approx_bytes"]):
            log.log(
                "skip", suite="mittelmann", name=c["name"],
                reason=f"total download cap reached ({budget.used}/{budget.cap})",
            )
            continue
        if max_instances is not None and len(got) >= max_instances:
            break
        tmp_bz2 = dest + ".bz2.tmp"
        try:
            n = _download(c["url"], tmp_bz2, timeout_s)
            budget.charge(n)
            _bz2_to_gz(tmp_bz2, dest)
            log.log("fetched", suite="mittelmann", name=stem, bytes=n)
            got.append(rec)
        except (urllib.error.URLError, OSError, TimeoutError, EOFError) as e:
            log.log("failed", suite="mittelmann", name=c["name"], error=repr(e))
        finally:
            if os.path.exists(tmp_bz2):
                os.unlink(tmp_bz2)
    return got


# ---------------------------------------------------------------------------
# (b) MIPLIB 2017 benchmark set (LP relaxations at load time)
# ---------------------------------------------------------------------------


def list_miplib(timeout_s: float, log: FetchLog) -> list[dict]:
    """Parse tag_benchmark.html into [{name, nnz_site, url}]."""
    try:
        req = urllib.request.Request(
            MIPLIB_TABLE_URL, headers={"User-Agent": "atlas-bench/0.1"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            html = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        log.log("listing_failed", url=MIPLIB_TABLE_URL, error=repr(e))
        return []
    out = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    for tr in rows:
        mname = re.search(r'instance_details_([^"]+)\.html', tr)
        if not mname:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        # columns: Instance, Status, Vars, Bins, Ints, Cont, Constraints,
        # Nonz., Submitter, Group, Objective, Tags
        if len(tds) < 8:
            continue
        nnz_txt = re.sub(r"<[^>]+>", "", tds[7]).strip().replace(",", "")
        try:
            nnz = int(nnz_txt)
        except ValueError:
            continue
        name = mname.group(1)
        out.append(
            {"name": name, "nnz_site": nnz, "url": MIPLIB_INSTANCE_URL.format(name=name)}
        )
    return out


def fetch_miplib(
    dest_dir: str,
    budget: Budget,
    log: FetchLog,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_nnz: int = DEFAULT_MIPLIB_MAX_NNZ,
    max_instances: int = DEFAULT_MAX_MIPLIB,
) -> list[dict]:
    """Fetch benchmark-set instances with site-reported nnz < max_nnz,
    smallest first, up to max_instances. Stored as .mps.gz verbatim."""
    os.makedirs(dest_dir, exist_ok=True)
    got: list[dict] = []
    cands = list_miplib(timeout_s, log)
    log.log("miplib_table", n_listed=len(cands))
    kept = [c for c in cands if c["nnz_site"] < max_nnz]
    for c in cands:
        if c["nnz_site"] >= max_nnz:
            log.log(
                "skip", suite="miplib", name=c["name"],
                reason=f"site nnz {c['nnz_site']} >= cap {max_nnz} (large tier)",
            )
    kept.sort(key=lambda c: c["nnz_site"])
    for c in kept:
        if len(got) >= max_instances:
            log.log("stop", suite="miplib", reason=f"max_instances={max_instances} reached")
            break
        rec = {
            "name": c["name"],
            "file": os.path.join("miplib", c["name"] + ".mps.gz"),
            "url": c["url"],
            "source": "miplib2017-benchmark",
        }
        dest = os.path.join(dest_dir, c["name"] + ".mps.gz")
        if os.path.exists(dest):
            log.log("cached", suite="miplib", name=c["name"])
            got.append(rec)
            continue
        if not budget.allows(4 * c["nnz_site"] + (1 << 16)):  # rough size guess
            log.log(
                "skip", suite="miplib", name=c["name"],
                reason=f"total download cap reached ({budget.used}/{budget.cap})",
            )
            continue
        try:
            n = _download(c["url"], dest, timeout_s)
            budget.charge(n)
            log.log("fetched", suite="miplib", name=c["name"], bytes=n)
            got.append(rec)
        except (urllib.error.URLError, OSError, TimeoutError, EOFError) as e:
            log.log("failed", suite="miplib", name=c["name"], error=repr(e))
    return got


# ---------------------------------------------------------------------------
# MANIFEST builder
# ---------------------------------------------------------------------------


def build_manifest(
    data_root: str,
    fetched: list[dict],
    log: FetchLog,
    medium_cap_nnz: int = DEFAULT_MIPLIB_MAX_NNZ,
) -> dict:
    """Read every stored file with highspy, record true rows/cols/nnz,
    sha256, holdout_tier; mark unreadable files unusable with the reason."""
    from atlas.bench.mps_cell import read_mps

    instances = []
    for rec in fetched:
        path = os.path.join(data_root, rec["file"])
        entry = dict(rec)
        entry["bytes"] = os.path.getsize(path)
        entry["sha256"] = _sha256(path)
        try:
            raw = read_mps(path)
            entry.update(
                rows=int(raw.M.shape[0]),
                cols=int(raw.M.shape[1]),
                nnz=int(raw.nnz),
                n_integer=int(raw.n_integer),
                maximize=bool(raw.maximize),
                usable=True,
                skip_reason=None,
            )
            nnz = entry["nnz"]
            entry["holdout_tier"] = (
                "small" if nnz <= SMALL_TIER_NNZ
                else "medium" if nnz <= medium_cap_nnz
                else "large"
            )
        except Exception as e:  # noqa: BLE001 - manifest must record any failure
            entry.update(
                rows=None, cols=None, nnz=None, usable=False,
                skip_reason=f"highspy read failed: {e!r}", holdout_tier=None,
            )
            log.log("unusable", name=rec["name"], error=repr(e))
        instances.append(entry)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "small_tier_nnz": SMALL_TIER_NNZ,
        "medium_tier_nnz": medium_cap_nnz,
        "instances": instances,
    }
    return manifest


def summarize(manifest: dict) -> str:
    inst = manifest["instances"]
    by = {}
    for e in inst:
        key = (e["source"], e.get("holdout_tier"), e.get("usable"))
        by[key] = by.get(key, 0) + 1
    lines = [f"{len(inst)} instances total"]
    for (src, tier, usable), n in sorted(by.items(), key=lambda kv: str(kv[0])):
        lines.append(f"  {src:24s} tier={str(tier):7s} usable={usable}: {n}")
    total_b = sum(e["bytes"] for e in inst)
    lines.append(f"  stored bytes: {total_b / 1e6:.1f} MB")
    return "\n".join(lines)


def _scan_disk(data_root: str) -> list[dict]:
    """Enumerate every stored instance file (both suites) as fetch-style
    records. The manifest is ALWAYS built from this scan, so a partial
    fetch (--suite miplib) cannot drop the other suite's entries."""
    out = []
    for sub, src in (("mittelmann", "mittelmann"), ("miplib", "miplib2017-benchmark")):
        d = os.path.join(data_root, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith((".mps", ".mps.gz")):
                name = fn[:-7] if fn.endswith(".mps.gz") else fn[:-4]
                out.append(
                    {"name": name, "file": os.path.join(sub, fn),
                     "url": None, "source": src}
                )
    return out


def _default_url(rec: dict) -> Optional[str]:
    if rec["source"] == "miplib2017-benchmark":
        return MIPLIB_INSTANCE_URL.format(name=rec["name"])
    return None  # Mittelmann files live in several subdirs; keep fetch record


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", choices=["both", "mittelmann", "miplib"], default="both")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--file-cap", type=int, default=DEFAULT_FILE_CAP,
                    help="per-file compressed byte cap (Mittelmann)")
    ap.add_argument("--total-cap", type=int, default=DEFAULT_TOTAL_CAP)
    ap.add_argument("--max-miplib", type=int, default=DEFAULT_MAX_MIPLIB)
    ap.add_argument("--miplib-max-nnz", type=int, default=DEFAULT_MIPLIB_MAX_NNZ)
    ap.add_argument("--max-mittelmann", type=int, default=None)
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild MANIFEST.json from files already on disk")
    args = ap.parse_args(argv)

    os.makedirs(args.data_root, exist_ok=True)
    log = FetchLog()
    budget = Budget(args.total_cap)
    fetched: list[dict] = []

    if not args.manifest_only:
        if args.suite in ("both", "mittelmann"):
            fetched += fetch_mittelmann(
                os.path.join(args.data_root, "mittelmann"), budget, log,
                timeout_s=args.timeout, file_cap=args.file_cap,
                max_instances=args.max_mittelmann,
            )
        if args.suite in ("both", "miplib"):
            fetched += fetch_miplib(
                os.path.join(args.data_root, "miplib"), budget, log,
                timeout_s=args.timeout, max_nnz=args.miplib_max_nnz,
                max_instances=args.max_miplib,
            )

    # The manifest always covers everything on disk (a partial --suite run
    # must not drop the other suite). URLs come from this run's fetch
    # records, falling back to the previous manifest, then to the known
    # per-instance URL pattern.
    mpath = os.path.join(args.data_root, "MANIFEST.json")
    url_by_file = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            for e in json.load(f).get("instances", []):
                if e.get("url"):
                    url_by_file[e["file"]] = e["url"]
    for r in fetched:
        if r.get("url"):
            url_by_file[r["file"]] = r["url"]
    records = _scan_disk(args.data_root)
    for r in records:
        r["url"] = url_by_file.get(r["file"]) or _default_url(r)

    manifest = build_manifest(args.data_root, records, log,
                              medium_cap_nnz=args.miplib_max_nnz)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    log.save(os.path.join(args.data_root, "fetch_log.json"))
    print(f"\nMANIFEST -> {mpath}")
    print(summarize(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
