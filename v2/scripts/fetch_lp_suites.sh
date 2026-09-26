#!/usr/bin/env bash
# Fetch the real-LP benchmark suites (Mittelmann + MIPLIB 2017 benchmark
# set) into V2/data/lp_suites/{mittelmann,miplib}/ and (re)build
# MANIFEST.json. All flags are passed through to
# atlas.bench.fetch_benchmarks (see its --help), e.g.:
#     scripts/fetch_lp_suites.sh --suite miplib --max-miplib 40
#     scripts/fetch_lp_suites.sh --manifest-only
set -euo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ATLAS_PYTHON:-/home/miria/jaxenv/bin/python}"
cd "$V2_ROOT"
JAX_PLATFORMS=cpu exec "$PY" -m atlas.bench.fetch_benchmarks "$@"
