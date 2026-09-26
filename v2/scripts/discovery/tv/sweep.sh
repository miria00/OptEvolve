#!/usr/bin/env bash
# Size sweep for isotropic TV.  One device at a time: this machine has already
# discarded runs to GPU contention once.
set -u
# Clearing this is load bearing: with it set, JAX silently falls back to CPU
# or fails outright on the 4090.  The sweep must not depend on the caller.
unset LD_LIBRARY_PATH
OUT=/home/miria/OptEvolve/v2/results/tv_20260924
mkdir -p "$OUT"
DEV=$1; DT=$2; TAG=$3; BUDGET=${4:-90}
for S in 32 64 128 256 512 1024 1356; do
  F=/home/miria/data/tv/0801_${S}_noisy.npy
  [ -f "$F" ] || continue
  /home/miria/jaxenv2/bin/python \
    /home/miria/OptEvolve/v2/scripts/discovery/tv/ours_tv.py \
    --npy "$F" --device "$DEV" --dtype "$DT" --lam 0.1 --budget "$BUDGET" \
    --out "$OUT/ours_${TAG}_${S}.json" >> "$OUT/run.log" 2>&1
  echo "  ${TAG} ${S} done"
done
