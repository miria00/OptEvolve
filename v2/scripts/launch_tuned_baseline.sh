#!/bin/bash
cd /home/miria/OptimizationAtlas/v2
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu /home/miria/jaxenv2/bin/python scripts/tune_baseline.py \
  --machine LOCAL --workers 16 --out results/sprint/phase2/tuned_baseline.json \
  > results/sprint/phase2/tuned_baseline_full.log 2>&1
echo $? > results/sprint/phase2/tuned_baseline.exit
touch results/sprint/phase2/tuned_baseline.DONE
