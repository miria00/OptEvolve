#!/bin/bash
# Phase 2 LOCAL holdout sequence: WITHIN (all configs) -> medium OOD ->
# N=5 fresh-process WITHIN repeats for headline wall numbers.
set -u
cd /home/miria/OptimizationAtlas/v2
P2=results/sprint/phase2
RUN="env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu /home/miria/jaxenv2/bin/python"

# Run A: WITHIN, all 10 champions + tuned + vanilla + textbook best + per-instance rule
$RUN scripts/eval_holdout.py \
  --genomes $P2/champions_all.json \
  --tuned $P2/tuned_baseline.json \
  --no-ood --machine LOCAL --workers 16 \
  --out $P2/holdout_within_rep1.json \
  > $P2/holdout_within_rep1.log 2>&1
echo "A:$?" >> $P2/local_holdout.status

# Runs rep2..rep5: WITHIN, headline configs only (fresh process pools)
for i in 2 3 4 5; do
  $RUN scripts/eval_holdout.py \
    --genomes $P2/champions_headline.json \
    --tuned $P2/tuned_baseline.json \
    --no-ood --no-textbook-per-instance --machine LOCAL --workers 16 \
    --out $P2/holdout_within_rep$i.json \
    > $P2/holdout_within_rep$i.log 2>&1
  echo "rep$i:$?" >> $P2/local_holdout.status
done

# Run B: OOD medium tier, headline configs + baselines
$RUN scripts/eval_holdout.py \
  --genomes $P2/champions_headline.json \
  --tuned $P2/tuned_baseline.json \
  --within-n 0 --ood-smallest --ood-small-n 0 --machine LOCAL --workers 16 \
  --out $P2/holdout_ood_medium.json \
  > $P2/holdout_ood_medium.log 2>&1
echo "B:$?" >> $P2/local_holdout.status

touch $P2/local_holdout.DONE
