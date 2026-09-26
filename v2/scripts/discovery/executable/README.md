# Executable solver discovery

The proposer receives standard method references, a working seed and execution
feedback. It writes function patches for eight executable layers: formulation,
algorithm, state, restriction, handoff, arithmetic contract, device and work width.
The present adapter solves batched simplex-constrained least squares. Supporting
another problem class requires its own fixed judge, data and reference notes.

The general Python path passes an API guard and small numerical admission cases.
It is not a formal proof of convergence. `typed_space.py` additionally checks the
existing FBS/DRS base-map construction rules. Adapter operations have separate
proof obligations. Neither guard is a security sandbox.

## Baseline-to-champion league

```text
standard references + parent + measured opponent
                       |
                 LLM function patch
                       |
          admission cases -> training executions
                       |
             repeated randomized contest
                       |
         promote only a certified, repeatable winner
                       |
          new champion + fixed anchor + old opponents
```

`league_campaign.py` implements this loop with frozen model weights. It saves
programs, prompts, responses, source hashes, measurements and promotion decisions.
Model sampling, parent choice and measurement order use separate RNG streams.
Baselines supply scores; their source is not shown to the proposer. A winning
LLM-generated program becomes the next parent and opponent. Independent resampling
is available as a control.

Example, on an explicitly reserved fleet slot:

```bash
env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=2 /home/miria/jaxenv2/bin/python \
  scripts/discovery/executable/league_campaign.py \
  --baselines admm=scripts/discovery/executable/audit_oracle.py \
  --data results/executable_discovery_20260926/data \
  --out results/my_fresh_league \
  --remote-root /root/my_fresh_league --slot 0 \
  --evaluations 24 --budget 30 --repeats 3 --patience 12
```

The default seed is projected gradient. `--seed-source` resumes from another
program; record that provenance rather than calling a continuation a fresh
rediscovery. `--arm resample` keeps the same seed and omits execution feedback.
`--cases train/name1,train/name2` selects a frozen training distribution. Validation
and test paths are forbidden in the search controller.

Promotion requires all outputs to pass, a repeatable geometric-mean speedup above
1.05x against the incumbent, and at most 1.20x per-case slowdown. These margins
are configurable and saved in the protocol. Termination uses the fresh-program
budget, a wall-time limit checked between contests, a patience limit, or optional
`--target-speedup` against the initial anchor. Promotion checks use training cases
only. Run `confirm.py` separately to freeze and evaluate candidates on holdouts.

## Other experiments and checks

- `campaign.py`: full-code or function-patch reference/card/feedback ablations.
- `typed_campaign.py`: LLM and random search over the same canonical composition
  support, with optional constrained decoding and detailed stage feedback.
- `make_data.py`: immutable train, validation, test and admission splits.
- `make_scale_data.py`: larger training batches excluding every originally
  allocated pixel. This creates a new experiment distribution.
- `export_preferences.py`: conservative success/nonconvergence pairs from
  identical-prompt siblings. Unchecked rationales are removed. It does not train.
- `report.py`: regenerate the recovery report from raw files, including failures.

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 \
  /home/miria/jaxenv2/bin/python -m unittest discover \
  -s scripts/discovery/executable -p 'test_*.py'
```

Runtime setup is outside the algorithm clock and included in `worker_wall_s`.
Compilation, algorithm setup, transfers, numerical work and certificates are
inside it. Report both regimes where relevant. A CPU implementation can ignore a
retained GPU declaration, so inspect actual array operations before attributing
an improvement to GPU execution.
