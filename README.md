# OptEvolve

The active OptEvolve code and ICLR paper live in [`v2/`](v2/).

- Solver framework: [`v2/atlas/`](v2/atlas/)
- Discovery and benchmark drivers: [`v2/scripts/`](v2/scripts/)
- Tests: [`v2/tests/`](v2/tests/)
- Paper source (the PDF is generated locally): [`v2/OptEvolve_tex_2026/`](v2/OptEvolve_tex_2026/)
- Hardware and paper notes: [`v2/docs/`](v2/docs/)

On the development machine, the previous project tree was preserved at
`/home/miria/OptEvolve_archive_20260926`. Large datasets, experiment results,
and external source checkouts are linked from `v2/` to that archive rather
than duplicated. These local links are not part of the GitHub source tree;
fresh clones need the corresponding datasets and results separately to replay
published runs.
The archived tree also retains the earlier root-level implementations and
pre-cleanup working files. This source tree retains the original Git history
and its uncommitted work; no existing research result was deleted.

To run a CPU smoke test from `v2/`:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu /home/miria/jaxenv2/bin/python -m pytest -q tests/test_schemes.py
```

See [`v2/README.md`](v2/README.md) for the full environment and workflow.
