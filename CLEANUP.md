# Source-tree cleanup (2026-09-26)

The active project is `/home/miria/OptEvolve/v2`. The complete previous tree,
including all 831 GB of local datasets, results, legacy code, backups, and
original Git metadata, is preserved at
`/home/miria/OptEvolve_archive_20260926`.

The replacement copies the active v2 source, tests, documentation, assets,
and paper. It omits generated Python caches, LaTeX intermediates, backup
copies, the legacy root-level implementations, and large local resources.
These retained resource paths are symlinks into the archive:

| Active path | Archived resource |
|---|---|
| `v2/data` | datasets and benchmark inputs |
| `v2/fastmri` | FastMRI data |
| `v2/results` | experiment outputs and evidence |
| `v2/skydiscover` | external checkout |
| `v2/scripts/discovery/simpletes_arm/SimpleTES` | external checkout |

The archived tree retains the original Git history. Original HEAD before
replacement: `8f6535331c628db04680ac0620a2c3d5de5ecfe2`.

After replacement, 19 focused CPU tests passed, and the paper built to a
37-page PDF. No original files were deleted: omitted files remain in the
archive.
