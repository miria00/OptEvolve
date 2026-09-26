"""Build an auditable status report from stored measurements, including failures."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
from summarize import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root)
    text=['# Executable discovery results',
        '',f'Generated {datetime.now(timezone.utc).isoformat()}.',
        '', 'Latest method-composition recovery and sealed confirmation: [RECOVERY_STATUS.md](RECOVERY_STATUS.md). The earlier speed-promotion experiments below use different criteria.',
        '', 'This is a recovery experiment on batched simplex least squares. Standard solver references are allowed; completed audit solver source is withheld. All performance tables below use an independent float64 output certificate.',
        '', '## Search outcomes',
        '', 'Counts describe generated programs, not independent discoveries. Related descendants share a lineage. Two seeds are a pilot, not a significance test.',
        '', '| Run / chain | Complete | Fresh evaluated | Admission passes | Programs solving both search cases | Best search geomean (s) |',
        '|---|---|---:|---:|---:|---:|']
    snapshots={}
    for name in ['pilot_v1','patch_v2_s0','patch_v2_s1','typed_v1','typed_profile_v2','typed_hybrid_random_v2']:
        if not (root/name).exists():continue
        summary=summarize(root/name);snapshots[name]=summary
        for row in summary['chains']:
            best=row['best'];score=f"{best['score']['penalized_geomean_s']:.3f}" if best else 'none'
            text.append(f"| {name}/{row['chain']} | {row['complete']} | {row['fresh_candidates']} | {row['admitted']} | {row['all_cases_solved']} | {score} |")
    text += ['', 'All failed cases incur the full search cap in fitness. Format failures and duplicates have separate recorded proposal counts. Whole-module and function-patch experiments are different treatments; do not pool their hit rates.',
             '', '## Frozen validation and test', '',
             'Sources were copied and hashed before these measurements. Candidate order is randomized within each case and repetition on the same GPU slot. Algorithm time includes setup, JIT compilation, execution, output and certificate checks. Process startup is reported separately. These are not fully cold first-visit timings.', '',
             '| Evaluation | Case | Program | Certified / repeats | Median accepted algorithm time (s) | Median worker process time (s) |',
             '|---|---|---|---:|---:|---:|']
    confirmation={}
    for name in ['confirmation_validation_v1','confirmation_test_v1','confirmation_typed_validation','confirmation_scale_final']:
        directory=root/name
        if not directory.exists():continue
        complete=(directory/'results.json').exists()
        if complete:rows=json.loads((directory/'results.json').read_text())
        else:
            rows=[]
            for path in directory.glob('raw/*/r*/*/*.json'):
                row=json.loads(path.read_text());row['label']=path.parent.name;rows.append(row)
        groups=defaultdict(list)
        for row in rows:groups[(row['case'],row['label'])].append(row)
        confirmation[name]={'complete':complete,'groups':[]}
        for (case,label),rr in sorted(groups.items()):
            accepted=[r for r in rr if r.get('accepted')]
            seconds=statistics.median(r['elapsed_s'] for r in accepted) if accepted else None
            process=[r['worker_wall_s'] for r in accepted if 'worker_wall_s' in r]
            proc=statistics.median(process) if process else None
            confirmation[name]['groups'].append({'case':case,'label':label,'accepted':len(accepted),
                'repeats':len(rr),'median_accepted_s':seconds,'median_worker_s':proc})
            ts=f'{seconds:.3f}' if seconds is not None else 'no accepted result'
            ps=f'{proc:.3f}' if proc is not None else 'n/a'
            text.append(f'| {name} ({"complete" if complete else "running"}) | {case} | {label} | {len(accepted)}/{len(rr)} | {ts} | {ps} |')
    text += ['', 'The active-set program executes NumPy linear algebra on the CPU even though it retains a GPU admission declaration. It must not be described as a GPU kernel speedup. Failed repeats remain visible; accepted-only medians do not count failures as solves.',
             '', '## Baseline and champion league']
    leagues={}
    for name in ['league_v1','league_scale_v1','league_scale_examples_v2','league_resample_v1','league_typed_v1','league_typed_hybrid_v2']:
        league=root/name/'status.json'
        if league.exists():
            status=json.loads(league.read_text());leagues[name]=status
            text += ['',f"**{name}**. Complete: {status['complete']}. Fresh candidates: {status['evaluations']}. Confirmed promotions: {len(status['promotions'])}.",
                     f"Termination: {status.get('termination','still running')}."]
        elif (root/name/'protocol.json').exists():text += ['',f'**{name}**: initial measurements or first contest in progress.']
        else:text += ['',f'**{name}**: queued for its reserved GPU.']
    text += ['', 'The patch leagues continue from the already LLM-generated active-set method. Their first promotion against ADMM does not establish improvement over that inherited method. The typed league starts from the simple FBS seed. These are different experiments. Promotion decisions and repeated opponent measurements are stored under each proposal.',
             '', 'Scale continuations have separate training pixels and a new sealed final test set. The original test was inspected during development, so final continuation claims must use the new final set.',
             '', '## Stronger reference schedule']
    reference=root/'reference_schedule_headroom/results.json'
    if reference.exists():
        text += ['', '| Program | Training case | Certified | Algorithm time (s) |', '|---|---|---|---:|']
        for row in json.loads(reference.read_text()):
            text.append(f"| {row['program']} | {row['case']} | {row.get('accepted')} | {row['elapsed_s']:.3f} |")
        text += ['', 'Single measurements establish reachability only. The schedule adapter ports device prechecks and pixel retirement from ours_v5.py to one target in the common driver. It is manually authored and is not an exact replay of the paper\'s multi-target timing protocol. It is never shown to the proposer.']
    else:text += ['', 'A separate stronger schedule comparison is pending. One initial source-copy timeout interrupted its launcher; the completed audit-reference measurements were preserved and only the remaining schedule measurements were resumed.']
    text += ['', '## What these results do and do not establish', '',
        '- The LLM writes executable algorithm changes, including accelerated projected gradient and an active-set QP method, from a simple seed plus standard references. Task-specific HSI cards were withheld for those lineages.',
        '- Numerical acceptance of arbitrary generated Python is not a convergence proof. The active-set program and its fallback behavior have not been admitted by the paper\'s constructive theorem rules. The FISTA candidate uses a power-iteration step estimate, not a rigorous spectral upper bound.',
        '- The typed experiment uses certified FBS/DRS base maps. Its retirement and handoff adapters still require separate proof obligations. Typed v1 and profile v2 preserve an earlier float32 retirement bug; the local compiler now has a regression-tested fix. Do not attribute those float32 failures solely to proposer quality.',
        '- The typed profile comparison exposes extra restriction, handoff, output and shape-change diagnostics. The arm named scalar already receives setup, iteration and certificate timing, so this is detailed versus coarse feedback, not a pure scalar-only ablation.',
        '- The later typed hybrid space adds a generic host-inverse factorization action. This is an explicit human expansion of the available implementation primitives, prompted by setup cost, not an LLM-invented operator. The new random control has identical expanded support. Earlier typed runs used a different space and must not be pooled with it.',
        '- Pixel holdouts share the same real spectral library; synthetic holdouts change dimensions and conditioning. This is not evidence across all six manuscript families.',
        '- No native CUDA/FFI synthesis, exact lasso D3–D6 ablation, model-weight RL training, or rediscovery of all six complete paper solvers is established by this experiment.',
        '- Pilot v1 seed-baseline paths could collide across workers. Later runs use per-chain/device paths and verify source hashes. Use the separate frozen confirmation measurements for performance comparisons.',
        '', '## Reproduction', '',
        'Each campaign has protocol.json, source snapshots where available, worker bundles, prompt/response records, generated source, admission results, full training measurements and lineage. Confirmation directories preserve frozen source hashes and raw repetitions. The league has independent proposal, parent and timing-order RNG streams.',
        '', 'Method and literature: [DISCOVERY_RECOVERY_20260926.md](../../docs/DISCOVERY_RECOVERY_20260926.md).']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    (root/'report_snapshot.json').write_text(json.dumps({'search':snapshots,'confirmation':confirmation,'leagues':leagues},indent=2))
    print(root/'REPORT.md')


if __name__=='__main__':main()
