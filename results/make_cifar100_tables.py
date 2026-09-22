#!/usr/bin/env python3
"""Build the CIFAR-100 tables from the imported results.

Reads `results/cifar100/{blocks,runs}` and writes, next to them:

    metrics_summary.csv / .md   mean +/- sd over the three clean D_E seeds
    metrics_per_seed.csv        one row per run
    costs_summary.csv           GPU-hours, peak memory, parameters, latency, energy
    accuracy_matrices.json      the 10x10 matrix of every clean run
    search_space.json / .md     the sampled space and every trial's D_HT score

Everything here is derived: delete the files and re-run to rebuild them.

    python results/make_cifar100_tables.py
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Block/run spelling -> the name this package uses.  Both SNV blocks exist and
# must not collide: 'snv_adaptive' is the reported SNV, and the dense SNV that
# preceded it keeps a name of its own.
RENAME = {'snv_adaptive': 'snv', 'snv': 'snv_dense', 'mcl3': 'mcl'}
METRICS = ('ACC', 'AvgAcc', 'BWT', 'FWT', 'PS', 'P', 'S', 'AF', 'HARMONIC')
COSTS = ('training_gpu_hours', 'train_minutes_per_task_mean', 'gpu_peak_allocated_bytes',
         'gpu_peak_device_used_bytes', 'energy_joules_device_counter',
         'gpu_util_percent_sample_mean')


def method_of(name):
    return RENAME.get(name, name)


def load_runs(root: Path):
    """Every clean run, named after its directory.

    ``result.json`` says ``method: snv`` for both SNV variants -- the variant is
    in ``method_variant`` -- so the directory name, which the campaign wrote
    from the block name, is what identifies the method here.
    """
    runs = []
    for path in sorted(root.glob('runs/*/result.json')):
        d = json.loads(path.read_text())
        d['_dir'] = path.parent.name
        block = path.parent.name.split('_class_il')[0].split('_task_il')[0]
        d['_method'] = method_of(block)
        runs.append(d)
    return runs


def mean_sd(values):
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=HERE / 'cifar100')
    args = p.parse_args()
    root = args.results
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f'no runs under {root}/runs')

    groups = {}
    for r in runs:
        groups.setdefault((r['_method'], r['scenario']), []).append(r)

    # ---- per seed ---------------------------------------------------------- #
    per_seed = []
    for (method, scenario), rows in sorted(groups.items()):
        for r in sorted(rows, key=lambda x: x['seed']):
            per_seed.append([method, scenario, r['seed'], r['_dir'],
                             *[round(r['metrics'].get(m), 6) if r['metrics'].get(m) is not None
                               else '' for m in METRICS]])
    write_csv(root / 'metrics_per_seed.csv',
              ['method', 'scenario', 'seed', 'run', *METRICS], per_seed)

    # ---- summary ----------------------------------------------------------- #
    summary_rows, md = [], []
    for (method, scenario), rows in sorted(groups.items()):
        row = [method, scenario, len(rows)]
        stats = {}
        for m in METRICS:
            values = [r['metrics'][m] for r in rows if r['metrics'].get(m) is not None]
            mean, sd = mean_sd(values)
            stats[m] = (mean, sd)
            row += [round(mean, 6) if mean is not None else '',
                    round(sd, 6) if sd is not None else '']
        summary_rows.append(row)
        md.append((method, scenario, stats))
    header = ['method', 'scenario', 'seeds']
    for m in METRICS:
        header += [f'{m}_mean', f'{m}_sd']
    write_csv(root / 'metrics_summary.csv', header, summary_rows)

    lines = ['# CIFAR-100 results (GTEP, clean D_E half)', '',
             'Mean +/- standard deviation over seeds 42, 43 and 44 of the selected',
             'configuration, evaluated on the held-out half (D_E) test split.',
             '`snv` is SNV-A, the reported SNV. Hyperparameters:',
             '[../../hyperparameters/cifar100_best.md](../../hyperparameters/cifar100_best.md).', '']
    for scenario in ('class_il', 'task_il'):
        rows = [(m, s) for m, sc, s in md if sc == scenario]
        if not rows:
            continue
        lines += [f'## {scenario}', '',
                  '| method | ACC | AvgAcc | BWT | FWT | PS | AF |', '|---|---:|---:|---:|---:|---:|---:|']
        for method, s in sorted(rows, key=lambda r: -(r[1]['ACC'][0] or -1)):
            def cell(key, pm=True):
                mean, sd = s[key]
                if mean is None:
                    return '-'
                return f'{mean:.4f} ± {sd:.4f}' if pm else f'{mean:.4f}'
            lines.append(f"| {method} | {cell('ACC')} | {cell('AvgAcc', False)} | "
                         f"{cell('BWT', False)} | {cell('FWT', False)} | "
                         f"{cell('PS', False)} | {cell('AF', False)} |")
        lines.append('')
    (root / 'metrics_summary.md').write_text('\n'.join(lines) + '\n')

    # ---- costs -------------------------------------------------------------- #
    cost_rows = []
    for (method, scenario), rows in sorted(groups.items()):
        row = [method, scenario]
        for key in COSTS:
            values = [r.get('cost_summary', {}).get(key) for r in rows]
            values = [v for v in values if isinstance(v, (int, float))]
            mean, _ = mean_sd(values)
            row.append(round(mean, 6) if mean is not None else '')
        final = rows[0].get('cost_summary', {}).get('final_state', {})
        row += [final.get('resident_parameter_count', ''),
                rows[0].get('checkpoint_bytes', ''),
                all(r.get('cost_summary', {}).get('gpu_exclusive_observed') for r in rows)]
        cost_rows.append(row)
    write_csv(root / 'costs_summary.csv',
              ['method', 'scenario', *COSTS, 'resident_parameters',
               'checkpoint_bytes', 'gpu_exclusive_all_seeds'], cost_rows)

    # ---- accuracy matrices --------------------------------------------------- #
    matrices = {r['_dir']: {'method': r['_method'], 'scenario': r['scenario'],
                            'seed': r['seed'], 'matrix': r['matrix']} for r in runs}
    (root / 'accuracy_matrices.json').write_text(json.dumps(matrices, indent=1) + '\n')

    # ---- search space --------------------------------------------------------- #
    import os
    import sys
    sys.path.insert(0, str(HERE.parent))
    # These results were searched under the legacy protocol, so the space is
    # read under it too -- 'paper' would print a space the campaign never used.
    os.environ['GTEP_PROTOCOL'] = 'legacy'
    import audited_gtep as G
    space = {}
    trials = {}
    for block_path in sorted(root.glob('blocks/*.json')):
        block = json.loads(block_path.read_text())
        raw, scenario = block['method'], block['scenario']
        method = method_of(raw)
        key = f'{method}/{scenario}'
        space[method] = G.SPACE.get('snv' if raw == 'snv_adaptive' else raw)
        trials[key] = [{'trial_index': t.get('trial_index', i), 'config': t['config'],
                        'D_HT_score': round(t['score'], 6)}
                       for i, t in enumerate(block.get('tuning', []))]
    (root / 'search_space.json').write_text(json.dumps(
        {'sampler': 'audited_gtep.SPACE with sample seed 7, R = 30 draws per method',
         'protocol': 'GTEP_PROTOCOL=legacy (Adam, batch 64, no schedule)',
         'selection': 'mean over seeds 42/43/44 of HARMONIC on the D_HT validation split',
         'space': space, 'trials': trials}, indent=1) + '\n')

    sl = ['# CIFAR-100 search space and every trial', '',
          'R = 30 configurations per method, drawn from the space below with sample',
          'seed 7 (`SPACE` in `audited_gtep.py`, `GTEP_PROTOCOL=legacy`), each run on',
          'D_HT with seeds 42/43/44. The winner is the highest mean HARMONIC.', '',
          '## Space', '', '| method | parameter | distribution |', '|---|---|---|']
    for method in sorted(space):
        for param, dist in (space[method] or {}).items():
            sl.append(f'| {method} | {param} | `{dist}` |')
    sl += ['', '## Trials scored on D_HT', '',
           '| method | scenario | trials | best score | worst score |',
           '|---|---|---:|---:|---:|']
    for key in sorted(trials):
        method, scenario = key.split('/')
        scores = [t['D_HT_score'] for t in trials[key]]
        if not scores:
            continue
        sl.append(f'| {method} | {scenario} | {len(scores)} | '
                  f'{max(scores):.4f} | {min(scores):.4f} |')
    sl += ['', 'Every trial with its configuration and score is in `search_space.json`.', '']
    (root / 'search_space.md').write_text('\n'.join(sl) + '\n')

    print(f'{len(runs)} runs, {len(groups)} method/scenario groups -> {root}')


if __name__ == '__main__':
    main()
