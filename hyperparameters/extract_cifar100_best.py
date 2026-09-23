#!/usr/bin/env python3
"""Export the winning CIFAR-100 configuration of every method from a campaign.

A GTEP campaign directory holds one block per (method, scenario):

    blocks/<method>_<scenario>.json
        tuning      R configs, each scored on D_HT (half 1) over 3 seeds
        best        the winning config and its D_HT score
        evaluation  the three clean D_E (half 2) runs of that winner

This script reads the blocks, follows the evaluation runs for the reported
metrics and writes ``cifar100_best.json`` plus a readable ``cifar100_best.md``.
Re-run it after a new campaign; nothing else in the package reads a campaign
directory at run time.

    python hyperparameters/extract_cifar100_best.py \
        --campaign /path/to/gtep_campaign --out hyperparameters
"""
import argparse
import json
import statistics
from datetime import date
from pathlib import Path

# Block name -> the method name this package uses.  'snv_adaptive' is SNV-A,
# the version reported for SNV.
RENAME = {'snv_adaptive': 'snv', 'snv': 'snv_dense'}
# Methods outside this package's scope (no implementation shipped here).
SKIP = {'lwu', 'dcnet', 'derpp', 'dytox', 'icarl', 'nfl', 'nfl+', 'mcl', 'mcl3'}
METRICS = ('ACC', 'AvgAcc', 'BWT', 'FWT', 'PS', 'P', 'S', 'AF', 'HARMONIC')


def load_block(path):
    with open(path) as f:
        return json.load(f)


def eval_metrics(result_paths):
    """Mean and standard deviation over the clean D_E seeds."""
    runs = []
    for p in result_paths:
        try:
            with open(p) as f:
                runs.append(json.load(f))
        except FileNotFoundError:
            continue
    if not runs:
        return None
    out = {'seeds': [r.get('seed') for r in runs]}
    for key in METRICS:
        vals = [r['metrics'][key] for r in runs
                if r.get('metrics', {}).get(key) is not None]
        if not vals:
            continue
        out[key] = round(statistics.mean(vals), 4)
        out[key + '_std'] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    return out


def collect(campaign: Path):
    entries = {}
    for path in sorted((campaign / 'blocks').glob('*.json')):
        block = load_block(path)
        raw, scenario = block.get('method'), block.get('scenario')
        if raw in SKIP:
            continue
        best = block.get('best')
        if not best or not best.get('config'):
            continue
        method = RENAME.get(raw, raw)
        entry = {
            'method': method,
            'block': raw,
            'scenario': scenario,
            'config': best['config'],
            'search_score_D_HT': round(best['score'], 4),
            'trial_index': best.get('trial_index'),
            'trials_scored': len(block.get('tuning', [])),
            'clean_eval_D_E': eval_metrics(block.get('evaluation', [])),
        }
        if block.get('variant'):
            entry['variant'] = block['variant']
        entries[f'{method}/{scenario}'] = entry
    return entries


def to_markdown(entries, campaign, tasks):
    def fmt(cfg):
        parts = []
        for k, v in sorted(cfg.items()):
            parts.append(f'{k}={v:.6g}' if isinstance(v, float) else f'{k}={v}')
        return '`' + ', '.join(parts) + '`'

    lines = [
        '# CIFAR-100: selected hyperparameters',
        '',
        f'Source campaign: `{campaign}`  ',
        f'Exported: {date.today().isoformat()}  ',
        f'Protocol: GTEP, {tasks} tasks, ResNet-18, tuned on D_HT (half 1, 3 seeds),',
        'reported on the three clean D_E runs (half 2, seeds 42/43/44).',
        '',
        'ACC is the mean over the clean D_E seeds, +/- the standard deviation.',
        '`snv` is SNV-A (the reported SNV); `snv_dense` is the dense SNV variant,',
        'kept only to document that its Class-IL search collapsed to chance.',
        '',
        'The search space and all trials: `results/cifar100/search_space.md`.',
        'The runs these numbers come from: `results/cifar100/`.',
        '',
    ]
    for scenario in ('class_il', 'task_il'):
        rows = [e for e in entries.values() if e['scenario'] == scenario]
        if not rows:
            continue
        lines += [f'## {scenario}', '',
                  '| method | ACC (D_E) | BWT | search ACC (D_HT) | hyperparameters |',
                  '|---|---:|---:|---:|---|']
        for e in sorted(rows, key=lambda r: -(r['clean_eval_D_E'] or {}).get('ACC', -1)):
            ev = e['clean_eval_D_E'] or {}
            acc = (f"{ev['ACC']:.4f} ± {ev.get('ACC_std', 0):.4f}" if 'ACC' in ev
                   else 'not evaluated')
            bwt = f"{ev['BWT']:.4f}" if 'BWT' in ev else '-'
            lines.append(f"| {e['method']} | {acc} | {bwt} | "
                         f"{e['search_score_D_HT']:.4f} | {fmt(e['config'])} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--tasks', type=int, default=10)
    args = p.parse_args()

    entries = collect(args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset': 'cifar100',
        'tasks': args.tasks,
        'protocol': 'GTEP (disjoint halves; D_HT tunes, D_E reports)',
        'source_campaign': str(args.campaign),
        'exported': date.today().isoformat(),
        'entries': entries,
    }
    (args.out / 'cifar100_best.json').write_text(json.dumps(payload, indent=2) + '\n')
    (args.out / 'cifar100_best.md').write_text(
        to_markdown(entries, args.campaign, args.tasks) + '\n')
    print(f'{len(entries)} (method, scenario) winners -> {args.out}')


if __name__ == '__main__':
    main()
