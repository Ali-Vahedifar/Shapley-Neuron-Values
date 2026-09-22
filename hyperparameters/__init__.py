"""Selected hyperparameters, read from the exported tables.

    from hyperparameters import best_config
    best_config('snv', 'class_il')        # SNV-A's CIFAR-100 Class-IL winner

The JSON is the record; this module only looks things up in it.  Print the
command that reproduces a winning run with

    python -m hyperparameters snv class_il
"""
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_FILES = {'cifar100': _DIR / 'cifar100_best.json'}


def _load(dataset='cifar100'):
    if dataset not in _FILES:
        raise KeyError(f'no exported hyperparameters for {dataset!r}; '
                       f'have {sorted(_FILES)}')
    with open(_FILES[dataset]) as f:
        return json.load(f)


def entry(method, scenario, dataset='cifar100'):
    """The full record: config, search score, clean D_E metrics."""
    entries = _load(dataset)['entries']
    key = f'{method}/{scenario}'
    if key not in entries:
        raise KeyError(f'no winner for {key} on {dataset}; '
                       f'have {sorted(entries)}')
    return entries[key]


def best_config(method, scenario, dataset='cifar100'):
    """The winning configuration, ready to pass as ``--config``."""
    return entry(method, scenario, dataset)['config']


def all_entries(dataset='cifar100'):
    return _load(dataset)['entries']


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('method')
    p.add_argument('scenario', choices=['class_il', 'task_il'])
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--half', type=int, default=2, help='2 = the clean D_E half')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    rec = entry(args.method, args.scenario, args.dataset)
    config = json.dumps(rec['config'])
    runner = ('snv_adaptive_run.py' if args.method == 'snv'
              else f'audited_gtep.py --one --method {rec["block"]}')
    print(f"# {args.method} / {args.scenario} / {args.dataset}")
    print(f"# D_HT search ACC {rec['search_score_D_HT']}, "
          f"clean D_E {rec.get('clean_eval_D_E')}")
    print(f"python {runner} --scenario {args.scenario} --dataset {args.dataset} "
          f"--half {args.half} --seed {args.seed} --epochs 200 --patience 15 "
          f"--out runs/{args.method}_{args.scenario}_s{args.seed} "
          f"--config '{config}'")


if __name__ == '__main__':
    _main()
