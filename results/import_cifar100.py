#!/usr/bin/env python3
"""Import the CIFAR-100 D_E results out of a GTEP campaign directory.

A campaign run directory is large -- a checkpoint is tens of MB and the raw
cost ledger is several MB per run -- so only what is needed to read, re-tabulate
or plot the reported numbers is kept here:

    blocks/<method>_<scenario>.json   every tuning trial, its score, the winner
    runs/<run>/result.json            the run, with the bulky per-task arrays
                                      (neuron masks, per-epoch mask logs) pruned
    runs/<run>/command.json           the exact command that produced it
    runs/<run>/train.log.gz           its training log

Checkpoints stay in the campaign directory; `checkpoints.md` says where.

    python results/import_cifar100.py --campaign DIR [--out results/cifar100]
    python results/import_cifar100.py --compact-only     # prune what is already here
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Methods with no implementation in this package (see README).
SKIP_PREFIXES = ('lwu_', 'dcnet_', 'derpp_', 'dytox_', 'icarl_', 'nfl_', 'nfl+_')
# Any list longer than this inside a history is a per-neuron or per-weight
# array: useful during a run, dead weight in an archived result.
MAX_LIST = 64


def prune(value, depth=0):
    """Drop the big per-task arrays, keep the structure and the scalars."""
    if isinstance(value, dict):
        return {k: prune(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_LIST and all(isinstance(v, (bool, int, float)) or v is None
                                         for v in value):
            return {'_pruned_array': len(value)}
        return [prune(v, depth + 1) for v in value]
    return value


def compact(result: dict) -> dict:
    out = dict(result)
    for key in ('history', 'all_task_histories'):
        if key in out:
            out[key] = prune(out[key])
    return out


def compact_file(path: Path) -> int:
    before = path.stat().st_size
    data = json.loads(path.read_text())
    path.write_text(json.dumps(compact(data)))
    return before - path.stat().st_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, help='GTEP campaign directory to import from')
    p.add_argument('--out', type=Path, default=HERE / 'cifar100')
    p.add_argument('--compact-only', action='store_true',
                   help='re-prune the result files already in --out')
    args = p.parse_args()
    out = args.out

    if args.campaign:
        (out / 'runs').mkdir(parents=True, exist_ok=True)
        (out / 'blocks').mkdir(parents=True, exist_ok=True)
        for block in sorted((args.campaign / 'blocks').glob('*.json')):
            if block.name.startswith(SKIP_PREFIXES):
                continue
            shutil.copy2(block, out / 'blocks' / block.name)
        for run in sorted((args.campaign / 'runs').glob('*clean_eval*')):
            if run.name.startswith(SKIP_PREFIXES):
                continue
            target = out / 'runs' / run.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ('result.json', 'command.json'):
                if (run / name).exists():
                    shutil.copy2(run / name, target / name)
            if (run / 'train.log').exists():
                with open(run / 'train.log', 'rb') as src, \
                        gzip.open(target / 'train.log.gz', 'wb') as dst:
                    shutil.copyfileobj(src, dst)

    saved = 0
    for result in sorted((out / 'runs').glob('*/result.json')):
        saved += compact_file(result)
    print(f'{len(list((out / "runs").glob("*/result.json")))} runs, '
          f'{saved / 1e6:.1f} MB of per-neuron arrays pruned')


if __name__ == '__main__':
    main()
