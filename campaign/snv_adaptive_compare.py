#!/usr/bin/env python3
"""Compare SNV-A pilot runs against the audited SNV trials and the baselines.

  python campaign/snv_adaptive_compare.py --runs CAMPAIGN/runs PILOT_DIR [PILOT_DIR ...]

Each PILOT_DIR holds a result.json written by snv_adaptive_run.py.  Reference
rows come from runs/<method>_class_il_ht_r*_s42 (best trial per method, seed 42
only, so the comparison uses the same task ordering as the pilots).
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

RUNS = Path('runs')      # overridden by --runs


def summary(path):
    j = json.loads(Path(path).read_text())
    m = np.array(j['matrix'], dtype=float)
    hist = j.get('all_task_histories', [])
    return dict(ACC=j['metrics']['ACC'], BWT=j['metrics']['BWT'], AvgAcc=j['metrics']['AvgAcc'],
                diag=np.diag(m).round(3).tolist(), final=m[-1].round(3).tolist(),
                selected=[h.get('selected') for h in hist],
                new=[h.get('newly_frozen') for h in hist],
                capacity=hist[-1].get('capacity_used') if hist else None,
                gpu_h=j.get('cost_summary', {}).get('training_gpu_hours'),
                config=j.get('config'))


def best_reference(method, scenario='class_il'):
    best = None
    for f in glob.glob(str(RUNS / f'{method}_{scenario}_ht_r*_s42' / 'result.json')):
        acc = json.loads(Path(f).read_text())['metrics']['ACC']
        if best is None or acc > best[1]:
            best = (f, acc)
    return best


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--runs':
        RUNS = Path(sys.argv[2])
        del sys.argv[1:3]
    print('reference (best trial, seed 42, class-IL ACC):')
    for method in ('joint', 'uniclun', 'mcl', 'lwf', 'sgd', 'snv'):
        b = best_reference(method)
        if b:
            trial = re.search(r'_ht_(r\d+)_', b[0])[1]     # no backslash inside an f-string (py3.10)
            print(f'  {method:8} {b[1]:.4f}  {trial}')
    for d in sys.argv[1:]:
        f = Path(d) / 'result.json'
        if not f.exists():
            print(f'\n{d}: no result.json yet')
            continue
        s = summary(f)
        print(f'\n{Path(d).name}  config={s["config"]}')
        print(f'  ACC {s["ACC"]:.4f}  BWT {s["BWT"]:+.4f}  AvgAcc {s["AvgAcc"]:.4f}  '
              f'capacity {s["capacity"]:.1f}%  GPU-h {s["gpu_h"]}')
        print(f'  diag  {s["diag"]}\n  final {s["final"]}')
        print(f'  |S_t| {s["selected"]}\n  new   {s["new"]}')
