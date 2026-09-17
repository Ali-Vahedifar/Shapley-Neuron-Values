#!/usr/bin/env python3
"""Run SNV-A through the unchanged audited GTEP worker (audited_gtep.one).

Only the SNV constructor is swapped; data splits, epoch policy, consolidation
budget, metrics, checkpointing and cost ledger are the audited ones, so a run
is directly comparable with snv_class_il_ht_* and every baseline.  The SNV-A
switches travel inside --config and are therefore recorded in result.json.

  python snv_adaptive_run.py --scenario class_il --seed 42 --out DIR \
      --config '{"lr":0.001,"truncation":0.05,"adaptive":true}'
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('GTEP_PROTOCOL', 'legacy')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audited_gtep as G
from method_loader import load

SNVAdaptive = load('SNV', 'snv_adaptive').SNVAdaptive
VARIANT_KEYS = ('task_local', 'routing', 'adaptive', 'adaptive_tol', 'adaptive_min',
                'adaptive_max', 'adaptive_grid', 'bootstrap', 'frozen_norm_eval',
                'adaptive_rule', 'adaptive_coverage', 'bn_recal', 'recal_samples',
                'bn_recal_mode', 'rot_aux')
_audited_make_method = G.make_method


def make_method(name, benchmark, device, config):
    if name != 'snv':
        return _audited_make_method(name, benchmark, device, config)
    from models import create_model
    variant = {k: config[k] for k in VARIANT_KEYS if k in config}
    # Identical to audited_gtep.make_method + baselines.build_method for 'snv'.
    method = SNVAdaptive(
        model=create_model('cifar100', 5, 10, benchmark.scenario), device=device,
        scenario=benchmark.scenario, lr=config['lr'], sparsity_ratio=0.1,
        truncation_threshold=config.get('truncation', 0.1),
        max_permutations=config.get('max_permutations', 32),     # override: smoke tests only
        shapley_eval_batches=0, selection='ablation', payoff='loss', use_mab=False,
        masked_inference=True, masked_training=True, consolidation_epochs=25,
        consolidation_within_budget=True, layer_floor=0.0, estimator_mode='reverse_tmc',
        **variant)
    method.name = 'snv_adaptive'
    return method


G.make_method = make_method

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', default='class_il')
    p.add_argument('--half', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--tasks', type=int, default=10)
    p.add_argument('--smoke_samples', type=int, default=0)
    p.add_argument('--flops', action='store_true')
    args = p.parse_args()
    args.method = 'snv'
    G.one(args)
