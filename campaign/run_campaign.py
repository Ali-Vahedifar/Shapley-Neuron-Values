#!/usr/bin/env python3
"""Run the CIFAR-100 GTEP campaign: every method, both scenarios, tuning then
clean evaluation.

Protocol (identical for every method, see docs/PROTOCOL.md):

* CIFAR-100 is split into two disjoint halves of 50 classes (split seed 1234).
  Half 1 (``D_HT``) selects hyper-parameters, half 2 (``D_E``) produces the
  reported numbers.  Each half is 10 tasks of 5 classes.
* R = 30 configurations are drawn per method from its search space with sample
  seed 7, and each is run on seeds 42/43/44 of ``D_HT``.  The winner is the
  configuration with the highest mean HARMONIC score (harmonic mean of final
  ACC and average-over-tasks ACC on the D_HT validation split).
* The winner is then run three times on ``D_E``, one run at a time on an
  otherwise idle GPU, so the published cost numbers
  (GPU-hours, peak memory, GFLOPs, inference time) are measured exclusively.
  A D_E result whose ledger did not observe an exclusive GPU is rejected.
* SNV means SNV-A (``SNV/snv_adaptive.py``), driven through the unchanged
  audited worker by ``snv_adaptive_run.py``; the SNV-A switches travel inside
  the config and are recorded in every ``result.json``.

The driver is resumable: a finished run whose ``result.json`` matches its
identity (method, scenario, half, seed, config, 10 tasks, 200-epoch policy) is
reused, so re-running the command after an interruption continues the campaign.

  python campaign/run_campaign.py --out runs/cifar100 --gpus 0 1 2 3
  python campaign/run_campaign.py --out runs/cifar100 --gpus 0 --methods snv lwf
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GTEP_PROTOCOL', 'legacy')
import audited_gtep as G                                            # noqa: E402

METHODS = ['sgd', 'joint', 'ewc', 'si', 'lwf', 'wsn', 'pec', 'spacenet',
           'nispa', 'uniclun', 'snv']
SEEDS = (42, 43, 44)
ROUNDS = 30
EPOCHS = 200
PATIENCE = 15
TASKS = 10
SAMPLE_SEED = 7

# SNV-A switches (SNV/snv_adaptive.py).  Fixed across the search, recorded in
# every config so a run file states the variant it measured.  Routing is the
# one scenario-dependent switch: Task-IL is given the task identity, so the
# subnetwork's own head decides (maxprob), while Class-IL has to pick the
# subnetwork first and routes on rotational energy.  Selecting a single routing
# rule for both is what collapsed the earlier Class-IL sweep to chance.
SNV_VARIANT = dict(task_local=True, frozen_norm_eval=True, bn_recal=True,
                   bn_recal_mode='batch', adaptive=True, adaptive_rule='coverage',
                   adaptive_coverage=0.9)
SNV_ROUTING = {'task_il': dict(routing='maxprob', rot_aux=0.0),
               'class_il': dict(routing='rot_energy_z', rot_aux=1.0)}


def snv_variant(scenario):
    return {**SNV_VARIANT, **SNV_ROUTING[scenario]}


def scenarios(method):
    """PEC is Class-IL only, WSN needs the task identity so it is Task-IL only."""
    return (['class_il'] if method == 'pec' else
            ['task_il'] if method == 'wsn' else ['class_il', 'task_il'])


# SNV estimates its neuron values on the network the task left behind, so a
# draw whose learning rate cannot move that network inside the epoch budget
# cannot produce a usable subnetwork either.  Those draws are dropped instead of
# spending three seeds each on them; the survivors keep their original index, so
# run directory ht_r6 is the seventh draw of the shared sample, as in the
# campaign behind results/cifar100.
SNV_MIN_LR = 3e-4


def configs_for(method, scenario):
    """[(trial index, config)] for a method, in draw order."""
    rng = random.Random(SAMPLE_SEED)
    draws = [G.sample(G.SPACE[method], rng) for _ in range(ROUNDS)]
    if method != 'snv':
        return list(enumerate(draws))
    return [(i, {**d, **snv_variant(scenario)})
            for i, d in enumerate(draws) if d['lr'] >= SNV_MIN_LR]


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def valid_result(path, method, scenario, half, seed, config, tasks=TASKS, epochs=EPOCHS):
    """Return the result only if it is this exact job, run under this protocol."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except ValueError:
        return None
    ok = (d.get('complete') and d.get('completed_tasks') == tasks
          and d.get('method') == method and d.get('scenario') == scenario
          and d.get('half') == half and d.get('seed') == seed and d.get('config') == config
          and d.get('training_policy', {}).get('max_epochs') == epochs)
    return d if ok else None


def cost_row(method, scenario, result_file, selected_config):
    d = json.loads(Path(result_file).read_text())
    cost = d['cost_summary']
    if not cost['gpu_exclusive_observed']:
        raise RuntimeError(f'Cost run was not observed GPU-exclusive: {result_file}')
    inference = [x for x in d['inference'] if x['batch_size'] == 64]
    mean = lambda key: (sum(x[key] for x in inference) / len(inference)) if inference else None
    return dict(method=method, scenario=scenario, seed=d['seed'],
                **{k: v for k, v in d['metrics'].items() if k != 'I'},
                train_minutes_per_task=cost['train_minutes_per_task_mean'],
                gpu_hours=cost['accounted_run_gpu_hours'],
                peak_gpu_MB=(cost['gpu_peak_allocated_bytes'] or 0) / 1e6 or None,
                model_MB=cost['final_state']['resident_parameter_bytes'] / 1e6,
                gpu_util_percent=cost['gpu_util_percent_sample_mean'],
                checkpoint_bytes=d['checkpoint_bytes'],
                infer_ms_per_sample=mean('inference_ms_per_sample'),
                GFLOPs_per_sample=mean('supported_operator_gflops_per_sample'),
                selected_config=json.dumps(selected_config, sort_keys=True),
                result=str(result_file))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='campaign directory (created, resumable)')
    p.add_argument('--gpus', type=int, nargs='+', required=True,
                   help='GPU indices in nvidia-smi order; runs are pinned by UUID')
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--scenarios', nargs='+', default=['class_il', 'task_il'],
                   choices=['class_il', 'task_il'])
    p.add_argument('--rounds', type=int, default=ROUNDS)
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--tasks', type=int, default=0,
                   help='0 = the protocol task count for the dataset')
    p.add_argument('--dataset', default='cifar100', choices=sorted(G.DATASET_CLASSES),
                   help='CIFAR-100 is the tuned benchmark; the others are wired up but untuned')
    p.add_argument('--backbone', default=None)
    p.add_argument('--poll', type=float, default=5.0)
    p.add_argument('--plan', action='store_true',
                   help='print the jobs this campaign would run, then exit without training')
    args = p.parse_args()
    if not args.tasks:
        args.tasks = G.task_count(args.dataset)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not G.GPUS:
        raise RuntimeError('no GPUs visible; set GTEP_GPU_UUIDS')
    for gpu in args.gpus:
        if gpu >= len(G.GPUS):
            raise RuntimeError(f'GPU index {gpu} out of range; {len(G.GPUS)} visible')

    # One campaign per directory.
    lock = (out / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    fingerprint = G.source_fingerprint()
    classes_per_half, classes_per_task = G.geometry(args.dataset, args.tasks)
    protocol = dict(dataset=args.dataset, split='disjoint halves',
                    classes_per_half=classes_per_half, tasks=args.tasks,
                    classes_per_task=classes_per_task, D_HT=1, D_E=2, split_seed=1234,
                    seeds=list(SEEDS),
                    R=args.rounds, sample_seed=SAMPLE_SEED, max_total_epochs=args.epochs,
                    patience=args.patience, gtep_protocol=os.environ['GTEP_PROTOCOL'],
                    selection='mean over seeds of the harmonic mean of final ACC and AvgAcc on D_HT validation',
                    costs='tuning runs share GPUs and their timings are not reported; D_E winner runs '
                          'take the GPU exclusively and carry the published cost numbers',
                    snv_variant={s: snv_variant(s) for s in SNV_ROUTING},
                    methods=args.methods,
                    spaces={m: G.SPACE[m] for m in args.methods},
                    source_sha256=fingerprint)
    protocol = G.serial(protocol)
    if (out / 'protocol.json').exists():
        previous = json.loads((out / 'protocol.json').read_text())
        if previous['source_sha256'] != protocol['source_sha256']:
            raise RuntimeError('training source changed since this campaign started; '
                               'start a new --out directory instead of mixing code versions')
    atomic(out / 'protocol.json', protocol)

    def verify_source():
        if G.source_fingerprint() != fingerprint:
            raise RuntimeError('training source changed while the campaign was running')

    jobs, blocks, running, failed = [], [], {}, []

    def add_job(method, scenario, half, seed, config, tag):
        name = f'{method}_{scenario}_{tag}_s{seed}'
        job = dict(method=method, scenario=scenario, half=half, seed=seed, config=config,
                   directory=out / 'runs' / name, state='pending', priority=len(jobs))
        if valid_result(job['directory'] / 'result.json', method, scenario, half, seed, config,
                        args.tasks, args.epochs):
            job['state'] = 'complete'
        jobs.append(job)
        return job

    def add_block(method, scenario):
        indexed = configs_for(method, scenario)[:args.rounds]
        block = dict(method=method, scenario=scenario, configs=dict(indexed), winner_jobs=None,
                     done=False, signature=None, trial_indices=[i for i, _ in indexed],
                     min_lr=SNV_MIN_LR if method == 'snv' else None,
                     path=out / 'blocks' / f'{method}_{scenario}.json')
        block['trials'] = [[add_job(method, scenario, 1, seed, config, f'ht_r{i}') for seed in SEEDS]
                           for i, config in indexed]
        blocks.append(block)

    def refresh():
        for block in blocks:
            complete = [dict(trial_index=i, config=block['configs'][i],
                             score=sum(json.loads((j['directory'] / 'result.json').read_text())
                                       ['metrics']['HARMONIC'] for j in trial) / len(SEEDS),
                             results=[str(j['directory'] / 'result.json') for j in trial])
                        for i, trial in zip(block['trial_indices'], block['trials'])
                        if all(j['state'] == 'complete' for j in trial)]
            best = max(complete, key=lambda t: t['score']) if complete else None
            if len(complete) == len(block['trials']) and block['winner_jobs'] is None:
                block['winner_jobs'] = [add_job(block['method'], block['scenario'], 2, seed,
                                                best['config'], 'clean_eval') for seed in SEEDS]
            evaluation = [str(j['directory'] / 'result.json')
                          for j in block['winner_jobs'] or [] if j['state'] == 'complete']
            block['done'] = (len(complete) == len(block['trials']) and len(evaluation) == len(SEEDS))
            signature = (len(complete), len(evaluation))
            if signature != block['signature']:
                origin = f'search space, sample seed {SAMPLE_SEED}'
                if block['min_lr']:
                    origin += (f"; draws with lr < {block['min_lr']} dropped, indices kept")
                atomic(block['path'], dict(method=block['method'], scenario=block['scenario'],
                                           config_origin=origin, min_lr=block['min_lr'],
                                           trials=block['trial_indices'],
                                           variant=(snv_variant(block['scenario'])
                                                    if block['method'] == 'snv' else None),
                                           tuning=complete, best=best, evaluation=evaluation))
                block['signature'] = signature

    def launch(job, gpu):
        """Take the GPU's exclusive lock, check it is idle, then start the run."""
        verify_source()
        lease = open(f'/tmp/gtep-{G.GPUS[gpu]}.lock', 'w')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            return False
        if gpu_busy(gpu):
            lease.close()
            return False
        directory = Path(job['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        worker = 'snv_adaptive_run.py' if job['method'] == 'snv' else 'audited_gtep.py'
        command = [sys.executable, '-u', str(ROOT / worker)]
        if job['method'] != 'snv':
            command += ['--one', '--method', job['method']]
        command += ['--scenario', job['scenario'], '--half', str(job['half']), '--seed', str(job['seed']),
                    '--config', json.dumps(job['config']), '--out', str(directory),
                    '--epochs', str(args.epochs), '--patience', str(args.patience),
                    '--tasks', str(args.tasks), '--dataset', args.dataset]
        if args.backbone:
            command += ['--backbone', args.backbone]
        # No --flops here: FLOP instrumentation would run inside the timed
        # training phases.  Inference GFLOPs come from the worker's separate,
        # untimed pass; training FLOPs need their own --flops run (docs/COSTS.md).
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=G.GPUS[gpu], GTEP_PROTOCOL=os.environ['GTEP_PROTOCOL'],
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        atomic(directory / 'command.json', dict(command=command, gpu=G.GPUS[gpu],
               cost_policy=('exclusive GPU; clean training timing; separate inference FLOP pass'
                            if job['half'] == 2 else 'tuning run; timings not reported')))
        log = (directory / 'train.log').open('a')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        job.update(state='running', gpu=gpu)
        running[gpu] = dict(job=job, process=process, lease=lease, log=log)
        print('START', directory.name, 'GPU', gpu, flush=True)
        return True

    try:
        import pynvml
        pynvml.nvmlInit()
        handles = {gpu: pynvml.nvmlDeviceGetHandleByUUID(G.GPUS[gpu]) for gpu in args.gpus}

        def gpu_busy(gpu):
            return bool(pynvml.nvmlDeviceGetComputeRunningProcesses(handles[gpu]))
    except Exception:                                   # pynvml is optional
        print('pynvml unavailable: relying on the per-GPU lock alone', flush=True)

        def gpu_busy(gpu):
            return False

    for method in args.methods:
        for scenario in scenarios(method):
            if scenario in args.scenarios:
                add_block(method, scenario)
    if args.plan:
        # What the campaign would run, without touching a GPU: the worker each
        # method goes through, its trial indices and its run directories.
        plan = []
        for block in blocks:
            worker = ('snv_adaptive_run.py' if block['method'] == 'snv'
                      else 'audited_gtep.py --one --method ' + block['method'])
            plan.append(dict(method=block['method'], scenario=block['scenario'], worker=worker,
                             trials=block['trial_indices'], min_lr=block['min_lr'],
                             tuning_runs=[j['directory'].name for t in block['trials'] for j in t],
                             clean_eval_runs=[f"{block['method']}_{block['scenario']}"
                                              f"_clean_eval_s{seed}" for seed in SEEDS],
                             variant=(snv_variant(block['scenario'])
                                      if block['method'] == 'snv' else None)))
        tuning = sum(len(b['tuning_runs']) for b in plan)
        clean = sum(len(b['clean_eval_runs']) for b in plan)
        atomic(out / 'plan.json', dict(dataset=args.dataset, protocol=protocol, blocks=plan,
                                       tuning_runs=tuning, clean_eval_runs=clean))
        for b in plan:
            print(f"{b['method']:9s} {b['scenario']:9s} {len(b['trials']):3d} trials x "
                  f"{len(SEEDS)} seeds -> {len(b['tuning_runs']):4d} tuning runs, "
                  f"{len(b['clean_eval_runs'])} clean runs via {b['worker']}")
        print(f"total {tuning} tuning runs + {clean} clean runs; plan written to {out / 'plan.json'}")
        return

    # SNV runs after the baselines when both are in the same campaign, so the
    # baseline table is fixed before the proposed method is measured.
    gate_snv = any(b['method'] != 'snv' for b in blocks) and any(b['method'] == 'snv' for b in blocks)

    while True:
        for gpu, entry in list(running.items()):
            code = entry['process'].poll()
            if code is None:
                continue
            entry['log'].close()
            entry['lease'].close()
            del running[gpu]
            job = entry['job']
            if valid_result(Path(job['directory']) / 'result.json', job['method'], job['scenario'],
                            job['half'], job['seed'], job['config'], args.tasks, args.epochs):
                job['state'] = 'complete'
            else:
                job['state'] = 'failed'
                failed.append(dict(directory=str(job['directory']), returncode=code))
                atomic(out / 'failed_jobs.json', failed)
                print('FAILED', Path(job['directory']).name, 'code', code, flush=True)
        refresh()
        baselines_done = all(b['done'] for b in blocks if b['method'] != 'snv')
        for gpu in args.gpus:
            if gpu in running:
                continue
            ready = [j for j in jobs if j['state'] == 'pending'
                     and (j['method'] != 'snv' or baselines_done or not gate_snv)]
            if not ready:
                continue
            # Finished searches' winner measurements first, then tuning in order.
            launch(min(ready, key=lambda j: (j['half'] != 2, j['priority'])), gpu)
        states = [j['state'] for j in jobs]
        atomic(out / 'status.json', dict(
            stage='snv' if baselines_done else 'baselines',
            active=[dict(gpu=g, run=Path(e['job']['directory']).name) for g, e in running.items()],
            complete=states.count('complete'), pending=states.count('pending'),
            running=states.count('running'), failed=len(failed),
            blocks_done=sum(b['done'] for b in blocks), blocks=len(blocks),
            updated_at=time.time()))
        if all(b['done'] for b in blocks):
            break
        if not running and not any(s == 'pending' for s in states):
            raise RuntimeError(f'campaign stalled on failed runs; see {out / "failed_jobs.json"}')
        time.sleep(args.poll)

    rows = []
    for block in blocks:
        state = json.loads(Path(block['path']).read_text())
        for result in state['evaluation']:
            rows.append(cost_row(block['method'], block['scenario'], result, state['best']['config']))
    with (out / 'results_per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic(out / 'CAMPAIGN_READY.json', dict(completed_at=time.time(),
           blocks=[dict(method=b['method'], scenario=b['scenario'], state=str(b['path'])) for b in blocks],
           table=str(out / 'results_per_seed.csv')))
    print('campaign complete:', out / 'results_per_seed.csv', flush=True)


if __name__ == '__main__':
    main()
