"""Export a finished campaign -- metrics, costs, hyperparameters, raw accuracy
matrices -- as CSV, an Excel workbook, Markdown, LaTeX and a static HTML page.

Reads only the JSON a campaign wrote; it never loads a model or touches a GPU.

  python campaign/build_report.py --campaign runs/cifar100 --out reports/cifar100

Every table is derived from the D_E winner runs (metrics and costs) and the
D_HT trials (hyperparameter search).  Rows are checked against the protocol as
they are read: a run that is incomplete, was measured on a shared GPU, or does
not match the configuration its block selected aborts the export rather than
being averaged into a published number.
"""
import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import os
import statistics as st
import sys
import zipfile
from xml.sax.saxutils import escape

ORDER = ['joint','sgd','ewc','si','lwf','wsn','pec','spacenet','nispa','uniclun','snv']
LABEL = {'joint':'Joint','sgd':'SGD','ewc':'EWC','si':'SI','lwf':'LwF','wsn':'WSN',
         'pec':'PEC','spacenet':'SpaceNet','nispa':'NISPA','uniclun':'UniCLUN*',
         'snv':'SNV'}
METRICS = ['ACC','AvgAcc','BWT','FWT','PS','P','S','AF','HARMONIC']


def read(p):
    return json.loads(Path(p).read_text())


def number(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)


def fmt(value, spec):
    """Format a display cell; '—' when the value was not measured."""
    return format(value, spec) if number(value) else '—'


def mb(value):
    """Bytes to MB (1e6); None when the field was not measured (e.g. a CPU run)."""
    return value / 1e6 if number(value) else None


def stats(values):
    v=[x for x in values if number(x)]
    return (st.mean(v),st.stdev(v) if len(v)>1 else 0) if v else (None,None)


def metric_values(d):
    return {k+'_pct' if k not in ('PS','P','S') else k:
            (d['metrics'][k]*(1 if k in ('PS','P','S') else 100)
             if number(d['metrics'].get(k)) else None) for k in METRICS}


def summarize(group,keys):
    out={}
    for key in keys:
        out[key+'_mean'],out[key+'_sd']=stats([r.get(key) for r in group])
    return out


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def cell(x):
    if x is None:return '—'
    if isinstance(x,float):return f'{x:.6g}'
    return str(x)


def markdown(rows):
    keys=list(rows[0])
    lines=['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
    lines+=['| '+' | '.join(cell(r.get(k)).replace('|','\\|').replace('\n',' ') for k in keys)+' |' for r in rows]
    return '\n'.join(lines)


def xlsx_write(path,tables):
    """Small dependency-free OOXML workbook with filters and frozen headers."""
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    def col(i):
        s=''
        while i:i,r=divmod(i-1,26);s=chr(65+r)+s
        return s
    def xcell(value,ref,style):
        if value is None:return f'<c r="{ref}"/>'
        if number(value):return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
        text=str(value).lower() if isinstance(value,bool) else str(value)
        assert len(text)<=32767, 'Excel cell too long'
        return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        overrides=''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(tables)+1))
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+overrides+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        sheets=''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,(name,_) in enumerate(tables,1))
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheets}</sheets></workbook>')
        rels=''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(tables)+1))
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+rels+'<Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml',f'<styleSheet xmlns="{ns}"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        for i,(name,rows) in enumerate(tables,1):
            keys=list(dict.fromkeys(k for r in rows for k in r)); grid=[keys]+[[r.get(k) for k in keys] for r in rows]
            body=''.join(f'<row r="{j}">'+''.join(xcell(v,f'{col(k)}{j}',1 if j==1 else 0) for k,v in enumerate(row,1))+'</row>' for j,row in enumerate(grid,1))
            z.writestr(f'xl/worksheets/sheet{i}.xml',f'<worksheet xmlns="{ns}"><dimension ref="A1:{col(len(keys))}{len(grid)}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="{len(keys)}" width="19" customWidth="1"/></cols><sheetData>{body}</sheetData><autoFilter ref="A1:{col(len(keys))}{len(grid)}"/></worksheet>')


def load_blocks(root):
    """Every finished block of the campaign, in report order."""
    ready = root / 'CAMPAIGN_READY.json'
    blocks = []
    for path in sorted((root / 'blocks').glob('*.json')):
        block = read(path)
        if block.get('evaluation') and block.get('best'):
            blocks.append(block)
    if not blocks:
        raise SystemExit(f'no finished blocks in {root / "blocks"}')
    blocks.sort(key=lambda b: (b['scenario'],
                               ORDER.index(b['method']) if b['method'] in ORDER else len(ORDER)))
    return blocks, (read(ready) if ready.exists() else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True, help='campaign directory written by run_campaign.py')
    p.add_argument('--out', required=True)
    p.add_argument('--label', default=None,
                   help='report title; the campaign dataset is used when omitted')
    args = p.parse_args()
    root = Path(args.campaign).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    blocks, ready = load_blocks(root)
    protocol = read(root / 'protocol.json') if (root / 'protocol.json').exists() else {}
    # The campaign states its own benchmark; the report follows it instead of
    # naming CIFAR-100, so a CIFAR-20 or TinyImageNet report is labelled right.
    dataset = str(protocol.get('dataset', 'cifar100')).split()[0]
    DATASET_LABEL = {'cifar20': 'CIFAR-20', 'cifar100': 'CIFAR-100',
                     'tinyimagenet': 'TinyImageNet-200', 'imagenet1k': 'ImageNet-1k'}
    dataset_label = DATASET_LABEL.get(dataset, dataset)
    if not args.label:
        args.label = f'{dataset_label} GTEP campaign'
    seeds = set(protocol.get('seeds', [42, 43, 44]))
    epochs = protocol.get('max_total_epochs', 200)

    hp_keys = sorted({k for b in blocks for t in b['tuning'] for k in t['config']})
    metric_keys = list(metric_values(read(blocks[0]['evaluation'][0])))
    results = []; result_summary = []; trials = []; trial_seeds = []; selected = []
    costs = []; cost_summary = []; task_rows = []; phase_rows = []; inference_rows = []
    matrices = []; all_configs = []; protocols = []; references = {}

    for b in blocks:
        method, scenario = b['method'], b['scenario']
        base = dict(method=method, scenario=scenario)
        best = max(b['tuning'], key=lambda t: t['score'])
        assert best['config'] == b['best']['config'], f'{method}/{scenario}: winner mismatch'
        ht_class_orders = {}; ht_best = None
        for order, t in enumerate(b['tuning']):
            index = t.get('trial_index', order)
            group = []
            for file in t['results']:
                d = read(file)
                assert (d['method'] == method and d['scenario'] == scenario and d['half'] == 1
                        and d['complete'] and d['config'] == t['config']), f'tuning run mismatch: {file}'
                assert d['training_policy']['max_epochs'] == epochs, f'wrong epoch cap: {file}'
                ht_class_orders[d['seed']] = set(d['class_order'])
                row = {**base, 'trial_index': index, 'seed': d['seed'], **metric_values(d),
                       'selected': t['config'] == best['config'],
                       **{k: t['config'].get(k) for k in hp_keys},
                       'config_json': json.dumps(t['config'], sort_keys=True),
                       'training_policy_json': json.dumps(d['training_policy'], sort_keys=True),
                       'elapsed_seconds_training_loop': d['elapsed_seconds'], 'result_path': file}
                trial_seeds.append(row); group.append(row)
                references[file] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            assert {r['seed'] for r in group} == seeds, f'{method}/{scenario} trial {index}: seeds'
            assert math.isclose(st.mean(r['HARMONIC_pct'] for r in group) / 100, t['score'], abs_tol=1e-9)
            agg = {**base, 'trial_index': index, 'selected': t['config'] == best['config'],
                   'selection_HARMONIC_pct': t['score'] * 100,
                   **summarize(group, metric_keys), **{k: t['config'].get(k) for k in hp_keys},
                   'config_json': json.dumps(t['config'], sort_keys=True),
                   'config_origin': b.get('config_origin', '')}
            trials.append(agg)
            if t['config'] == best['config']:
                ht_best = agg
        winner_group = []; cost_group = []
        for file in b['evaluation']:
            d = read(file); c = d['cost_summary']
            assert (d['method'] == method and d['scenario'] == scenario and d['half'] == 2
                    and d['complete'] and d['config'] == best['config']), f'winner run mismatch: {file}'
            assert not (ht_class_orders[d['seed']] & set(d['class_order'])), (
                f'D_E classes overlap the tuning half: {file}')
            assert c['gpu_exclusive_observed'], f'cost run shared the GPU: {file}'
            row = {**base, 'seed': d['seed'], **metric_values(d), 'result_path': file}
            results.append(row); winner_group.append(row)
            references[file] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            protocols.append({**base, 'seed': d['seed'], 'batch_size': d['config'].get('batch_size', 64),
                              **d['training_policy'], 'gpu': d['hardware']['gpu'],
                              'gpu_uuid': d['hardware']['gpu_uuid'], 'torch': d['hardware']['torch'],
                              'cuda': d['hardware']['cuda'], 'method_variant': d['method_variant'],
                              'D_E_class_order': json.dumps(d['class_order'])})
            cost = {**base, 'seed': d['seed'], 'train_minutes': sum(c['train_minutes_per_task']),
                    'train_minutes_per_task': c['train_minutes_per_task_mean'],
                    'training_gpu_hours': c['training_gpu_hours'],
                    'accounted_run_gpu_hours': c['accounted_run_gpu_hours'],
                    'peak_allocated_MB': mb(c['gpu_peak_allocated_bytes']),
                    'peak_reserved_MB': mb(c['gpu_peak_reserved_bytes']),
                    'peak_device_used_MB': mb(c['gpu_peak_device_used_bytes']),
                    'peak_process_tree_RSS_MB': mb(c['process_tree_peak_rss_bytes']),
                    'optimizer_peak_MB': mb(c['optimizer_state_peak_bytes']),
                    'model_parameters_MB': mb(c['final_state']['resident_parameter_bytes']),
                    'resident_tensor_storage_MB': mb(c['final_state']['resident_tensor_storage_bytes']),
                    'resident_parameters': c['final_state']['resident_parameter_count'],
                    'requires_grad_parameters': c['final_state']['requires_grad_parameter_count'],
                    'replay_examples': c['final_state']['replay_examples'],
                    'checkpoint_MB': mb(d['checkpoint_bytes']),
                    'gpu_util_percent': c['gpu_util_percent_sample_mean'],
                    'gpu_util_p95': c['gpu_util_percent_p95'],
                    'energy_joules': c['energy_joules_device_counter'],
                    'energy_Wh': (c['energy_joules_device_counter'] / 3600
                                  if number(c['energy_joules_device_counter']) else None),
                    'gpu_exclusive_observed': c['gpu_exclusive_observed'], 'result_path': file}
            for batch in (1, 64):
                sampled = [v for v in d['inference'] if v['batch_size'] == batch]
                for output, key in [('infer_ms_per_sample', 'inference_ms_per_sample'),
                                    ('inference_GFLOPs_per_sample', 'supported_operator_gflops_per_sample'),
                                    ('throughput_samples_s', 'throughput_samples_per_second')]:
                    cost[f'{output}_batch{batch}'] = stats([v.get(key) for v in sampled])[0]
            inference_rows += [{**base, 'seed': d['seed'], **v} for v in d['inference']]
            records = read(Path(file).parent / 'costs.json')
            assert not any(x.get('flop_instrumentation_during_timing') for x in records), (
                f'FLOP counting ran inside the timed phase: {file}')
            assert all(x.get('gpu_exclusive_observed') for x in records)
            for key in ('disk_read_bytes', 'disk_write_bytes', 'main_process_cpu_seconds', 'optimizer_steps'):
                cost[key] = sum(x.get(key, 0) for x in records)
            for record in records:
                phase_rows.append({**base, 'seed': d['seed'],
                                   **{k: v for k, v in record.items() if not isinstance(v, (list, dict))}})
            cost_group.append(cost); costs.append(cost)
            for t, h in enumerate(d['all_task_histories']):
                task_rows.append({**base, 'seed': d['seed'], 'task_index': t,
                                  'train_minutes': c['train_minutes_per_task'][t],
                                  'epochs_run_recorded': h.get('epochs_run'),
                                  'epoch_cap_recorded': h.get('epoch_cap', d['training_policy']['max_epochs']),
                                  'best_epoch': h.get('best_epoch'), 'stop_reason': h.get('stop_reason'),
                                  'accepted_phases_NISPA': h.get('phases'),
                                  'capacity_used_percent': h.get('capacity_used'),
                                  'selected_neurons': h.get('selected'),
                                  'best_validation_loss': h.get('best_validation_loss'),
                                  'cap_reached_while_improving': h.get('cap_reached_while_improving')})
            for i, a in enumerate(d['matrix']):
                for j, value in enumerate(a):
                    matrices.append({**base, 'seed': d['seed'], 'after_task_index': i,
                                     'evaluated_task_index': j,
                                     'accuracy_pct': 100 * value if number(value) else None,
                                     'evaluation_space': ('seen-class evaluation' if j <= i
                                                          else 'full-output-space future diagnostic')})
        assert {r['seed'] for r in winner_group} == seeds, f'{method}/{scenario}: winner seeds'
        agg = {**base, **summarize(winner_group, metric_keys),
               'D_HT_ACC_pct_mean': ht_best['ACC_pct_mean'], 'D_HT_ACC_pct_sd': ht_best['ACC_pct_sd'],
               'D_HT_HARMONIC_pct': best['score'] * 100}
        agg['gap_DE_minus_DHT_ACC_pp'] = agg['ACC_pct_mean'] - agg['D_HT_ACC_pct_mean']
        result_summary.append(agg)
        keys = [k for k, v in cost_group[0].items() if number(v) and k != 'seed']
        ca = {**base, **summarize(cost_group, keys), 'all_seeds_gpu_exclusive': True,
              'seed_accounted_gpu_hours_sum': sum(x['accounted_run_gpu_hours'] for x in cost_group)}
        for key in ('peak_allocated_MB', 'peak_reserved_MB', 'peak_device_used_MB', 'peak_process_tree_RSS_MB'):
            ca[key + '_max_over_seeds'] = max((x[key] for x in cost_group if number(x[key])), default=None)
        cost_summary.append(ca)
        selected.append({**base, 'trial_index': best.get('trial_index'),
                         'D_HT_HARMONIC_pct': best['score'] * 100,
                         'D_HT_ACC_pct': ht_best['ACC_pct_mean'], 'D_E_ACC_pct': agg['ACC_pct_mean'],
                         **{k: best['config'].get(k) for k in hp_keys},
                         'config_json': json.dumps(best['config'], sort_keys=True),
                         'config_origin': b.get('config_origin', '')})
        all_configs.append(b)

    notes = [
        ('Scope', f"{args.label}: {len({b['method'] for b in blocks})} methods, {len(blocks)} "
                  f"method/scenario blocks, {len(trials)} configurations, {len(trial_seeds)} tuning runs, "
                  f"{len(results)} final winner runs. PEC is Class-IL only; WSN is Task-IL only. "
                  "SNV means SNV-A (SNV/snv_adaptive.py)."),
        ('Dataset', f"{dataset_label}, disjoint {protocol.get('classes_per_half', '?')}-class halves, "
                    f"{protocol.get('tasks', '?')} tasks x {protocol.get('classes_per_task', '?')} classes "
                    f"per half. Fixed half membership split_seed={protocol.get('split_seed', 1234)}; "
                    f"class/task order seeds {', '.join(str(s) for s in sorted(seeds))}. Original train+test "
                    'pooled and split 70/10/20 into train/validation/test per class.'),
        ('Training protocol', f"GTEP_PROTOCOL={protocol.get('gtep_protocol', 'legacy')}: batch 64, weight_decay=0, "
                    f"scheduler=none, maximum {epochs} epochs, patience {protocol.get('patience', 15)}. "
                    'Momentum/milestone/lr_decay fields in the raw policy are inactive with Adam and no scheduler.'),
        ('Training exceptions', 'NISPA uses its phase/recovery stopping rule (5 epochs per phase, at most '
                    '30 phases) rather than generic patience stopping; its history records accepted phases, '
                    'not total epochs. PEC uses independent per-class optimizers with a shared task-level '
                    'validation stopping decision.'),
        ('Selection', f"{len(blocks[0]['tuning'])} configurations per block; the harmonic mean of final ACC "
                    'and AvgAcc is computed per seed on the D_HT validation split, then averaged over seeds. '
                    'The maximum selects the winner. D_E never selects hyper-parameters.'),
        ('Statistics', 'Mean and sample SD over seeds (not standard error). ACC/AvgAcc/HARMONIC in percent; '
                    'BWT/FWT/AF in percentage points; PS/P/S dimensionless. Rounded display values do not '
                    'replace the full-precision CSV/XLSX values.'),
        ('ACC and AvgAcc', 'ACC: mean final accuracy across the 10 tasks. AvgAcc: mean over training stages '
                    'of the average accuracy on the tasks seen by that stage.'),
        ('BWT and AF', 'BWT: mean(final accuracy - accuracy immediately after learning), excluding the last '
                    'task. AF: mean(max accuracy from learning through the final stage - final accuracy), '
                    'excluding the last task.'),
        ('FWT', 'Mean(next-task accuracy before learning - random-initialization accuracy). Future-task CIL '
                    'diagnostics use the full 50-class output space; seen-task CIL accuracy uses seen classes.'),
        ('PS', 'P = mean[(A[t,t]-A[t-1,t])/(1-A[t-1,t])]; S = 1 + BWT on raw fractions; '
                    'PS = 2*max(P,0)*max(S,0)/(max(P,0)+max(S,0)). P, S and PS are all exported.'),
        ('Joint', 'Joint trains a fresh model on each seen-task prefix. Its BWT/FWT/AF/PS are retained as '
                    'logged; they are reference-trajectory statistics, not forgetting in a carried-over model.'),
        ('Costs', 'Costs come only from the selected D_E winner runs, each of which passed exclusive-GPU '
                    'monitoring. Training minutes include internal validation and method consolidation. '
                    'Accounted GPU hours also include setup, evaluation, checkpointing and inference for that '
                    'run -- not the whole hyper-parameter campaign.'),
        ('FLOPs and latency', 'GFLOPs are supported-operator INFERENCE FLOPs per sample (a multiply-add counts '
                    'as 2), measured in a pass outside the latency timer. Dense masked operators keep their '
                    'dense FLOP cost. Training FLOPs are not measured in the clean cost runs.'),
        ('Inference measurement', 'The final trained model is evaluated across the 10 task inputs; the mean is '
                    'taken over tasks, then seeds. Batches 1 and 64 are exported separately: device-resident '
                    'input, host dispatch plus synchronized device execution, 10 timed repeats after 5 warmups. '
                    'Data loading is excluded.'),
        ('Memory', 'MB = 1e6 bytes. Mean per-run peak and the maximum across seeds are both exported. GPU '
                    'allocated, reserved and device-used differ. Resident parameter bytes include retained '
                    'auxiliary modules (teachers, a Joint initial model) across CPU and GPU; they are not the '
                    'deployed inference model size, and differ from checkpoint bytes.'),
        ('Isolation', 'Per-GPU exclusivity is not whole-machine isolation: jobs on other GPUs share CPU, RAM, '
                    'PCIe and power. GPU utilization is a sensor-sample mean; energy is a device counter over '
                    'the recorded phases.'),
        ('Implementation labels', 'UniCLUN* is a paper-based re-implementation (the upstream repository omits '
                    'the model module), not a verified reproduction of official code. PEC uses the official '
                    'mechanism adapted to multi-epoch five-class tasks. NISPA and SpaceNet use their native '
                    'architectures.'),
        ('Provenance', f'{len(references)} result files hashed into provenance.json; no training was started '
                    'or modified to produce this report.'),
    ]
    if ready:
        notes.insert(1, ('Completion UTC',
                         dt.datetime.fromtimestamp(ready['completed_at'], dt.timezone.utc).isoformat()))

    os.environ.setdefault('GTEP_PROTOCOL', 'legacy')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import audited_gtep as g
    spaces = []
    for method in dict.fromkeys(b['method'] for b in blocks):
        spec = g.SPACE.get(method, {})
        for key in sorted({k for b in blocks if b['method'] == method for t in b['tuning'] for k in t['config']}):
            sampled = [t['config'][key] for b in blocks if b['method'] == method
                       for t in b['tuning'] if key in t['config']]
            definition = spec.get(key)
            spaces.append({'method': method, 'parameter': key,
                           'distribution': definition[0] if definition else 'fixed switch (not searched)',
                           'defined_domain': json.dumps(definition[1:]) if definition else None,
                           'observed_min': min(sampled, key=str), 'observed_max': max(sampled, key=str),
                           'actual_distinct_values': json.dumps(sorted(set(map(str, sampled))))})

    tables = [('README', [{'topic': k, 'details': v} for k, v in notes]),
              ('Metrics_summary', result_summary), ('Metrics_per_seed', results),
              ('Costs_summary', cost_summary), ('Costs_per_seed', costs),
              ('Selected_hyperparameters', selected), ('All_configs', trials),
              ('All_tuning_seeds', trial_seeds), ('Search_spaces', spaces),
              ('Resolved_training_policy', protocols), ('Epochs_and_tasks', task_rows),
              ('Inference_per_task', inference_rows), ('Cost_phases', phase_rows),
              ('Accuracy_matrices', matrices)]
    for name, rows in tables:
        csv_write(out / (name + '.csv'), rows)
    xlsx_write(out / 'campaign_tables.xlsx', tables)
    (out / 'all_hyperparameters_and_winners.json').write_text(json.dumps(all_configs, indent=2))
    (out / 'provenance.json').write_text(json.dumps(
        {'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'campaign': str(root),
         'protocol': protocol, 'result_sha256': references,
         'counts': {name: len(rows) for name, rows in tables}}, indent=2))

    label = lambda m: LABEL.get(m, m)
    displays = []
    for scen, title in [('task_il', 'Task-incremental D_E'), ('class_il', 'Class-incremental D_E')]:
        table = [{'Method': label(r['method']),
                  'ACC (%)': f"{r['ACC_pct_mean']:.2f} ± {r['ACC_pct_sd']:.2f}",
                  'AvgAcc (%)': fmt(r.get('AvgAcc_pct_mean'), '.2f'),
                  'BWT (pp)': f"{r['BWT_pct_mean']:+.2f}",
                  'FWT (pp)': f"{r['FWT_pct_mean']:+.2f}" if r['FWT_pct_mean'] is not None else '—',
                  'PS': fmt(r.get('PS_mean'), '.4f'), 'AF (pp)': fmt(r.get('AF_pct_mean'), '.2f')}
                 for r in result_summary if r['scenario'] == scen]
        if table:
            displays.append((title, table))
    for scen, title in [('task_il', 'Task-incremental winner costs'),
                        ('class_il', 'Class-incremental winner costs')]:
        table = [{'Method': label(r['method']), 'Train min': fmt(r.get('train_minutes_mean'), '.2f'),
                  'Min/task': fmt(r.get('train_minutes_per_task_mean'), '.2f'),
                  'Infer GFLOPs/sample': fmt(r.get('inference_GFLOPs_per_sample_batch64_mean'), '.3f'),
                  'Peak alloc MB': fmt(r.get('peak_allocated_MB_mean'), '.0f'),
                  'Infer ms/sample (B64)': fmt(r.get('infer_ms_per_sample_batch64_mean'), '.3f'),
                  'GPU util %': fmt(r.get('gpu_util_percent_mean'), '.1f'),
                  'GPU h/run': fmt(r.get('accounted_run_gpu_hours_mean'), '.3f'),
                  'Resident params MB': fmt(r.get('model_parameters_MB_mean'), '.2f')}
                 for r in cost_summary if r['scenario'] == scen]
        if table:
            displays.append((title, table))
    selected_display = []
    for r in selected:
        config = json.loads(r['config_json']); lr = config.pop('lr', None)
        selected_display.append({'Method': label(r['method']),
                                 'Scenario': 'CIL' if r['scenario'] == 'class_il' else 'TIL',
                                 'Trial (0-based)': r['trial_index'],
                                 'lr': f'{lr:.9g}' if number(lr) else '—',
                                 'Other selected parameters': '; '.join(
                                     f'{k}={v:.9g}' if isinstance(v, float) else f'{k}={v}'
                                     for k, v in config.items()) or '—'})
    displays.append(('Selected hyperparameters', selected_display))

    md = [args.label, '', f'{len(blocks)} blocks · {len(trials)} configurations · '
          f'{len(trial_seeds)} tuning runs · {len(results)} winner runs', '']
    for title, rows in displays:
        md.extend([f'**{title}**', '', markdown(rows), ''])
    md += ['**Protocol, definitions and limitations**', ''] + [f'- **{k}:** {v}' for k, v in notes]
    md += ['', 'Every configuration, every tuning seed, the exact selected values, full mean/SD cost '
           'tables, per-task histories, resolved training policies and the raw accuracy matrices are in '
           'campaign_tables.xlsx and the CSV files beside it.']
    (out / 'report.md').write_text('\n'.join(md) + '\n')

    panels = []
    for title, rows in displays + [('All hyperparameter trials', trials), ('Search spaces', spaces),
                                   ('Training policy', protocols),
                                   ('All cost fields (mean and SD)', cost_summary)]:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        head_html = ''.join('<th>' + html.escape(k) + '</th>' for k in keys)
        body = ''.join('<tr>' + ''.join('<td>' + html.escape(cell(r.get(k))) + '</td>' for k in keys) + '</tr>'
                       for r in rows)
        panels.append('<section><h2>' + html.escape(title) + '</h2><input aria-label="Filter table" '
                      'placeholder="Filter this table…" oninput="filterTable(this)"><div class="scroll">'
                      '<table><thead><tr>' + head_html + '</tr></thead><tbody>' + body +
                      '</tbody></table></div></section>')
    intro = ''.join('<p><b>' + html.escape(k) + ':</b> ' + html.escape(v) + '</p>' for k, v in notes)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>' + html.escape(args.label) + '</title>'
            '<style>body{font:15px system-ui;margin:30px;color:#172133;background:#fafbfd}'
            'h1,h2{color:#132e56}section{margin:32px 0}.scroll{overflow:auto;max-height:700px}'
            'table{border-collapse:collapse;background:white;font-size:13px}'
            'th,td{padding:9px;border:1px solid #dce3ed;text-align:left;white-space:nowrap}'
            'th{position:sticky;top:0;background:#eaf0f8}tr:nth-child(even){background:#f5f8fc}'
            'input{padding:10px;margin-bottom:10px;width:280px}p{max-width:1100px;line-height:1.5}</style>'
            '<h1>' + html.escape(args.label) + '</h1><p>' + f'{len(blocks)} blocks · {len(trials)} '
            f'configurations · {len(trial_seeds)} tuning runs · {len(results)} winner runs' + '</p>'
            '<p><a href="campaign_tables.xlsx">Excel workbook</a> · <a href="report.md">Markdown report</a> · '
            '<a href="all_hyperparameters_and_winners.json">Full hyperparameter records</a></p>'
            + ''.join(panels) + '<h2>Protocol and interpretation</h2>' + intro +
            '<script>function filterTable(input){const q=input.value.toLowerCase();'
            'for(const row of input.parentElement.querySelectorAll("tbody tr")){'
            'row.hidden=!row.textContent.toLowerCase().includes(q)}}</script></html>')
    (out / 'report.html').write_text(page)

    def tex(v):
        s = str(v)
        for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
            s = s.replace(a, b)
        return s.replace('±', r'$\pm$').replace('—', '--')
    lines = ['% Generated from saved winner results. See report.md for units and caveats.']
    for title, rows in displays[:4]:
        keys = list(rows[0])
        lines += ['% ' + title, r'\begin{tabular}{' + 'l' + 'r' * (len(keys) - 1) + '}', r'\hline',
                  ' & '.join(tex(k) for k in keys) + r' \\', r'\hline']
        lines += [' & '.join(tex(r[k]) for k in keys) + r' \\' for r in rows]
        lines += [r'\hline', r'\end{tabular}', '']
    (out / 'tables.tex').write_text('\n'.join(lines))

    with zipfile.ZipFile(out / 'campaign_report.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                z.write(path, path.name)
    print(json.dumps({'output': str(out), 'counts': {name: len(rows) for name, rows in tables}}, indent=2))


if __name__ == '__main__':
    main()
