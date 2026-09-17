"""GTEP protocol: corrected halves, immutable-source run queue, explicit cost ledger.

One module holds the whole measurement protocol used for the paper:

* the CIFAR-100 disjoint-half split (D_HT for tuning, D_E for reporting),
* the per-method search spaces and the resolved training policy,
* ``one()``  -- a single (method, scenario, half, seed) run: random-init
  baseline row, per-task training, the full accuracy matrix including the
  superdiagonal that FWT and plasticity need, checkpointing, the cost ledger
  and the inference benchmark,
* ``queue()`` -- the R x 3-seed search followed by the three clean D_E runs.

Environment:
  GTEP_PROTOCOL   'legacy' (Adam, batch 64, lr-only search; the protocol the
                  CIFAR-100 campaign ran) or 'paper' (default: SGD with
                  method-specific schedules).
  GTEP_DATA_ROOT  dataset directory (default ./data; CIFAR-100 downloads).
  GTEP_DEVICE     torch device for a single run (default cuda:0).
  GTEP_GPU_UUIDS  comma-separated GPU UUIDs for the queue; default: every
                  visible GPU, in nvidia-smi order.
"""
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
def visible_gpu_uuids():
    """GPU UUIDs the queue may dispatch to.

    Runs are pinned by UUID rather than by index so a job cannot migrate to
    another card when CUDA_VISIBLE_DEVICES changes between processes.
    """
    override=os.environ.get('GTEP_GPU_UUIDS')
    if override:return [u.strip() for u in override.split(',') if u.strip()]
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                           capture_output=True,text=True,check=True).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError,subprocess.CalledProcessError):
        return []


GPUS=visible_gpu_uuids()
DATA_ROOT=os.environ.get('GTEP_DATA_ROOT','./data')
LR=('logu',1e-4,1e-2)
SPACE={
 'sgd':{'lr':LR},'lwf':{'lr':LR,'lwf_lambda':('logu',.1,10),
     'temperature':('choice',[1.,2.,3.,4.])},
 'ewc':{'lr':LR,'ewc_lambda':('logu',1,1e4),'ewc_gamma':('u',.9,1.)},
 'si':{'lr':LR,'si_c':('logu',.01,10),'si_xi':('logu',1e-4,1e-2)},
 'pec':{'lr':LR},
 'uniclun':{'lr':LR,'alpha1':('logu',.1,10),'alpha2':('logu',.1,10),'alpha3':('logu',.1,10)},
 'snv':{'lr':LR,'truncation':('choice',[.05,.1,.2])},
 'wsn':{'lr':LR,'wsn_density':('choice',[.1,.3,.5,.7])},
 'nispa':{'lr':LR,'nispa_prune_perc':('u',70,95),'nispa_recovery_perc':('u',1,5)},
 'spacenet':{'lr':LR,'spacenet_s_init':('u',.1,.3),'spacenet_rewire_fraction':('u',.05,.4)},
 'mcl':{'lr':LR,'lwf_lambda':('logu',.1,10),'temperature':('choice',[1.,2.,4.]),
     'mcl_density_alpha':('choice',[.25,.5,1.])},
}

# The paper's methods, in the order the campaign dispatches them. 'mcl3' is the
# legacy spelling of 'mcl' kept so CIFAR-100 run directories stay readable.
ORDER=['sgd','lwf','si','ewc','pec','joint','mcl','wsn','spacenet','nispa','uniclun','snv']
SPACE['joint']={'lr':LR}
SPACE['spacenet']={'lr':LR,'density_factor':('choice',[.5,1.,1.5]),'rewire_fraction':('choice',[.1,.2,.3])}
SPACE={m:SPACE[m] for m in ORDER}
SPACE['mcl3']=SPACE['mcl']
# Two protocols. 'paper' follows the published GTEP training settings (SGD and
# method-specific schedules). 'legacy' reproduces the completed gtep_cifar100
# campaign -- Adam, batch 64, lr-only search -- so reruns of the methods whose
# implementations were corrected stay comparable with the runs kept from it.
LEGACY_PROTOCOL=os.environ.get('GTEP_PROTOCOL','paper')=='legacy'
if not LEGACY_PROTOCOL:
    for name,space in SPACE.items():
        if name in ('pec','nispa'):
            space.update(batch_size=('choice',[32,64,128]),scheduler=('choice',['linear','none']))
        else:
            space.update(lr=('choice',[.05,.1,.15,.2,.3]),batch_size=('choice',[32,64,128,256,512]),
                         weight_decay=('choice',[.0001,.0005,.001,.005]),scheduler=('choice',['steplr','cosine']),
                         milestone_count=('choice',[2,3,4]),lr_decay=('choice',[.1,.3,.5]))

def resolved_policy(name,config,epochs,patience):
    if LEGACY_PROTOCOL:
        # gtep_cifar100 settings: Adam, no weight decay, no LR schedule.
        return dict(optimizer='adam',momentum=.9,weight_decay=0.,scheduler='none',
            milestone_count=2,lr_decay=.1,min_epochs=1,max_epochs=epochs,patience=patience)
    return dict(optimizer='adam' if name in ('pec','nispa') else 'sgd',
        momentum=.9,weight_decay=config.get('weight_decay',0.),scheduler=config.get('scheduler','none'),
        milestone_count=config.get('milestone_count',2),lr_decay=config.get('lr_decay',.1),
        min_epochs=min(epochs,100),max_epochs=epochs,patience=patience)


SOURCE_FILES=('cl_base.py','baselines.py','models.py','datasets.py','metrics.py',
    'training_policy.py','audit_cost.py','train.py','SNV/snv_core.py','SNV/snv_adaptive.py',
    'PEC/pec.py','SpaceNet/audited_spacenet.py','UniCLUN/uniclun.py','WSN/wsn.py',
    'joint_audited.py','NISPA/nispa.py','MCL/mcl.py','EWC/ewc.py','SI/si.py','LwF/lwf.py',
    'SGD/sgd.py')
UPSTREAM=('third_party/pec','third_party/uniclun','third_party/spacenet','third_party/gtep')


def source_fingerprint():
    """Hash every file that can change a result, so a later edit cannot be
    mistaken for the code that produced a run."""
    out={}
    for rel in SOURCE_FILES:
        path=ROOT/rel
        if path.exists():out[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def upstream_commits():
    out={}
    for rel in UPSTREAM:
        head=ROOT/rel/'.git'/'HEAD'
        if not head.exists():continue
        try:
            ref=head.read_text().strip()
            if ref.startswith('ref: '):
                ref=(ROOT/rel/'.git'/ref[5:]).read_text().strip()
            out[rel]=ref
        except OSError:pass
    return out


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def serial(value):
    import numpy as np
    import torch
    if isinstance(value,torch.Tensor):return serial(value.detach().cpu().tolist())
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def sample(space,rng):
    return {k:(rng.choice(s[1]) if s[0]=='choice' else math.exp(rng.uniform(math.log(s[1]),math.log(s[2])))
                if s[0]=='logu' else rng.uniform(s[1],s[2])) for k,s in space.items()}


@contextlib.contextmanager
def full_space(model,total_tasks):
    if hasattr(model,'full_output_space'):
        with model.full_output_space(total_tasks):yield
    elif hasattr(model,'seen_upto'):
        before=model.seen_upto;model.seen_upto=total_tasks-1
        try:yield
        finally:model.seen_upto=before
    elif hasattr(model,'current_task'):
        before=model.current_task;model.current_task=total_tasks-1
        try:yield
        finally:model.current_task=before
    else:
        yield


def make_method(name,benchmark,device,config):
    from baselines import build_method,build_custom_model,CUSTOM_BACKBONE
    from models import create_model
    config={k:v for k,v in config.items() if k not in ('batch_size','scheduler','milestone_count','lr_decay','weight_decay')}
    if name=='spacenet':
        from SpaceNet.audited_spacenet import AuditedSpaceNet,SpaceNetMLP
        return AuditedSpaceNet(SpaceNetMLP(),device,scenario=benchmark.scenario,**config)
    model=(build_custom_model(name,'cifar100',5,benchmark.scenario,10) if name in CUSTOM_BACKBONE
           else create_model('cifar100',5,10,benchmark.scenario))
    if name=='joint':
        from joint_audited import JointPrefix
        return JointPrefix(model,device,scenario=benchmark.scenario,**config)
    if name=='uniclun':
        from UniCLUN.uniclun import UniCLUN
        return UniCLUN(model,device,scenario=benchmark.scenario,buffer_size=1000,**config)
    options=dict(config);lr=options.pop('lr')
    options.update(num_classes=50,buffer_size=1000)
    if name=='snv':options.update(estimator_mode='reverse_tmc',max_permutations=32,
        use_mab=False,payoff='loss',shapley_eval_batches=0,masked_training=True,
        consolidation_epochs=25,consolidation_within_budget=True)
    return build_method(name,model,device,benchmark.scenario,lr,**options)


def one(args):
    import numpy as np
    import torch
    from datasets import ContinualLearningBenchmark
    from utils import set_seed
    from metrics import ContinualLearningMetrics
    from audit_cost import CostLedger
    from train import save_run_checkpoint
    config=json.loads(args.config)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    set_seed(args.seed);torch.set_num_threads(2)
    device=torch.device(os.environ.get('GTEP_DEVICE','cuda:0'))
    ledger=CostLedger(device)
    with ledger.phase('setup',count_flops=False):
        benchmark=ContinualLearningBenchmark('cifar100',10,DATA_ROOT,args.seed,
            args.scenario,2,gtep_half=args.half,split_seed=1234)
        loaders=[benchmark.get_task_data(t,config.get('batch_size',64)) for t in range(args.tasks)]
        if args.smoke_samples:
            from torch.utils.data import DataLoader,Subset
            loaders=[tuple(DataLoader(Subset(l.dataset,range(min(args.smoke_samples,len(l.dataset)))),
                batch_size=min(16,config.get('batch_size',64))) for l in triple) for triple in loaders]
        method=make_method(args.method,benchmark,device,config)
        method.training_policy=resolved_policy(args.method,config,args.epochs,args.patience)
        if args.method=='snv':method.consolidation_epochs=max(1,args.epochs//2)
        method.model.to(device)
    def output_space():
        return (method.full_output_space(10) if hasattr(method,'full_output_space') else full_space(method.model,10))
    checkpoint_model=method.model
    with ledger.phase('random_initialization_evaluation',method):
        with output_space():
            baseline=[method.evaluate(triple[1 if args.half==1 else 2],t)
                      for t,triple in enumerate(loaders)]
    tracker=ContinualLearningMetrics(args.tasks);tracker.set_random_baseline(np.array(baseline))
    validations=[];costs=[];started=time.perf_counter()
    for t,(train,val,test) in enumerate(loaders):
        epochs=args.epochs
        if args.method=='snv' and epochs<=25:
            method.consolidation_epochs=max(1,epochs//2)
        with ledger.phase(f'task_{t}/training_including_internal_validation',method,count_flops=args.flops) as cost:
            history=method.train_task(t,train,val,epochs,args.patience,True)
        costs.append(cost)
        with ledger.phase(f'task_{t}/evaluation',method):
            valrow=[method.evaluate(loaders[k][1],k) for k in range(t+1)]
            row=np.full(args.tasks,np.nan)
            for k in range(t+1):row[k]=(valrow[k] if args.half==1 else method.evaluate(loaders[k][2],k))
            with output_space():
                for k in range(t+1,args.tasks):row[k]=method.evaluate(loaders[k][1 if args.half==1 else 2],k)
            tracker.update(t,row);validations.append(valrow)
        with ledger.phase(f'task_{t}/checkpoint',method):
            save_run_checkpoint(str(out),0,t,method.model,method,tracker)
        progress={'method':args.method,'scenario':args.scenario,'half':args.half,'seed':args.seed,
            'completed_tasks':t+1,'matrix':serial(tracker.get_accuracy_matrix()),
            'validation_rows':validations,'history':serial(history),
            'training_policy':method.training_policy,
            'elapsed_seconds':time.perf_counter()-started,'complete':False}
        atomic(out/'progress.json',progress);atomic(out/'costs.json',serial(ledger.records))
        print('AUDIT_TASK '+json.dumps(progress),flush=True)
    matrix=tracker.get_accuracy_matrix()
    metrics=serial(tracker.summary()) if hasattr(tracker,'summary') else serial(tracker.get_all_metrics())
    final=float(np.mean(matrix[-1]));average=float(np.mean([np.mean(matrix[t,:t+1]) for t in range(args.tasks)]))
    metrics.update(ACC=final,AvgAcc=average,HARMONIC=2*final*average/(final+average) if final+average else 0.)
    inference=[]
    method.model.eval()
    with ledger.phase('inference_benchmark',method,count_flops=False):
        for t in range(args.tasks):
            x=next(iter(loaders[t][1 if args.half==1 else 2]))[0]
            for batch_size in (1,64):
                with (method.prediction_stream(t) if hasattr(method,'prediction_stream') else contextlib.nullcontext()):
                    r=ledger.inference(lambda inp:method.predict(inp,t),x[:batch_size],repeats=10)
                r['task_id']=t;inference.append(r)
    output={**progress,'complete':True,'metrics':metrics,'config':config,'inference':inference,
            'checkpoint_bytes':(out/'ckpt_run0.pt').stat().st_size,
            'evaluation_split':'D_HT validation' if args.half==1 else 'D_E test',
            'hardware':{'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu','gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'torch':torch.__version__,'cuda':torch.version.cuda},
            'class_order':benchmark.class_order.tolist()}
    from audit_cost import summarize
    output['cost_summary']=summarize(ledger.records,inference,output['checkpoint_bytes'])
    output['method_variant']=getattr(method,'name',args.method)
    output['all_task_histories']=serial(method.history)
    atomic(out/'costs.json',serial(ledger.records));atomic(out/'result.json',serial(output))


def queue(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol={'dataset':'CIFAR-100 disjoint halves','classes_per_half':50,'tasks':10,
        'classes_per_task':5,'D_HT':1,'D_E':2,'split_seed':1234,'seeds':[42,43,44],
        'R':args.rounds,'sample_seed':7,'max_total_epochs':args.epochs,'patience':args.patience,
        'batch_size':64,'optimizer':'Adam; explicit method-specific schedules',
        'selection':'harmonic mean of final ACC and AvgAcc on D_HT validation',
        'memory_methods_buffer_examples':1000,'gtep_paper_exact_reproduction':False,
        'costs':'tuning runs share GPUs (their timings are not reported); selected winner (D_E) runs take the GPU exclusively and carry the published cost numbers and FLOP instrumentation',
        'methods':args.methods,'spaces':{m:SPACE[m] for m in args.methods},
        'gpu_placement':'SNV runs last, pinned to GPU2; all other methods use all four GPUs; one active run per GPU enforced by an exclusive file lock',
        'upstream_adaptations':{
            'pec':'github.com/michalzajac-ml/pec@3c15633 mechanism (single shared frozen teacher, independent per-class students, identical init) on the GTEP schedule; deviates from the published 1-epoch/batch-1/1-class-per-task online setting',
            'uniclun':'paper-based reimplementation; upstream repo omits the model module'},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method_source_sha256':source_fingerprint(),
        'upstream_commits':upstream_commits()}
    protocol=serial(protocol)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Cannot mix protocols in an existing output directory')
    atomic(out/'protocol.json',protocol)
    def run(name,scenario,half,seed,config,tag,gpu_index,flops=False,measured=False):
        directory=out/'runs'/f'{name}_{scenario}_{tag}_s{seed}'
        if (directory/'result.json').exists():return json.loads((directory/'result.json').read_text())
        directory.mkdir(parents=True,exist_ok=True)
        # Winner runs are the ones whose cost numbers get published, so they take
        # the GPU exclusively. Tuning runs only decide which config wins -- accuracy
        # is unaffected by contention -- so they pack densely instead.
        gpu_lock=open('/tmp/audited-gtep-'+GPUS[gpu_index]+'.lock','w')
        fcntl.flock(gpu_lock,fcntl.LOCK_EX if measured else fcntl.LOCK_SH)
        cmd=[sys.executable,'-u',str(Path(__file__)),'--one','--method',name,'--scenario',scenario,
             '--half',str(half),'--seed',str(seed),'--config',json.dumps(config),'--out',str(directory),
             '--epochs',str(args.epochs),'--patience',str(args.patience),'--tasks',str(args.tasks)]
        if flops:cmd.append('--flops')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu_index],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        atomic(directory/'command.json',{'command':cmd,'gpu':GPUS[gpu_index]})
        print('START',directory.name,flush=True)
        with (directory/'train.log').open('a') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        gpu_lock.close()
        return json.loads((directory/'result.json').read_text())
    if not GPUS:raise RuntimeError('no GPUs visible; set GTEP_GPU_UUIDS')
    # SNV runs last and alone on one pinned card (the last visible one).
    SNV_GPU=len(GPUS)-1
    non_snv=[m for m in args.methods if m!='snv']
    def lanes(name):
        # Each non-SNV method only has three concurrent seeds, so rotate the
        # starting GPU per method: across the queue every GPU (including SNV's,
        # until SNV claims it) does work.
        if name=='snv':return [SNV_GPU]*3
        offset=non_snv.index(name) if name in non_snv else 0
        return [(offset+j)%len(GPUS) for j in range(3)]
    # Tuning packs several configs per GPU (~1.2GB each on 49GB cards); winner
    # runs serialize behind the exclusive lock regardless of this width.
    pack=max(1,args.pack)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)*pack) as pool:
        for name in args.methods:
            lane=lanes(name)
            rng=random.Random(7);configs=[sample(SPACE[name],rng) for _ in range(args.rounds)]
            scenarios=['task_il'] if name=='wsn' else ['class_il'] if name=='pec' else ['class_il','task_il']
            for scenario in scenarios:
                trials=[]
                state_path=out/f'{name}_{scenario}.json'
                # Dispatch every (config, seed) tuning job at once; the pool width
                # and the per-GPU shared lock decide how many actually run together.
                futures={}
                slot=0
                for i,config in enumerate(configs):
                    for j in range(3):
                        # Round-robin every (config, seed) job over all GPUs so no
                        # card idles just because a method has only three seeds.
                        gpu=SNV_GPU if name=='snv' else slot%len(GPUS)
                        slot+=1
                        futures[(i,j)]=pool.submit(run,name,scenario,1,42+j,config,f'ht_r{i}',gpu)
                # Collect per config so state is written as trials land, not only
                # at the end: a crash keeps every completed config.
                for i,config in enumerate(configs):
                    rows=[futures[(i,j)].result() for j in range(3)]
                    score=sum(r['metrics']['HARMONIC'] for r in rows)/3
                    trials.append({'config':config,'score':score,'per_seed_metrics':[r['metrics'] for r in rows]})
                    atomic(state_path,{'tuning':trials,'best':max(trials,key=lambda v:v['score']),'evaluation':None})
                winner=max(trials,key=lambda v:v['score'])
                if args.tasks<10:
                    continue # pilots never open the D_E phase
                rows=list(pool.map(lambda j:run(name,scenario,2,42+j,winner['config'],'eval',lane[j%len(lane)],args.flops,True),range(3)))
                atomic(state_path,{'tuning':trials,'best':winner,'evaluation':{
                    'per_seed_metrics':[r['metrics'] for r in rows],
                    'mean_ACC':sum(r['metrics']['ACC'] for r in rows)/3}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--one',action='store_true')
    p.add_argument('--method',choices=list(SPACE));p.add_argument('--methods',nargs='+',default=['pec','ewc','si','lwf','sgd','uniclun','snv'])
    p.add_argument('--scenario',default='class_il');p.add_argument('--half',type=int,default=1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--config',default='{"lr":0.001}')
    p.add_argument('--out',required=True);p.add_argument('--rounds',type=int,default=30)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--tasks',type=int,default=10);p.add_argument('--flops',action='store_true')
    p.add_argument('--pack',type=int,default=4,
        help='Concurrent tuning runs per GPU. Winner runs always take the GPU exclusively.')
    p.add_argument('--smoke_samples',type=int,default=0,
        help='Truncate every split to N examples for pipeline smoke tests (0 = full data).')
    args=p.parse_args();one(args) if args.one else queue(args)


if __name__=='__main__':main()
