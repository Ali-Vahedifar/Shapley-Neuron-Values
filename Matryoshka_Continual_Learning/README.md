# Matryoshka Continual Learning (MCL)

Code for the MCL paper: the method, every continual-learning baseline it is
compared against, the GTEP evaluation protocol, the CL metrics and the cost
measurements. No checkpoints or results are included. Everything here is
source code you can run.

## What is included

| Group | Methods | Folder |
|---|---|---|
| Bounds | SGD (lower), Joint training (upper) | `SGD/`, `Joint/` + `joint_audited.py` |
| Regularisation | EWC, SI, LwF | `EWC/`, `SI/`, `LwF/` |
| Sparse / architecture | WSN (Task-IL only), PEC (Class-IL only), SpaceNet, NISPA | `WSN/`, `PEC/`, `SpaceNet/`, `NISPA/` |
| CL + unlearning (†) | UniCLUN | `UniCLUN/` |
| Shapley valuation | SNV, the SNV-A version used in the paper | `SNV/snv_adaptive.py` (on top of `SNV/snv_core.py`) |
| **Ours** | **MCL** | `MCL/mcl.py` |

WSN needs the task identity at test time, so it is evaluated only in Task-IL.
PEC is defined only for Class-IL. † marks methods designed jointly for
continual learning and machine unlearning.

## Layout

```
audited_gtep.py      GTEP protocol: splits, search spaces, one run, cost ledger hooks
snv_adaptive_run.py  runs SNV-A through the unchanged GTEP worker
campaign/
  run_campaign.py    full campaign: 30 configs x 3 seeds tuning, 3 clean D_E runs per winner
  build_report.py    metrics / cost / hyperparameter tables (CSV, XLSX, Markdown, LaTeX, HTML)
  snv_adaptive_compare.py
metrics.py           ACC, BWT, FWT, PS (+ P, S, AF)
audit_cost.py        cost ledger: GPU-hours, peak memory, parameters, GFLOPs, latency, energy
cost.py              lightweight cost tracker used by train.py
train.py             standalone single-method trainer (any dataset / scenario)
baselines.py         method registry
cl_base.py, models.py, datasets.py, training_policy.py, utils.py, inrun.py
<Method>/            one folder per method
docs/                PROTOCOL.md, METRICS.md, COSTS.md, BASELINES.md
scripts/smoke_test.sh
tests/
```

## Install

```bash
pip install -r requirements.txt       # Python 3.10, PyTorch 2.6 / CUDA 12.4 used for the paper
```

CIFAR-100 downloads into `./data` on first use. To use a different directory,
set `GTEP_DATA_ROOT=/path/to/data`.

## Quick check (runs on CPU, a few minutes)

```bash
bash scripts/smoke_test.sh            # every method, 2 tasks, 1 epoch, 64 samples
python -m pytest -q tests
```

On CPU the smoke script skips SNV-A, because its Shapley valuation takes
more than 30 minutes on ResNet-18 without a GPU. `tests/test_snv.py::TestSNVAdaptive`
covers SNV-A on CPU. Use `GTEP_DEVICE=cuda:0 bash scripts/smoke_test.sh` to include it.

## Reproduce the CIFAR-100 results

```bash
# Every method, Class-IL and Task-IL, on four GPUs (resumable: rerun the same command after an interruption)
python campaign/run_campaign.py --out runs/cifar100 --gpus 0 1 2 3

# Tables for the paper
python campaign/build_report.py --campaign runs/cifar100 --out reports/cifar100
```

To run a subset, use `--methods mcl snv` or `--scenarios class_il`.

A single run (one method, scenario, half and seed):

```bash
GTEP_PROTOCOL=legacy python audited_gtep.py --one --method mcl --scenario class_il \
    --half 2 --seed 42 --epochs 200 --patience 15 --out runs/mcl_cil_s42 \
    --config '{"lr":0.001,"lwf_lambda":1.0,"temperature":2.0,"mcl_density_alpha":0.25}'
```

SNV runs through `snv_adaptive_run.py` with the same flags, without `--one` and
`--method`.

The protocol is documented in [docs/PROTOCOL.md](docs/PROTOCOL.md), the
metric definitions in [docs/METRICS.md](docs/METRICS.md) and the cost
measurements in [docs/COSTS.md](docs/COSTS.md). Implementation notes for each
baseline are in [docs/BASELINES.md](docs/BASELINES.md), and the upstream
repositories they reference are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

Each experiment in the paper used a single NVIDIA RTX PRO 6000 GPU.
