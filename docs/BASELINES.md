# Baselines and implementation notes

Every method implements the `ContinualMethod` interface (`cl_base.py`), which
provides `train_task`, `evaluate` and `predict`. All methods are built by
`audited_gtep.make_method`, which calls `baselines.build_method` (`baselines/__init__.py`).

| Key | Method | Scenarios | Searched hyperparameters (besides lr) | Notes |
|---|---|---|---|---|
| `sgd` | Fine-tuning (lower bound) | CIL, TIL | none | |
| `joint` | Joint training (upper bound) | CIL, TIL | none | `baselines/joint_audited.py`, `JointPrefix` retrains on all seen tasks after each task. |
| `ewc` | EWC (Kirkpatrick et al., 2017) | CIL, TIL | `ewc_lambda`, `ewc_gamma` | Online EWC when γ < 1 |
| `si` | SI (Zenke et al., 2017) | CIL, TIL | `si_c`, `si_xi` | |
| `lwf` | LwF (Li & Hoiem, 2017) | CIL, TIL | `lwf_lambda`, `temperature` | |
| `wsn` | WSN (Kang et al., 2022) | **TIL only** | `wsn_density` | Needs the task identity at test time |
| `pec` | PEC (Zając et al., 2023) | **CIL only** | none (legacy protocol) | Official mechanism (a shared frozen teacher and independent per-class students) adapted to multi-epoch five-class tasks |
| `spacenet` | SpaceNet (Sokar et al., 2021) | CIL, TIL | `density_factor`, `rewire_fraction` | Its native MLP (input-400-400-classes; 3072-400-400-50 on a CIFAR-100 half) |
| `nispa` | NISPA (Gurbuz & Dovrolis, 2022) | CIL, TIL | `nispa_prune_perc`, `nispa_recovery_perc` | Native conv architecture (the fc1 fan-in follows the input size, so it also runs the 64px and 224px benchmarks). It uses its own phase/recovery stopping rule. |
| `uniclun` | UniCLUN† (Chatterjee et al., 2024) | CIL, TIL | `alpha1`, `alpha2`, `alpha3` | Re-implemented from the paper, because the upstream code omits the model module. The buffer holds 1,000 examples. |
| `snv` | SNV (SNV-A, `SNV/snv_adaptive.py`) | CIL, TIL | `truncation` | Run through `snv_adaptive_run.py` |

## SNV

`SNV/snv_core.py` contains the Shapley neuron-valuation learner. It covers
the truncated Monte-Carlo and reverse-TMC estimators, neuron masks and
consolidation. `SNV/snv_adaptive.py` (SNV-A) adds the following, each of
which can be switched off:

* **task-local training:** only the current head enters the loss;
* **routed Class-IL inference:** per-head log-softmax, picking the most
  confident subnetwork;
* **an adaptive subnetwork size:** the smallest prefix of neurons sorted by
  descending φ that recovers a coverage fraction of the full-model value;
* **frozen-norm evaluation and BatchNorm recalibration.**