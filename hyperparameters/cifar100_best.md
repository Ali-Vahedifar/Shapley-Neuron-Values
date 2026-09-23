# CIFAR-100: selected hyperparameters

Source campaign: `/root/Snv_continual_learning/gtep_paper_first_20260906`  
Exported: 2026-09-22  
Protocol: GTEP, 10 tasks, ResNet-18, tuned on D_HT (half 1, 3 seeds),
reported on the three clean D_E runs (half 2, seeds 42/43/44).

ACC is the mean over the clean D_E seeds, +/- the standard deviation.
`snv` is SNV-A (the reported SNV); `snv_dense` is the dense SNV variant,
kept only to document that its Class-IL search collapsed to chance.

The search space and all trials: `results/cifar100/search_space.md`.
The runs these numbers come from: `results/cifar100/`.

## class_il

| method | ACC (D_E) | BWT | search ACC (D_HT) | hyperparameters |
|---|---:|---:|---:|---|
| joint | 0.6543 ± 0.0240 | -0.0701 | 0.7471 | `lr=0.00070638` |
| snv | 0.3618 ± 0.0206 | -0.1862 | 0.4980 | `adaptive=True, adaptive_coverage=0.9, adaptive_rule=coverage, bn_recal=True, bn_recal_mode=batch, frozen_norm_eval=True, lr=0.00129816, rot_aux=1, routing=rot_energy_z, task_local=True, truncation=0.05` |
| pec | 0.3284 ± 0.0125 | -0.1270 | 0.3842 | `lr=0.000139596` |
| uniclun | 0.3275 ± 0.0404 | -0.5884 | 0.4509 | `alpha1=0.39768, alpha2=3.87935, alpha3=2.50028, lr=0.000806063` |
| lwf | 0.1006 ± 0.0179 | -0.8002 | 0.1387 | `lr=0.00011095, lwf_lambda=0.838283, temperature=2` |
| sgd | 0.0872 ± 0.0070 | -0.8585 | 0.1351 | `lr=0.00145613` |
| si | 0.0856 ± 0.0049 | -0.8510 | 0.1300 | `lr=0.000444289, si_c=0.0283496, si_xi=0.00200387` |
| ewc | 0.0839 ± 0.0066 | -0.8446 | 0.1329 | `ewc_gamma=0.970149, ewc_lambda=1.74855, lr=0.00212963` |
| nispa | 0.0824 ± 0.0054 | -0.7889 | 0.1223 | `lr=0.00142628, nispa_prune_perc=79.917, nispa_recovery_perc=4.90502` |
| spacenet | 0.0677 ± 0.0043 | -0.6546 | 0.0986 | `density_factor=1.5, lr=0.000689093, rewire_fraction=0.1` |
| snv_dense | not evaluated | - | 0.0444 | `lr=0.000736701, truncation=0.05` |

## task_il

| method | ACC (D_E) | BWT | search ACC (D_HT) | hyperparameters |
|---|---:|---:|---:|---|
| joint | 0.8789 ± 0.0125 | 0.0180 | 0.9084 | `lr=0.000736701` |
| wsn | 0.8470 ± 0.0184 | 0.0000 | 0.8951 | `lr=0.00131982, wsn_density=0.3` |
| snv | 0.8175 ± 0.0157 | 0.0000 | 0.8747 | `adaptive=True, adaptive_coverage=0.9, adaptive_rule=coverage, bn_recal=True, bn_recal_mode=batch, frozen_norm_eval=True, lr=0.00129816, rot_aux=0, routing=maxprob, task_local=True, truncation=0.05` |
| nispa | 0.8002 ± 0.0155 | 0.0000 | 0.8393 | `lr=0.00189595, nispa_prune_perc=79.3099, nispa_recovery_perc=3.19098` |
| lwf | 0.7781 ± 0.0249 | -0.0796 | 0.8387 | `lr=0.00172989, lwf_lambda=0.983624, temperature=4` |
| ewc | 0.7088 ± 0.0265 | -0.0579 | 0.7680 | `ewc_gamma=0.91238, ewc_lambda=2029.59, lr=0.00070638` |
| uniclun | 0.6946 ± 0.0270 | -0.2136 | 0.7871 | `alpha1=0.13353, alpha2=0.131584, alpha3=0.258177, lr=0.00124592` |
| si | 0.6734 ± 0.0114 | -0.1140 | 0.7251 | `lr=0.000130616, si_c=0.332895, si_xi=0.000118848` |
| spacenet | 0.6361 ± 0.0132 | -0.0162 | 0.6489 | `density_factor=0.5, lr=0.00146553, rewire_fraction=0.3` |
| sgd | 0.5081 ± 0.0299 | -0.3444 | 0.5691 | `lr=0.000123927` |

