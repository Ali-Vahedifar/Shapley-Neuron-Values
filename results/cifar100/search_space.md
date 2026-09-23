# CIFAR-100 search space and every trial

R = 30 configurations per method, drawn from the space below with sample
seed 7 (`SPACE` in `audited_gtep.py`, `GTEP_PROTOCOL=legacy`), each run on
D_HT with seeds 42/43/44. The winner is the highest mean HARMONIC.

## Space

| method | parameter | distribution |
|---|---|---|
| ewc | lr | `('logu', 0.0001, 0.01)` |
| ewc | ewc_lambda | `('logu', 1, 10000.0)` |
| ewc | ewc_gamma | `('u', 0.9, 1.0)` |
| joint | lr | `('logu', 0.0001, 0.01)` |
| lwf | lr | `('logu', 0.0001, 0.01)` |
| lwf | lwf_lambda | `('logu', 0.1, 10)` |
| lwf | temperature | `('choice', [1.0, 2.0, 3.0, 4.0])` |
| nispa | lr | `('logu', 0.0001, 0.01)` |
| nispa | nispa_prune_perc | `('u', 70, 95)` |
| nispa | nispa_recovery_perc | `('u', 1, 5)` |
| pec | lr | `('logu', 0.0001, 0.01)` |
| sgd | lr | `('logu', 0.0001, 0.01)` |
| si | lr | `('logu', 0.0001, 0.01)` |
| si | si_c | `('logu', 0.01, 10)` |
| si | si_xi | `('logu', 0.0001, 0.01)` |
| snv | lr | `('logu', 0.0001, 0.01)` |
| snv | truncation | `('choice', [0.05, 0.1, 0.2])` |
| snv_dense | lr | `('logu', 0.0001, 0.01)` |
| snv_dense | truncation | `('choice', [0.05, 0.1, 0.2])` |
| spacenet | lr | `('logu', 0.0001, 0.01)` |
| spacenet | density_factor | `('choice', [0.5, 1.0, 1.5])` |
| spacenet | rewire_fraction | `('choice', [0.1, 0.2, 0.3])` |
| uniclun | lr | `('logu', 0.0001, 0.01)` |
| uniclun | alpha1 | `('logu', 0.1, 10)` |
| uniclun | alpha2 | `('logu', 0.1, 10)` |
| uniclun | alpha3 | `('logu', 0.1, 10)` |
| wsn | lr | `('logu', 0.0001, 0.01)` |
| wsn | wsn_density | `('choice', [0.1, 0.3, 0.5, 0.7])` |

## Trials scored on D_HT

| method | scenario | trials | best score | worst score |
|---|---|---:|---:|---:|
| ewc | class_il | 30 | 0.1329 | 0.1009 |
| ewc | task_il | 30 | 0.7680 | 0.4944 |
| joint | class_il | 30 | 0.7471 | 0.6580 |
| joint | task_il | 30 | 0.9084 | 0.8489 |
| lwf | class_il | 30 | 0.1387 | 0.1264 |
| lwf | task_il | 30 | 0.8387 | 0.6954 |
| nispa | class_il | 30 | 0.1223 | 0.0827 |
| nispa | task_il | 30 | 0.8393 | 0.6447 |
| pec | class_il | 30 | 0.3842 | 0.3367 |
| sgd | class_il | 30 | 0.1351 | 0.1248 |
| sgd | task_il | 30 | 0.5691 | 0.4805 |
| si | class_il | 30 | 0.1300 | 0.0970 |
| si | task_il | 30 | 0.7251 | 0.5494 |
| snv | class_il | 23 | 0.4980 | 0.4100 |
| snv | task_il | 23 | 0.8747 | 0.8083 |
| snv_dense | class_il | 18 | 0.0444 | 0.0318 |
| spacenet | class_il | 30 | 0.0986 | 0.0628 |
| spacenet | task_il | 30 | 0.6489 | 0.5695 |
| uniclun | class_il | 30 | 0.4509 | 0.0558 |
| uniclun | task_il | 30 | 0.7871 | 0.3905 |
| wsn | task_il | 30 | 0.8951 | 0.8651 |

Every trial with its configuration and score is in `search_space.json`.

