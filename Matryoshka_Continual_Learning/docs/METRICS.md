# Continual-learning metrics

The metrics are implemented in `metrics.py` (`ContinualLearningMetrics`).
`A[i, j]` is the accuracy on task j after training on task i, with T = 10
tasks. The formulas use 1-based indices; the code is 0-based.

| Metric | Definition |
|---|---|
| **ACC** (final average accuracy) | `ACC = (1/T) Σ_t A[T, t]` |
| **BWT** (backward transfer) | `BWT = (1/(T−1)) Σ_{t<T} (A[T, t] − A[t, t])` |
| **FWT** (forward transfer) | `FWT = (1/(T−1)) Σ_{t≥2} (A[t−1, t] − b_t)`, where `b_t` is the accuracy of the randomly initialised model on task t |
| **P** (plasticity) | `P = (1/(T−1)) Σ_{t≥2} (A[t, t] − A[t−1, t]) / (1 − A[t−1, t])` |
| **S** (stability) | `S = 1 − (1/(T−1)) Σ_{t<T} (A[t, t] − A[T, t]) = 1 + BWT` |
| **PS** (harmonic plasticity–stability) | `PS = 2·P⁺·S⁺ / (P⁺ + S⁺)`, with `P⁺ = max(P, 0)` and `S⁺ = max(S, 0)` |
| AF (average forgetting, supplementary) | `(1/(T−1)) Σ_{t<T} max_{k≥t} A[k, t] − A[T, t]` |
| AvgAcc (supplementary) | the mean over stages t of the average accuracy on tasks 1..t |
| HARMONIC (selection only) | `2·ACC·AvgAcc / (ACC + AvgAcc)` |

Notes:

* PS is bounded by plasticity. A method with no forgetting has S = 1 and
  PS = 2P/(P+1) < 1.
* FWT and P use the superdiagonal `A[t−1, t]` (task t before it is trained).
  In Class-IL this is evaluated in the full 50-class output space.
* Joint training retrains from scratch on each seen-task prefix. Its
  BWT/FWT/PS are reported as logged, but they describe a reference trajectory,
  not forgetting.
* Reporting: the mean ± sample SD over seeds 42/43/44. ACC is reported in %,
  BWT/FWT/AF in percentage points, and PS/P/S are dimensionless.
* The intransigence measure `I` exists in `metrics.py` for completeness but is
  not reported.
