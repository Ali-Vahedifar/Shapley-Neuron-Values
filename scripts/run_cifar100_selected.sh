#!/usr/bin/env bash
# Rerun every method's *selected* CIFAR-100 configuration on the clean half
# (D_E, seeds 42/43/44).  The configurations come from
# hyperparameters/cifar100_best.json -- no search, just the winners.
#
#   bash scripts/run_cifar100_selected.sh                    # all methods
#   METHODS="snv lwf" SEEDS="42" bash scripts/run_cifar100_selected.sh
#   OUT=runs/selected GPU=2 bash scripts/run_cifar100_selected.sh
set -eu
cd "$(dirname "$0")/.."
export GTEP_PROTOCOL=${GTEP_PROTOCOL:-legacy}
export GTEP_DATA_ROOT=${GTEP_DATA_ROOT:-./data}
export GTEP_DEVICE=${GTEP_DEVICE:-cuda:0}
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
OUT=${OUT:-runs/cifar100_selected}
SEEDS=${SEEDS:-42 43 44}
METHODS=${METHODS:-snv joint sgd ewc si lwf wsn pec spacenet nispa uniclun}
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-15}
mkdir -p "$OUT"

for method in $METHODS; do
  for scenario in class_il task_il; do
    # WSN needs the task identity (TIL only); PEC is defined for CIL only.
    [ "$method" = wsn ] && [ "$scenario" = class_il ] && continue
    [ "$method" = pec ] && [ "$scenario" = task_il ] && continue
    config=$(python - "$method" "$scenario" <<'PY'
import json, sys
from hyperparameters import best_config
try:
    print(json.dumps(best_config(sys.argv[1], sys.argv[2])))
except KeyError:
    print('')
PY
)
    [ -z "$config" ] && { echo "SKIP $method $scenario (no selected config)"; continue; }
    for seed in $SEEDS; do
      dir="$OUT/${method}_${scenario}_s${seed}"
      echo "== $method $scenario seed $seed"
      if [ "$method" = snv ]; then
        # SNV means SNV-A, which is built by its own runner.
        python snv_adaptive_run.py --scenario "$scenario" --dataset cifar100 \
            --half 2 --seed "$seed" --epochs "$EPOCHS" --patience "$PATIENCE" \
            --out "$dir" --config "$config"
      else
        python audited_gtep.py --one --method "$method" --scenario "$scenario" \
            --dataset cifar100 --half 2 --seed "$seed" --epochs "$EPOCHS" \
            --patience "$PATIENCE" --out "$dir" --config "$config"
      fi
    done
  done
done
echo "runs in $OUT"
