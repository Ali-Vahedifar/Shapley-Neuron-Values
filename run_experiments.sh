#!/usr/bin/env bash
# Reproduce the paper's experiments.
#
#   Table 1   ImageNet-1k, 10 / 20 / 50 tasks, CIL and TIL, every method
#   Table 3   CIFAR-100 and TinyImageNet TIL, c in {0.03, 0.05, 0.1, 0.3, 0.5}
#   Fig. 4    weight-pruning efficiency, CIL, 10 tasks
#   GTEP      first-task grid search, run before anything else
#
# Usage:
#   bash run_experiments.sh                # everything
#   bash run_experiments.sh hparams        # phase 1 only
#   bash run_experiments.sh imagenet       # Table 1 only
#   bash run_experiments.sh sparsity       # Table 3 only
#   bash run_experiments.sh pruning        # Fig. 4 only
#
# ImageNet-1k must be present at $DATA_ROOT/imagenet/{train,val} as ImageFolder
# directories; it cannot be downloaded automatically.

set -euo pipefail

NUM_RUNS=${NUM_RUNS:-10}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
DATA_ROOT=${DATA_ROOT:-./data}
WORKERS=${WORKERS:-4}
STAGE=${1:-all}

BUFFER_FREE=(snv nfl+ dcnet nispa nfl pec spacenet lwf si ewc)
MEMORY_BASED=(dytox derpp icarl)
TIL_ONLY=(wsn)

common() {
  echo "--data_root $DATA_ROOT --output_dir $OUTPUT_DIR --gpu $GPU \
        --num_workers $WORKERS --num_runs $NUM_RUNS --verbose"
}

# --------------------------------------------------------------------------- #
# Phase 1: hyperparameters, chosen on task 1's validation split and then frozen
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "hparams" ]]; then
  echo "### GTEP phase 1: first-task grid search"
  for METHOD in "${BUFFER_FREE[@]}" "${MEMORY_BASED[@]}" "${TIL_ONLY[@]}"; do
    for DATASET in cifar100 tinyimagenet imagenet1k; do
      python hparam_search.py --method "$METHOD" --dataset "$DATASET" \
        --scenario class_il --data_root "$DATA_ROOT" --gpu "$GPU" \
        --num_workers "$WORKERS" || echo "  (skipped $METHOD/$DATASET)"
    done
  done
fi

# --------------------------------------------------------------------------- #
# Table 1: ImageNet-1k, 10 / 20 / 50 tasks
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "imagenet" ]]; then
  echo "### Table 1: ImageNet-1k"
  for TASKS in 10 20 50; do
    C=$(python -c "print(round(1.0/$TASKS, 4))")     # c = 1/T, per Section 2
    for SCENARIO in class_il task_il; do
      for METHOD in "${BUFFER_FREE[@]}" "${MEMORY_BASED[@]}"; do
        echo "  imagenet1k / $TASKS tasks / $SCENARIO / $METHOD"
        python train.py --method "$METHOD" --dataset imagenet1k \
          --num_tasks "$TASKS" --scenario "$SCENARIO" --sparsity "$C" \
          $(common) || echo "  (failed: $METHOD)"
      done
      if [[ "$SCENARIO" == "task_il" ]]; then
        for METHOD in "${TIL_ONLY[@]}"; do
          python train.py --method "$METHOD" --dataset imagenet1k \
            --num_tasks "$TASKS" --scenario task_il --sparsity "$C" \
            $(common) || echo "  (failed: $METHOD)"
        done
      fi
    done
  done
fi

# --------------------------------------------------------------------------- #
# Table 3: capacity budget sweep, CIFAR-100 and TinyImageNet, TIL
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "sparsity" ]]; then
  echo "### Table 3: capacity budget sweep"
  for DATASET in cifar100 tinyimagenet; do
    for TASKS in 10 20; do
      for C in 0.03 0.05 0.1 0.3 0.5; do
        for METHOD in snv wsn; do
          echo "  $DATASET / $TASKS tasks / c=$C / $METHOD"
          python train.py --method "$METHOD" --dataset "$DATASET" \
            --num_tasks "$TASKS" --scenario task_il --sparsity "$C" \
            $(common) || echo "  (failed: $METHOD)"
        done
      done
      for METHOD in nfl+ dcnet nispa spacenet lwf si ewc derpp icarl dytox; do
        python train.py --method "$METHOD" --dataset "$DATASET" \
          --num_tasks "$TASKS" --scenario task_il $(common) || true
      done
    done
  done

  echo "### CIFAR-100 / TinyImageNet, CIL"
  for DATASET in cifar100 tinyimagenet; do
    for TASKS in 10 20; do
      C=$(python -c "print(round(1.0/$TASKS, 4))")
      for METHOD in "${BUFFER_FREE[@]}" "${MEMORY_BASED[@]}"; do
        python train.py --method "$METHOD" --dataset "$DATASET" \
          --num_tasks "$TASKS" --scenario class_il --sparsity "$C" \
          $(common) || true
      done
    done
  done
fi

# --------------------------------------------------------------------------- #
# Fig. 4: weight-pruning efficiency
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "pruning" ]]; then
  echo "### Fig. 4: weight-pruning efficiency (CIL, 10 tasks)"
  for DATASET in cifar100 tinyimagenet imagenet1k; do
    for METHOD in snv nfl+ pec lwf ewc; do
      python pruning.py --method "$METHOD" --dataset "$DATASET" --num_tasks 10 \
        --scenario class_il --sparsity 0.1 --data_root "$DATA_ROOT" \
        --gpu "$GPU" --num_workers "$WORKERS" \
        --output_dir "$OUTPUT_DIR/pruning" || true
    done
  done
fi

echo "done -- results in $OUTPUT_DIR"
