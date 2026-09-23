#!/usr/bin/env bash
# Run every method through the real GTEP worker on a tiny slice of the dataset
# (2 tasks, 1 epoch, 64 examples per split).  Checks the pipeline, not accuracy.
#   bash scripts/smoke_test.sh            # CPU, CIFAR-100
#   GTEP_DEVICE=cuda:0 bash scripts/smoke_test.sh
#   DATASET=cifar20 bash scripts/smoke_test.sh
set -u
cd "$(dirname "$0")/.."
export GTEP_PROTOCOL=legacy
export GTEP_DEVICE=${GTEP_DEVICE:-cpu}
export GTEP_DATA_ROOT=${GTEP_DATA_ROOT:-./data}
DATASET=${DATASET:-cifar100}
[ "$GTEP_DEVICE" = cpu ] && export CUDA_VISIBLE_DEVICES=''
OUT=${OUT:-runs/smoke}
COMMON="--half 1 --seed 42 --epochs 1 --patience 1 --tasks 2 --smoke_samples 64 --dataset $DATASET"
status=0
run() {
  local method=$1 scenario=$2 config=$3
  rm -rf "$OUT/$method-$scenario"
  if python audited_gtep.py --one --method "$method" --scenario "$scenario" --config "$config" \
       --out "$OUT/$method-$scenario" $COMMON > "$OUT/$method-$scenario.log" 2>&1; then
    echo "PASS $method $scenario"
  else
    echo "FAIL $method $scenario (see $OUT/$method-$scenario.log)"; status=1
  fi
}
mkdir -p "$OUT"
run sgd      class_il '{"lr":0.001}'
run joint    class_il '{"lr":0.001}'
run ewc      class_il '{"lr":0.001,"ewc_lambda":100,"ewc_gamma":1.0}'
run si       class_il '{"lr":0.001,"si_c":0.1,"si_xi":0.001}'
run lwf      class_il '{"lr":0.001,"lwf_lambda":1.0,"temperature":2.0}'
run wsn      task_il  '{"lr":0.001,"wsn_density":0.5}'
run pec      class_il '{"lr":0.001}'
run spacenet class_il '{"lr":0.001,"density_factor":1.0,"rewire_fraction":0.2}'
run nispa    class_il '{"lr":0.001,"nispa_prune_perc":90,"nispa_recovery_perc":2.5}'
run uniclun  class_il '{"lr":0.001,"alpha1":1.0,"alpha2":1.0,"alpha3":1.0}'
run lwf      task_il  '{"lr":0.001,"lwf_lambda":1.0,"temperature":2.0}'
# SNV-A: the Shapley valuation is the expensive step.  On CPU with ResNet-18
# it takes well over 30 minutes even at this size, so it runs only on a GPU;
# tests/test_snv.py::TestSNVAdaptive covers SNV-A on CPU in seconds.
if [ "$GTEP_DEVICE" = cpu ]; then echo "SKIP snv class_il (CPU; set GTEP_DEVICE=cuda:0)"; exit $status; fi
rm -rf "$OUT/snv-class_il"
if python snv_adaptive_run.py --scenario class_il --out "$OUT/snv-class_il" --epochs 2 --patience 1 \
     --tasks 2 --smoke_samples 64 --seed 42 --dataset "$DATASET" \
     --config '{"lr":0.001,"truncation":0.1,"max_permutations":2,"task_local":true,"frozen_norm_eval":true,"bn_recal":true,"routing":"maxprob","adaptive":true,"adaptive_rule":"coverage","adaptive_coverage":0.9}' \
     > "$OUT/snv-class_il.log" 2>&1; then echo "PASS snv class_il"; else echo "FAIL snv class_il"; status=1; fi
exit $status
