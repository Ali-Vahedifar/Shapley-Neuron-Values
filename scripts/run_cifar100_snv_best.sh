#!/usr/bin/env bash
# The best version -- SNV-A with the selected CIFAR-100 configuration -- on the
# clean half, both scenarios, three seeds.  Equivalent to
# scripts/run_cifar100_selected.sh restricted to SNV.
#
#   bash scripts/run_cifar100_snv_best.sh
#   GPU=2 OUT=runs/snv_best bash scripts/run_cifar100_snv_best.sh
set -eu
cd "$(dirname "$0")/.."
METHODS=snv OUT=${OUT:-runs/cifar100_snv_best} exec bash scripts/run_cifar100_selected.sh "$@"
