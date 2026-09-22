#!/usr/bin/env bash
# Rebuild everything derived from a campaign: the selected hyperparameters, the
# imported D_E artifacts and the result tables.
#
#   bash scripts/export_cifar100_results.sh                       # re-tabulate what is here
#   CAMPAIGN=/path/to/campaign bash scripts/export_cifar100_results.sh   # import first
set -eu
cd "$(dirname "$0")/.."
if [ -n "${CAMPAIGN:-}" ]; then
  python hyperparameters/extract_cifar100_best.py --campaign "$CAMPAIGN" --out hyperparameters
  python results/import_cifar100.py --campaign "$CAMPAIGN"
fi
python results/make_cifar100_tables.py
echo "hyperparameters/cifar100_best.{json,md} and results/cifar100/*.{csv,md,json} rebuilt"
