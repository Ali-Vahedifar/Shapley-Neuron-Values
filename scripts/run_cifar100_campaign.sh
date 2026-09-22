#!/usr/bin/env bash
# The full CIFAR-100 GTEP campaign: every method, both scenarios, R = 30
# configurations x 3 seeds on D_HT, then the winner 3x on D_E.
# This is what produced results/cifar100/.  It takes days on four GPUs and is
# resumable: rerun the same command after an interruption.
#
#   bash scripts/run_cifar100_campaign.sh                  # GPUs 0 1 2 3
#   GPUS="2 3" OUT=runs/my_campaign bash scripts/run_cifar100_campaign.sh
set -eu
cd "$(dirname "$0")/.."
export GTEP_PROTOCOL=${GTEP_PROTOCOL:-legacy}
export GTEP_DATA_ROOT=${GTEP_DATA_ROOT:-./data}
OUT=${OUT:-runs/cifar100}
GPUS=${GPUS:-0 1 2 3}
python campaign/run_campaign.py --out "$OUT" --gpus $GPUS --dataset cifar100 \
    --epochs "${EPOCHS:-200}" --patience "${PATIENCE:-15}" --rounds "${ROUNDS:-30}" "$@"
echo "campaign in $OUT; build the tables with scripts/export_cifar100_results.sh"
