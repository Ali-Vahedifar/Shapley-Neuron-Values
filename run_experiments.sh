#!/bin/bash
# Run Script for SNV Continual Learning Experiments
# 
# This script runs the complete set of experiments as described in the paper:
# - PMNIST: 10 tasks, batch_size=10, epochs=20
# - CIFAR-100: 10 and 20 tasks, batch_size=64, epochs=200
# - TinyImageNet: 10 and 20 tasks, batch_size=64, epochs=200
#
# Sparsity ratios: c ∈ {0.03, 0.05, 0.1, 0.3, 0.5}
# Scenarios: Class-IL and Task-IL
#
# Anonymous submission for ICML 2026.

# Configuration
NUM_RUNS=10
GPU=0
OUTPUT_DIR="./results"
DATA_ROOT="./data"

# Learning rates (from paper Table 2)
LR=0.001

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "SNV Continual Learning Experiments"
echo "=========================================="

# ==========================================
# PMNIST Experiments
# ==========================================
echo ""
echo "Running PMNIST experiments..."
echo "=========================================="

for SPARSITY in 0.03 0.05 0.1 0.3 0.5; do
    for SCENARIO in class_il task_il; do
        echo "PMNIST - c=$SPARSITY - $SCENARIO"
        python train.py \
            --dataset pmnist \
            --num_tasks 10 \
            --scenario $SCENARIO \
            --sparsity $SPARSITY \
            --lr $LR \
            --batch_size 10 \
            --epochs 20 \
            --num_runs $NUM_RUNS \
            --gpu $GPU \
            --data_root $DATA_ROOT \
            --output_dir $OUTPUT_DIR \
            --verbose
    done
done

# ==========================================
# CIFAR-100 Experiments
# ==========================================
echo ""
echo "Running CIFAR-100 experiments..."
echo "=========================================="

for NUM_TASKS in 10 20; do
    for SPARSITY in 0.03 0.05 0.1 0.3 0.5; do
        for SCENARIO in class_il task_il; do
            echo "CIFAR-100 - $NUM_TASKS tasks - c=$SPARSITY - $SCENARIO"
            python train.py \
                --dataset cifar100 \
                --num_tasks $NUM_TASKS \
                --scenario $SCENARIO \
                --sparsity $SPARSITY \
                --lr $LR \
                --batch_size 64 \
                --epochs 200 \
                --patience 20 \
                --num_runs $NUM_RUNS \
                --gpu $GPU \
                --data_root $DATA_ROOT \
                --output_dir $OUTPUT_DIR \
                --verbose
        done
    done
done

# ==========================================
# TinyImageNet Experiments
# ==========================================
echo ""
echo "Running TinyImageNet experiments..."
echo "=========================================="

for NUM_TASKS in 10 20; do
    for SPARSITY in 0.03 0.05 0.1 0.3 0.5; do
        for SCENARIO in class_il task_il; do
            echo "TinyImageNet - $NUM_TASKS tasks - c=$SPARSITY - $SCENARIO"
            python train.py \
                --dataset tinyimagenet \
                --num_tasks $NUM_TASKS \
                --scenario $SCENARIO \
                --sparsity $SPARSITY \
                --lr $LR \
                --batch_size 64 \
                --epochs 200 \
                --patience 20 \
                --num_runs $NUM_RUNS \
                --gpu $GPU \
                --data_root $DATA_ROOT \
                --output_dir $OUTPUT_DIR \
                --verbose
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
