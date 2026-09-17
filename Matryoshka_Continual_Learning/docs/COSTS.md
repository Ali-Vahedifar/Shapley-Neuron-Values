# Measurement costs

Costs are recorded by `audit_cost.CostLedger` inside every run (`costs.json`,
plus `cost_summary` in `result.json`). Only the three D_E winner runs per
method and scenario are reported, and each must have held the GPU exclusively
(see PROTOCOL.md).

| Reported quantity | Field(s) | How it is measured |
|---|---|---|
| GPU hours | `training_gpu_hours`, `accounted_run_gpu_hours` | Wall time of the training phases (including internal validation and consolidation). The accounted figure adds setup, evaluation, checkpointing and inference for that run. |
| Training time | `train_minutes_per_task`, `train_minutes_per_task_mean` | Timed per task, with CUDA synchronisation. |
| Peak GPU memory | `gpu_peak_allocated_bytes`, `gpu_peak_reserved_bytes`, `gpu_peak_device_used_bytes` | PyTorch allocator peaks and NVML device memory |
| Memory (host) | `process_tree_peak_rss_bytes`, `optimizer_state_peak_bytes` | psutil process tree and optimiser state |
| Parameters | `final_state.resident_parameter_count/bytes`, `requires_grad_parameter_count`, `checkpoint_bytes` | Every parameter the method keeps (including teachers and masks), trainable parameters and on-disk size |
| Inference GFLOPs | `inference[*].supported_operator_gflops_per_sample` | `torch.utils.flop_counter.FlopCounterMode` counts supported operators, with a multiply-add counted as 2 FLOPs. This runs in a separate pass outside the timer. |
| Inference time | `inference[*].inference_ms_per_sample`, `throughput_samples_per_second` | Batch sizes 1 and 64, device-resident inputs, 5 warm-up and 10 timed repeats per task, averaged over the 10 tasks |
| GPU utilisation / energy | `gpu_util_percent_sample_mean`, `gpu_util_percent_p95`, `energy_joules_device_counter` | A 4 Hz NVML sampler and the device energy counter |
| Exclusivity | `gpu_exclusive_observed` | NVML sees no other compute process on the device during any phase. |

**Training FLOPs.** Pass `--flops` to `audited_gtep.py --one` to count FLOPs
during training. Because the instrumentation then runs inside the timed phase,
do this in a separate run, never in the run whose times are reported.
`build_report.py` rejects a cost run in which FLOP counting overlapped timing.

`campaign/build_report.py` turns these records into the `Costs_summary`,
`Costs_per_seed`, `Cost_phases` and `Inference_per_task` tables.
