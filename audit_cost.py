"""Explicit, measured cost ledger. Unsupported sensors are null, never zero.

NVML energy/power/utilization are device-wide; contamination is recorded.
PyTorch FLOPs count supported operators, not all machine instructions.
"""
import contextlib
import os
import time
import threading
from unittest.mock import patch
import numpy as np
import psutil
import torch


def tensors_in(value, visited=None):
    visited = set() if visited is None else visited
    if id(value) in visited:
        return
    visited.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, torch.nn.Module):
        yield from value.parameters()
        yield from value.buffers()
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors_in(v, visited)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from tensors_in(v, visited)
    elif hasattr(value, '__dict__') and value.__class__.__module__.startswith(
            ('cl_base', 'SNV.', 'PEC.', 'UniCLUN.', 'SpaceNet.', 'NISPA.')):
        yield from tensors_in(vars(value), visited)


def tensor_storage_bytes(value):
    storages = {}
    for t in tensors_in(value):
        if t.device.type == 'meta':
            continue
        s = t.untyped_storage()
        storages[(str(t.device), s.data_ptr())] = s.nbytes()
    return sum(storages.values())


def model_inventory(method):
    modules = [v for v in vars(method).values() if isinstance(v, torch.nn.Module)]
    parameters = {id(p): p for m in modules for p in m.parameters()}
    return {'resident_parameter_count': sum(p.numel() for p in parameters.values()),
            'requires_grad_parameter_count': sum(p.numel() for p in parameters.values() if p.requires_grad),
            'resident_parameter_bytes': sum(p.numel()*p.element_size() for p in parameters.values()),
            'resident_tensor_storage_bytes': tensor_storage_bytes(vars(method)),
            'replay_examples': len(method.buffer) if hasattr(method, 'buffer') else
                sum(len(x) for x in getattr(method, 'exemplars', {}).values())}


class Sensors(threading.Thread):
    def __init__(self, device, period=.25):
        super().__init__(daemon=True)
        self.period, self.event, self.samples = period, threading.Event(), []
        self.process = psutil.Process()
        self.nvml = self.handle = None
        self.errors = []
        if device.type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', str(device.index or 0)).split(',')
                ident = visible[device.index or 0]
                self.handle = (pynvml.nvmlDeviceGetHandleByIndex(int(ident)) if ident.isdigit()
                               else pynvml.nvmlDeviceGetHandleByUUID(ident))
                self.nvml = pynvml
            except Exception as e:
                self.errors.append('NVML: ' + str(e))

    def energy(self):
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000
        except Exception:
            return None

    def sample(self):
        row = {'time': time.perf_counter()}
        processes = [self.process] + self.process.children(recursive=True)
        rss = cpu = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
                ct = p.cpu_times()
                cpu += ct.user + ct.system
            except psutil.Error:
                pass
        row.update(process_tree_rss_bytes=rss, process_tree_cpu_seconds=cpu,
                   host_cpu_percent=psutil.cpu_percent())
        if self.nvml:
            n, h = self.nvml, self.handle
            fields = {
                'gpu_util_percent': lambda: n.nvmlDeviceGetUtilizationRates(h).gpu,
                'gpu_memory_used_bytes': lambda: n.nvmlDeviceGetMemoryInfo(h).used,
                'power_watts': lambda: n.nvmlDeviceGetPowerUsage(h)/1000,
                'temperature_c': lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                'sm_clock_mhz': lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                'other_compute_pids': lambda: [p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)
                                               if p.pid not in {q.pid for q in processes}],
            }
            for name, fn in fields.items():
                try:
                    row[name] = fn()
                except Exception:
                    row[name] = None
        self.samples.append(row)

    def run(self):
        while not self.event.is_set():
            self.sample()
            self.event.wait(self.period)


class CostLedger:
    def __init__(self, device):
        self.device, self.records = device, []

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def phase(self, name, method=None, count_flops=False):
        self.sync()
        sensor = Sensors(self.device)
        energy_start = sensor.energy()
        process = psutil.Process()
        cpu0, io0 = process.cpu_times(), process.io_counters()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        record = {'phase': name, 'optimizer_steps': 0, 'optimizer_state_peak_bytes': 0}
        def instrument(original):
            def step(optimizer, *args, **kw):
                result = original(optimizer, *args, **kw)
                record['optimizer_steps'] += 1
                record['optimizer_state_peak_bytes'] = max(record['optimizer_state_peak_bytes'],
                                                            tensor_storage_bytes(optimizer.state))
                return result
            return step
        counter = None
        if count_flops:
            from torch.utils.flop_counter import FlopCounterMode
            counter = FlopCounterMode(display=False)
        start = time.perf_counter()
        sensor.start()
        try:
            with contextlib.ExitStack() as stack:
                if counter:
                    stack.enter_context(counter)
                for optimizer in (torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW):
                    stack.enter_context(patch.object(optimizer, 'step', instrument(optimizer.step)))
                yield record
        finally:
            self.sync()
            record['wall_seconds'] = time.perf_counter() - start
            sensor.event.set()
            sensor.join(timeout=2)
            energy_end = sensor.energy()
            cpu1, io1 = process.cpu_times(), process.io_counters()
            record['main_process_cpu_seconds'] = cpu1.user+cpu1.system-cpu0.user-cpu0.system
            record['disk_read_bytes'] = io1.read_bytes - io0.read_bytes
            record['disk_write_bytes'] = io1.write_bytes - io0.write_bytes
            record['gpu_energy_joules_device_counter'] = (energy_end-energy_start
                if energy_start is not None and energy_end is not None else None)
            record['gpu_energy_joules_power_integral'] = None
            powered = [s for s in sensor.samples if s.get('power_watts') is not None]
            if len(powered) >= 2:
                record['gpu_energy_joules_power_integral'] = float(np.trapz(
                    [s['power_watts'] for s in powered], [s['time'] for s in powered]))
            record['gpu_exclusive_observed'] = (all(not s['other_compute_pids'] for s in sensor.samples)
                if sensor.samples and all(s.get('other_compute_pids') is not None for s in sensor.samples) else None)
            record['sensor_errors'] = sensor.errors
            record['sensor_samples'] = sensor.samples
            record['supported_operator_flops'] = counter.get_total_flops() if counter else None
            record['flop_instrumentation_during_timing'] = count_flops
            record['gpu_peak_allocated_bytes'] = (torch.cuda.max_memory_allocated(self.device)
                                                  if self.device.type == 'cuda' else None)
            record['gpu_peak_reserved_bytes'] = (torch.cuda.max_memory_reserved(self.device)
                                                 if self.device.type == 'cuda' else None)
            if method is not None:
                record['state'] = model_inventory(method)
            self.records.append(record)

    def inference(self, predict, x, repeats=30, warmup=5):
        x = x.to(self.device)
        with torch.no_grad():
            for _ in range(warmup):
                predict(x)
            self.sync()
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                predict(x)
                self.sync()
                timings.append(1000*(time.perf_counter()-start))
            # Count one additional pass, outside the latency measurements.
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as flops:
                predict(x)
        return {'batch_size': len(x), 'repeats': repeats, 'warmup': warmup,
                'scope': 'device-resident inputs; host dispatch plus synchronized device execution',
                'batch_latency_ms_mean': float(np.mean(timings)),
                'inference_ms_per_sample': float(np.mean(timings))/len(x),
                'supported_operator_gflops_per_sample': flops.get_total_flops()/len(x)/1e9,
                'flop_definition': 'PyTorch supported operators, multiply-add counts as 2; extra pass outside latency timer',
                'batch_latency_ms_p50': float(np.percentile(timings, 50)),
                'batch_latency_ms_p95': float(np.percentile(timings, 95)),
                'batch_latency_ms_p99': float(np.percentile(timings, 99)),
                'throughput_samples_per_second': len(x)*1000/float(np.mean(timings))}


def summarize(records, inference, checkpoint_bytes):
    training=[r for r in records if '/training' in r['phase']]
    samples=[s for r in records for s in r['sensor_samples']]
    def peak(key):
        values=[r[key] for r in records if r.get(key) is not None]
        return max(values) if values else None
    def sensor_peak(key):
        values=[s[key] for s in samples if s.get(key) is not None]
        return max(values) if values else None
    energy=[r['gpu_energy_joules_device_counter'] for r in records]
    util=[s['gpu_util_percent'] for s in samples if s.get('gpu_util_percent') is not None]
    states=[r['state'] for r in records if 'state' in r]
    return dict(train_minutes_per_task=[r['wall_seconds']/60 for r in training],
        train_minutes_per_task_mean=float(np.mean([r['wall_seconds']/60 for r in training])),
        training_gpu_hours=sum(r['wall_seconds'] for r in training)/3600,
        accounted_run_gpu_hours=sum(r['wall_seconds'] for r in records)/3600,
        gpu_peak_allocated_bytes=peak('gpu_peak_allocated_bytes'),
        gpu_peak_reserved_bytes=peak('gpu_peak_reserved_bytes'),
        gpu_peak_device_used_bytes=sensor_peak('gpu_memory_used_bytes'),
        process_tree_peak_rss_bytes=sensor_peak('process_tree_rss_bytes'),
        optimizer_state_peak_bytes=peak('optimizer_state_peak_bytes'),
        gpu_util_percent_sample_mean=float(np.mean(util)) if util else None,
        gpu_util_percent_p95=float(np.percentile(util,95)) if util else None,
        energy_joules_device_counter=sum(energy) if all(e is not None for e in energy) else None,
        gpu_exclusive_observed=all(r['gpu_exclusive_observed'] is True for r in records),
        final_state=states[-1] if states else None,
        checkpoint_bytes=checkpoint_bytes,inference=inference,
        notes=['One process per GPU; other GPUs share host CPU, RAM, PCIe and power budget.',
               'CPU RSS sums process tree and may double-count shared pages.',
               'Training phases include validation and method consolidation; checkpoints/evaluation separately accounted.',
               'Dense masked operators have dense FLOPs unless executed by an actual sparse kernel.'])
