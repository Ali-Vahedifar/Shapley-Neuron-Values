"""
Cost instrumentation: train time, GPU utilisation, GPU memory, and GFLOPs.

Reported alongside ACC/BWT/PS so the accuracy table and the cost table come from
the same run rather than from separate profiling passes.

  TRAIN_TIME_MIN  wall-clock minutes inside train_task(), summed over tasks.
                  Excludes evaluation and checkpointing.  For SNV this DOES
                  include Shapley estimation, which runs inside train_task();
                  VALUATION_MIN reports that portion separately, so SNV's
                  optimisation-only cost is TRAIN_TIME_MIN - VALUATION_MIN.
  GPU_UTIL_MEAN   mean SM utilisation (%) sampled at 2 Hz while training.
  GPU_MEM_PEAK_MB peak torch-allocated device memory (MiB).
  GFLOPS_FWD      forward-pass GFLOPs for a single sample, counted once on the
                  final model with torch.utils.flop_counter.
"""

import os
import threading
import time
from typing import Dict, Optional

import torch


class _UtilSampler(threading.Thread):
    """Samples SM utilisation for one device until stopped."""

    def __init__(self, device_index: int, period: float = 0.5):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.period = period
        self._stop_evt = threading.Event()
        self.samples = []
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            # CUDA remaps logical ordinals under CUDA_VISIBLE_DEVICES; NVML does
            # not.  Resolve logical cuda:0 back to the physical index/UUID or
            # workers 1..N all sample physical GPU 0.
            visible = [item.strip() for item in
                       os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
                       if item.strip()]
            identifier = (visible[device_index]
                          if device_index < len(visible) else str(device_index))
            if identifier.isdigit():
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(identifier))
            else:
                self._handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
        except Exception:
            self._pynvml = None

    def run(self):
        if self._handle is None:
            return
        while not self._stop_evt.is_set():
            try:
                u = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples.append(float(u.gpu))
            except Exception:
                pass
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()


class CostTracker:
    """Accumulates cost measurements across the tasks of one run."""

    def __init__(self, device: torch.device):
        self.device = device
        self.train_seconds = 0.0
        self.task_seconds = []
        self.valuation_seconds = 0.0
        self.prior_peak_mb = 0.0
        self._util_samples = []
        self._sampler: Optional[_UtilSampler] = None
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    # -- timed sections ---------------------------------------------------- #
    def start_train(self):
        self._t0 = time.time()
        if self.device.type == 'cuda':
            self._sampler = _UtilSampler(self.device.index or 0)
            self._sampler.start()

    def stop_train(self):
        elapsed = time.time() - self._t0
        self.train_seconds += elapsed
        self.task_seconds.append(elapsed)
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=2.0)
            self._util_samples.extend(self._sampler.samples)
            self._sampler = None

    def add_valuation(self, seconds: float):
        self.valuation_seconds += seconds

    # -- one-shot measurements --------------------------------------------- #
    def gflops_forward(self, model, sample_input, forward_fn=None) -> float:
        """Forward GFLOPs for a single sample.

        ``forward_fn`` is used for task-aware methods.  Counting ``model(x)``
        in TIL would execute every head even though
        the actual inference path is ``predict(x, task_id)``.
        """
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except Exception:
            return float('nan')
        was_training = model.training
        model.eval()
        x = sample_input[:1].to(self.device)
        try:
            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                (forward_fn or model)(x)
            total = counter.get_total_flops()
            return float(total) / 1e9
        except Exception:
            return float('nan')
        finally:
            if was_training:
                model.train()

    def measure_inference(self, predict_fn, loader, max_batches: int = 20) -> float:
        """Milliseconds per sample at inference, averaged over up to `max_batches`.

        Timed with CUDA synchronisation around each batch so the number reflects
        completed work rather than queued kernels.
        """
        import torch as _t
        n = 0
        t0 = time.time()
        with _t.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                x = batch[0].to(self.device)
                predict_fn(x)
                if self.device.type == 'cuda':
                    _t.cuda.synchronize(self.device)
                n += x.shape[0]
        if n == 0:
            return float('nan')
        return (time.time() - t0) * 1000.0 / n

    def summary(self) -> Dict[str, float]:
        peak = float('nan')
        if self.device.type == 'cuda':
            peak = max(self.prior_peak_mb,
                       torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        util = (sum(self._util_samples) / len(self._util_samples)
                if self._util_samples else float('nan'))
        return {
            'TRAIN_TIME_MIN': self.train_seconds / 60.0,
            'VALUATION_MIN': self.valuation_seconds / 60.0,
            'GPU_UTIL_MEAN': util,
            'GPU_MEM_PEAK_MB': peak,
        }
