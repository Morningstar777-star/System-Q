"""
Energy and Power Telemetry Monitor for System Q.
Monitors GPU power consumption using NVML and computes energy per decision.
"""

import time
import threading
from typing import List, Dict, Any, Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    try:
        import nvidia_ml_py as pynvml
        HAS_NVML = True
    except ImportError:
        HAS_NVML = False


class GPUEnergyMonitor:
    """Samples GPU power draw via NVML to calculate energy consumption (Joules)."""

    def __init__(self, device_idx: int = 0, sample_interval_ms: int = 25):
        self.device_idx = device_idx
        self.sample_interval = sample_interval_ms / 1000.0
        self.nvml_active = False
        self.handle = None
        self._samples: List[float] = []
        self._timestamps: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.idle_power_w: Optional[float] = None

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                self.nvml_active = True
                # Query idle baseline
                self.idle_power_w = self._get_current_power_w()
            except Exception:
                self.nvml_active = False

    def _get_current_power_w(self) -> Optional[float]:
        """Fetch instantaneous power draw in Watts."""
        if not self.nvml_active or self.handle is None:
            return None
        try:
            # Power in milliwatts
            mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return mw / 1000.0
        except Exception:
            return None

    def _sampling_worker(self):
        """Worker thread to sample power at high frequency."""
        while not self._stop_event.is_set():
            p = self._get_current_power_w()
            if p is not None:
                self._samples.append(p)
                self._timestamps.append(time.perf_counter())
            time.sleep(self.sample_interval)

    def start(self):
        """Start power sampling."""
        self._samples.clear()
        self._timestamps.clear()
        self._stop_event.clear()
        if self.nvml_active:
            self._thread = threading.Thread(target=self._sampling_worker, daemon=True)
            self._thread.start()

    def stop(self, num_decisions: int = 1) -> Dict[str, Any]:
        """Stop sampling and compute energy metrics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if not self.nvml_active or not self._samples:
            return {
                "measured": False,
                "power_avg_w": 0.0,
                "power_peak_w": 0.0,
                "energy_joules": 0.0,
                "energy_mj_per_decision": 0.0,
                "idle_power_w": 0.0,
                "note": "Hardware power telemetry unavailable or insufficient samples"
            }

        avg_power = sum(self._samples) / len(self._samples)
        peak_power = max(self._samples)

        # Trapezoidal or duration-based integration
        if len(self._timestamps) >= 2:
            duration = self._timestamps[-1] - self._timestamps[0]
        else:
            duration = self.sample_interval

        total_energy_j = avg_power * duration
        net_power = max(0.0, avg_power - (self.idle_power_w or 0.0))
        net_energy_j = net_power * duration
        energy_per_dec_mj = (total_energy_j / max(1, num_decisions)) * 1000.0

        return {
            "measured": True,
            "duration_s": round(duration, 4),
            "sample_count": len(self._samples),
            "power_avg_w": round(avg_power, 2),
            "power_peak_w": round(peak_power, 2),
            "idle_power_w": round(self.idle_power_w or 0.0, 2),
            "energy_joules": round(total_energy_j, 4),
            "net_energy_joules": round(net_energy_j, 4),
            "energy_mj_per_decision": round(energy_per_dec_mj, 2),
        }


class EnergyContext:
    """Context manager for easily measuring energy across an inference code block."""

    def __init__(self, label: str = "inference", num_decisions: int = 1, sample_interval_ms: int = 25):
        self.label = label
        self.num_decisions = num_decisions
        self.monitor = GPUEnergyMonitor(sample_interval_ms=sample_interval_ms)
        self.results: Dict[str, Any] = {}
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.results = self.monitor.stop(num_decisions=self.num_decisions)
        self.results["wall_time_ms"] = round((self.end_time - self.start_time) * 1000.0, 2)
        self.results["label"] = self.label


if __name__ == "__main__":
    print("Testing GPUEnergyMonitor...")
    with EnergyContext(label="test_idle_measurement", sample_interval_ms=20) as ctx:
        time.sleep(0.5)
    print("Energy Context Results:")
    for k, v in ctx.results.items():
        print(f"  {k}: {v}")
