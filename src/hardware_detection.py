"""
Hardware Detection and Diagnostics for System Q.
Monitors GPU, VRAM, System RAM, and CUDA execution environment.
"""

import os
import platform
import psutil
import torch
from typing import Dict, Any, Optional

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    try:
        import nvidia_ml_py as pynvml
        HAS_PYNVML = True
    except ImportError:
        HAS_PYNVML = False


class HardwareDetector:
    """Detects and reports system hardware characteristics for System Q execution."""

    def __init__(self):
        self.os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
        self.cpu_info = {
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        }
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0

    def get_gpu_info(self, device_idx: int = 0) -> Dict[str, Any]:
        """Fetch granular GPU information including VRAM and compute capability."""
        if not self.cuda_available or device_idx >= self.device_count:
            return {
                "cuda_available": False,
                "device_name": "CPU",
                "total_vram_gb": 0.0,
                "free_vram_gb": 0.0,
                "allocated_vram_gb": 0.0,
                "compute_capability": None,
                "bf16_supported": False,
                "fp16_supported": False,
            }

        props = torch.cuda.get_device_properties(device_idx)
        total_vram = props.total_memory / (1024 ** 3)
        allocated_vram = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
        reserved_vram = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)

        # Query NVML if available for true system-level free memory
        free_vram = total_vram - reserved_vram
        driver_version = "Unknown"
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_vram = mem_info.free / (1024 ** 3)
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_version, bytes):
                    driver_version = driver_version.decode("utf-8")
            except Exception:
                pass

        # Compute capability and precision support
        major, minor = props.major, props.minor
        bf16_supported = torch.cuda.is_bf16_supported()
        fp16_supported = major >= 6  # Pascal and newer

        return {
            "cuda_available": True,
            "device_index": device_idx,
            "device_name": props.name,
            "driver_version": driver_version,
            "compute_capability": f"{major}.{minor}",
            "total_vram_gb": round(total_vram, 3),
            "free_vram_gb": round(free_vram, 3),
            "allocated_vram_gb": round(allocated_vram, 3),
            "reserved_vram_gb": round(reserved_vram, 3),
            "bf16_supported": bf16_supported,
            "fp16_supported": fp16_supported,
        }

    def get_full_profile(self) -> Dict[str, Any]:
        """Return comprehensive hardware profile and safe configuration recommendations."""
        gpu_info = self.get_gpu_info(0)
        free_vram = gpu_info.get("free_vram_gb", 0.0)

        # Determine safe execution mode for 3B parameter model (~6GB FP16)
        if gpu_info["cuda_available"]:
            if free_vram < 2.5:
                recommendation = "INT4_QUANTIZATION_REQUIRED"
                safe_batch_size = 1
            elif free_vram < 4.0:
                recommendation = "INT4_RECOMMENDED_INT8_MAY_REQUIRE_OFFLOAD"
                safe_batch_size = 2
            else:
                recommendation = "INT8_OR_FP16_POSSIBLE"
                safe_batch_size = 4
        else:
            recommendation = "CPU_ONLY"
            safe_batch_size = 1

        return {
            "os": self.os_info,
            "cpu": self.cpu_info,
            "gpu": gpu_info,
            "pytorch_version": torch.__version__,
            "recommendation": recommendation,
            "safe_batch_size": safe_batch_size,
        }

    def print_summary(self):
        """Format and print hardware diagnostics to console."""
        profile = self.get_full_profile()
        print("=" * 60)
        print("          SYSTEM Q — HARDWARE DETECTION PROFILE")
        print("=" * 60)
        print(f"OS:                {profile['os']}")
        print(f"PyTorch Version:   {profile['pytorch_version']}")
        print(f"RAM Total / Avail: {profile['cpu']['total_ram_gb']} GB / {profile['cpu']['available_ram_gb']} GB")
        gpu = profile['gpu']
        if gpu['cuda_available']:
            print(f"GPU:               {gpu['device_name']}")
            print(f"Compute Capability:{gpu['compute_capability']} (Driver: {gpu['driver_version']})")
            print(f"VRAM Total / Free: {gpu['total_vram_gb']} GB / {gpu['free_vram_gb']} GB")
            print(f"BF16 / FP16:       {gpu['bf16_supported']} / {gpu['fp16_supported']}")
        else:
            print("GPU:               CUDA not available")
        print(f"Recommendation:    {profile['recommendation']}")
        print(f"Safe Batch Size:   {profile['safe_batch_size']}")
        print("=" * 60)


if __name__ == "__main__":
    detector = HardwareDetector()
    detector.print_summary()
