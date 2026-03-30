"""GPU monitoring via pynvml."""

import logging

logger = logging.getLogger("agent.gpu")


def get_gpu_info() -> dict | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        return {
            "name": name,
            "vram_total_mb": mem.total // (1024 * 1024),
            "vram_used_mb": mem.used // (1024 * 1024),
            "vram_free_mb": mem.free // (1024 * 1024),
            "temperature_c": temp,
            "utilization_percent": util.gpu,
        }
    except Exception as e:
        logger.warning(f"GPU info unavailable: {e}")
        return None
