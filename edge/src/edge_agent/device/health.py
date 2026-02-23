"""Device health monitoring — CPU/GPU temp, memory, inference latency."""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from edge_agent.config import EdgeConfig

logger = structlog.get_logger(__name__)


class HealthMonitor:
    """Monitors Jetson Orin hardware health metrics."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

    def snapshot(self) -> dict:
        """Capture current health metrics.

        Returns:
            Dict with cpu_temp, gpu_temp, mem_used_mb keys.
        """
        return {
            "cpu_temp": self._read_cpu_temp(),
            "gpu_temp": self._read_gpu_temp(),
            "mem_used_mb": self._read_memory_used(),
        }

    def _read_cpu_temp(self) -> float:
        """Read CPU temperature from sysfs (Jetson thermal zones)."""
        thermal_paths = [
            Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
            Path("/sys/class/thermal/thermal_zone0/temp"),
        ]
        for path in thermal_paths:
            try:
                if path.exists():
                    temp_raw = path.read_text().strip()
                    return float(temp_raw) / 1000.0  # millidegrees → degrees
            except (OSError, ValueError):
                continue
        return 0.0

    def _read_gpu_temp(self) -> float:
        """Read GPU temperature from sysfs."""
        # Jetson Orin GPU thermal zone is typically zone1 or zone2
        for zone_id in range(1, 10):
            path = Path(f"/sys/devices/virtual/thermal/thermal_zone{zone_id}/temp")
            type_path = Path(f"/sys/devices/virtual/thermal/thermal_zone{zone_id}/type")
            try:
                if type_path.exists():
                    zone_type = type_path.read_text().strip()
                    if "GPU" in zone_type.upper() or "gpu" in zone_type:
                        temp_raw = path.read_text().strip()
                        return float(temp_raw) / 1000.0
            except (OSError, ValueError):
                continue
        return 0.0

    def _read_memory_used(self) -> int:
        """Read used memory in MB from /proc/meminfo."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    mem_info[key] = int(val)

            total_kb = mem_info.get("MemTotal", 0)
            available_kb = mem_info.get("MemAvailable", 0)
            used_kb = total_kb - available_kb
            return used_kb // 1024  # KB → MB
        except (OSError, ValueError, KeyError):
            return 0
