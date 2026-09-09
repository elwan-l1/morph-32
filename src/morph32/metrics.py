"""Unified-memory and power measurements on macOS."""

import ctypes
import resource
import subprocess

import mlx.core as mx


def memory():
    result = dict(
        mlx_active_bytes=mx.get_active_memory(),
        mlx_cache_bytes=mx.get_cache_memory(),
        mlx_peak_bytes=mx.get_peak_memory(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    # Offsets verified against local macOS SDK mach/task_info.h, rev3 prefix.
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    task = ctypes.c_uint.in_dll(library, "mach_task_self_").value
    values = (ctypes.c_uint64 * 128)()
    count = ctypes.c_uint(256)
    status = library.task_info(task, 22, ctypes.byref(values), ctypes.byref(count))
    if status == 0 and count.value >= 44:
        result.update(
            process_phys_footprint_bytes=values[18],
            process_lifetime_peak_phys_footprint_bytes=values[21],
            mach_resident_bytes=values[2],
            mach_resident_peak_bytes=values[3],
        )
    else:
        result["mach_task_info_error"] = status
    return result


def power_state():
    result = {}
    for label, command in [
        ("power", ["pmset", "-g", "batt"]),
        ("thermal", ["pmset", "-g", "therm"]),
    ]:
        try:
            result[label] = subprocess.run(
                command, capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[label] = type(error).__name__
    return result
