"""Portable process identity, memory and safety evidence."""

import ctypes
import os
import platform
import re
import shutil
import subprocess
import time


def _run(args, timeout=5):
    try:
        done = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
        return done.stdout.strip()
    except Exception:
        return ""


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    state = _run(["ps", "-p", str(pid), "-o", "stat="])
    return not state.startswith("Z") if state else True


def process_identity(pid):
    if not pid_alive(pid):
        return None
    pid = int(pid)
    executable = None
    if platform.system() == "Linux":
        try:
            executable = os.path.realpath("/proc/{}/exe".format(pid))
        except OSError:
            executable = None
    elif platform.system() == "Darwin":
        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            buffer = ctypes.create_string_buffer(4096)
            size = libproc.proc_pidpath(pid, buffer, len(buffer))
            if size > 0:
                executable = os.path.realpath(
                    buffer.value.decode("utf-8", errors="replace"))
        except Exception:
            executable = None
    if not executable:
        command = _run(["ps", "-p", str(pid), "-o", "comm="])
        if command:
            executable = os.path.realpath(command)
    ppid = _run(["ps", "-p", str(pid), "-o", "ppid="])
    started = _run(["ps", "-p", str(pid), "-o", "lstart="])
    return {
        "pid": pid,
        "executablePath": executable,
        "startTime": " ".join(started.split()) or None,
        "parentPid": int(ppid) if ppid.strip().isdigit() else None,
    }


def same_process(identity):
    if not isinstance(identity, dict):
        return False
    current = process_identity(identity.get("pid"))
    if not current:
        return False
    return all(current.get(field) == identity.get(field)
               for field in ("pid", "executablePath", "startTime"))


def stable_process_identity(pid, timeout=1.0):
    """Wait through launcher exec so the recorded executable is final."""
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        current = process_identity(pid)
        if not current:
            return None
        signature = (current.get("executablePath"), current.get("startTime"),
                     current.get("parentPid"))
        if previous == signature:
            return current
        previous = signature
        time.sleep(0.03)
    return process_identity(pid)


def resolve_executable(command):
    if not command:
        return None
    found = shutil.which(command[0]) or command[0]
    return os.path.realpath(os.path.expanduser(found))


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _mac_memory(pid=None):
    raw = _run(["vm_stat"])
    page_match = re.search(r"page size of (\d+) bytes", raw)
    page = int(page_match.group(1)) if page_match else 4096
    pages = {}
    for name, value in re.findall(r"^([^:]+):\s+(\d+)\.", raw, re.M):
        pages[name.strip()] = int(value)
    total_raw = _run(["sysctl", "-n", "hw.memsize"])
    total = int(total_raw) if total_raw.isdigit() else None
    reclaim_names = ("Pages free", "Pages inactive", "Pages speculative",
                     "Pages purgeable")
    reclaimable = sum(pages.get(name, 0) for name in reclaim_names) * page
    pressure = _run(["memory_pressure", "-Q"])
    match = re.search(r"free percentage:\s*([\d.]+)%", pressure, re.I)
    free_percent = _number(match.group(1)) if match else (
        reclaimable * 100.0 / total if total else None)
    swap = _run(["sysctl", "-n", "vm.swapusage"])
    used_match = re.search(r"used\s*=\s*([\d.]+)([MG])", swap)
    swap_used = None
    if used_match:
        scale = 1024 ** (3 if used_match.group(2) == "G" else 2)
        swap_used = int(float(used_match.group(1)) * scale)
    swapout = pages.get("Pageouts")
    rss = None
    if pid:
        rss_raw = _run(["ps", "-p", str(pid), "-o", "rss="])
        rss = int(rss_raw) * 1024 if rss_raw.strip().isdigit() else None
    therm = _run(["pmset", "-g", "therm"])
    speed = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    warning = re.search(r"thermal warning level\s*=?\s*(\d+)", therm,
                        re.I)
    thermal_raised = bool((speed and int(speed.group(1)) < 100) or
                          (warning and int(warning.group(1)) > 0))
    return {
        "totalBytes": total,
        "freePercent": free_percent,
        "reclaimableBytes": reclaimable,
        "wiredBytes": pages.get("Pages wired down", 0) * page,
        "compressedBytes": pages.get("Pages occupied by compressor", 0)
        * page,
        "swapUsedBytes": swap_used,
        "swapoutBytes": swapout * page if swapout is not None else None,
        "rssBytes": rss,
        "thermalRaised": thermal_raised,
    }


def _linux_memory(pid=None):
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                match = re.match(r"([^:]+):\s+(\d+)\s+kB", line)
                if match:
                    values[match.group(1)] = int(match.group(2)) * 1024
    except OSError:
        pass
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swapout = None
    try:
        with open("/proc/vmstat", encoding="utf-8") as handle:
            match = re.search(r"^pswpout\s+(\d+)$", handle.read(), re.M)
            if match:
                swapout = int(match.group(1)) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        pass
    rss = None
    if pid:
        try:
            with open("/proc/{}/statm".format(pid), encoding="utf-8") as h:
                rss = int(h.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    return {
        "totalBytes": total,
        "freePercent": available * 100.0 / total
        if available is not None and total else None,
        "reclaimableBytes": available,
        "wiredBytes": None,
        "compressedBytes": None,
        "swapUsedBytes": (values.get("SwapTotal", 0) -
                          values.get("SwapFree", 0)),
        "swapoutBytes": swapout,
        "rssBytes": rss,
        "thermalRaised": None,
    }


def memory_snapshot(pid=None):
    system = platform.system()
    if system == "Darwin":
        values = _mac_memory(pid)
    elif system == "Linux":
        values = _linux_memory(pid)
    else:
        values = {name: None for name in (
            "totalBytes", "freePercent", "reclaimableBytes", "wiredBytes",
            "compressedBytes", "swapUsedBytes", "swapoutBytes", "rssBytes",
            "thermalRaised")}
    values.update({"capturedAt": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": system})
    return values


def memory_delta(before, after):
    first = before.get("swapoutBytes")
    last = after.get("swapoutBytes")
    return max(0, last - first) if first is not None and last is not None \
        else None


def evaluate_safety(before, after, limits):
    limits = limits or {}
    latest = after or before
    reasons = []
    value = latest.get("freePercent")
    floor = limits.get("minFreePercent")
    if floor is not None and value is not None and value < float(floor):
        reasons.append({"code": "free_percent_below_limit",
                        "actual": value, "limit": float(floor)})
    value = latest.get("reclaimableBytes")
    floor = limits.get("minReclaimableBytes")
    if floor is not None and value is not None and value < int(floor):
        reasons.append({"code": "reclaimable_below_limit",
                        "actual": value, "limit": int(floor)})
    delta = memory_delta(before, after)
    ceiling = limits.get("maxSwapoutDeltaBytes")
    if ceiling is not None and delta is not None and delta > int(ceiling):
        reasons.append({"code": "swapout_delta_above_limit",
                        "actual": delta, "limit": int(ceiling)})
    value = latest.get("rssBytes")
    ceiling = limits.get("maxRssBytes")
    if ceiling is not None and value is not None and value > int(ceiling):
        reasons.append({"code": "rss_above_limit", "actual": value,
                        "limit": int(ceiling)})
    return reasons
