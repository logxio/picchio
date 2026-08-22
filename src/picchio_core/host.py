"""Portable process identity, memory and safety evidence."""

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import time

WINDOWS = os.name == "nt"


def _run(args, timeout=5):
    try:
        done = subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout)
        return done.stdout.strip()
    except Exception:
        return ""


# ------------------------------------------------------------- windows
# kernel32 through ctypes, nothing outside the stdlib. The POSIX liveness
# idiom os.kill(pid, 0) must never run here: on Windows every signal
# number means TerminateProcess, so the probe would kill the engine it
# was asked about.

_STILL_ACTIVE = 259
_QUERY_LIMITED = 0x1000
_ACCESS_DENIED = 5


class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                ("total_phys", ctypes.c_uint64),
                ("avail_phys", ctypes.c_uint64),
                ("total_pagefile", ctypes.c_uint64),
                ("avail_pagefile", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("avail_virtual", ctypes.c_uint64),
                ("avail_extended", ctypes.c_uint64)]


class _PROCESSENTRY32(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("usage", ctypes.c_uint32),
                ("pid", ctypes.c_uint32),
                ("default_heap", ctypes.c_size_t),
                ("module", ctypes.c_uint32), ("threads", ctypes.c_uint32),
                ("parent", ctypes.c_uint32),
                ("priority", ctypes.c_long), ("flags", ctypes.c_uint32),
                ("exe", ctypes.c_char * 260)]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("page_faults", ctypes.c_uint32),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("quota_peak_paged", ctypes.c_size_t),
                ("quota_paged", ctypes.c_size_t),
                ("quota_peak_nonpaged", ctypes.c_size_t),
                ("quota_nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t)]


def _kernel32():
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.OpenProcess.restype = ctypes.c_void_p
    lib.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                ctypes.c_uint32]
    lib.CloseHandle.argtypes = [ctypes.c_void_p]
    lib.GetExitCodeProcess.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_uint32)]
    lib.QueryFullProcessImageNameW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_uint32)]
    lib.GetProcessTimes.argtypes = [ctypes.c_void_p] + [
        ctypes.POINTER(_FILETIME)] * 4
    lib.K32GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.c_uint32]
    return lib


def _win_parent(pid):
    """The pid that created this one, the field ps prints as ppid. The
    process table is the only place Windows publishes it."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32,
                                             ctypes.c_uint32]
    k32.Process32First.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(_PROCESSENTRY32)]
    k32.Process32Next.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(_PROCESSENTRY32)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = k32.CreateToolhelp32Snapshot(0x00000002, 0)  # SNAPPROCESS
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return None
    try:
        entry = _PROCESSENTRY32()
        entry.size = ctypes.sizeof(entry)
        found = k32.Process32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.pid == pid:
                return int(entry.parent)
            found = k32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snapshot)
    return None


def _win_process(pid):
    """{exe, start, rss} for a live process, None for a dead or absent
    one. A handle we are not allowed to open still proves the process
    exists, so that case returns a dict with nothing in it."""
    k32 = _kernel32()
    handle = k32.OpenProcess(_QUERY_LIMITED, 0, pid)
    if not handle:
        if ctypes.get_last_error() == _ACCESS_DENIED:
            return {"exe": None, "start": None, "rss": None}
        return None
    try:
        code = ctypes.c_uint32()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) \
                or code.value != _STILL_ACTIVE:
            return None
        size = ctypes.c_uint32(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        exe = buf.value if k32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size)) else None
        times = [_FILETIME() for _ in range(4)]
        start = None
        if k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            # creation time in 100 ns ticks: the job ps lstart does on
            # POSIX, a value that changes when the pid is reused
            start = str((times[0].hi << 32) | times[0].lo)
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        rss = int(counters.working_set) if k32.K32GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb) else None
        return {"exe": exe or None, "start": start, "rss": rss}
    finally:
        k32.CloseHandle(handle)


def windows_memory_status():
    """(total physical bytes, available physical bytes), or (None, None)."""
    status = _MEMORYSTATUSEX()
    status.length = ctypes.sizeof(status)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(
            ctypes.byref(status)):
        return None, None
    return int(status.total_phys), int(status.avail_phys)


def windows_machine():
    """(cpu brand, ram bytes, os label) from the registry and kernel32."""
    chip = ""
    try:
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            chip = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
    except (ImportError, OSError):
        chip = ""
    total, _ = windows_memory_status()
    build = getattr(sys.getwindowsversion(), "build", 0)
    label = "Windows 11" if build >= 22000 else \
        "Windows " + platform.release()
    return " ".join(chip.split()), total, label


def process_list():
    """[(pid, command line)] for every process the OS will show. POSIX
    asks ps; Windows asks the CIM process table through PowerShell, the
    built-in that still prints command lines now that wmic is gone from
    fresh installs."""
    if WINDOWS:
        text = _run(["powershell", "-NoProfile", "-NonInteractive",
                     "-Command",
                     "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
                     "Get-CimInstance Win32_Process | ForEach-Object {"
                     " '{0} {1}' -f $_.ProcessId, $_.CommandLine }"],
                    timeout=30)
    else:
        text = _run(["ps", "-axo", "pid=,command="])
    out = []
    for line in text.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[0].isdigit():
            out.append((int(fields[0]), fields[1]))
    return out


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if WINDOWS:
        return _win_process(pid) is not None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by another user
    except OSError:
        return False
    # Signal 0 also succeeds for a dead child waiting to be reaped. That
    # zombie counts as exited, or a guarded pid could wait an hour after
    # its work ended.
    state = _run(["ps", "-p", str(pid), "-o", "stat="])
    return not state.startswith("Z") if state else True


def process_identity(pid):
    if not pid_alive(pid):
        return None
    pid = int(pid)
    if WINDOWS:
        info = _win_process(pid) or {}
        return {"pid": pid, "executablePath": info.get("exe"),
                "startTime": info.get("start"),
                "parentPid": _win_parent(pid)}
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


def _windows_memory(pid=None):
    total, available = windows_memory_status()
    info = _win_process(int(pid)) if pid else None
    return {
        "totalBytes": total,
        "freePercent": available * 100.0 / total
        if available is not None and total else None,
        "reclaimableBytes": available,
        "wiredBytes": None,
        "compressedBytes": None,
        "swapUsedBytes": None,
        "swapoutBytes": None,
        "rssBytes": info.get("rss") if info else None,
        "thermalRaised": None,
    }


def memory_snapshot(pid=None):
    system = platform.system()
    if system == "Darwin":
        values = _mac_memory(pid)
    elif system == "Linux":
        values = _linux_memory(pid)
    elif system == "Windows":
        values = _windows_memory(pid)
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
