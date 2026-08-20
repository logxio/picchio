"""The OS's own GPU meters: ioreg/IOReport, NVML, amdgpu sysfs.

One backend per platform, all answering the same three questions, so the
sampler above them never learns which silicon it is watching:

    .src            a short name for the source, recorded in the summary
    .sample()       one reading, or None
    .throttled()    whether the machine says it is thermally limited
    .device_name()  a display name for the machine fingerprint, or None

A sample is {"t": monotonic, "dev": 0-100, "mem": bytes or None,
"gpu_w": watts or None}; the macOS one carries its renderer and tiler
lanes too, which only the raw curve keeps. Every optional field degrades
on its own: a symbol that moved or a sysfs file that is not there drops
that field and never reaches the judgment.

Machines with more than one card report one collapsed reading, because
one number per tick is what the window math consumes. Utilization takes
the max (any card doing work means the GPU did work, and a three card
box splitting layers must not look idle because two of them finished
first); memory and power take the sum, since the weights and the draw
are spread over all of them.
"""

import ctypes
import glob
import os
import platform
import re
import shutil
import subprocess
import time


RE_TELE = {
    "dev": re.compile(r'"Device Utilization %"=(\d+)'),
    "ren": re.compile(r'"Renderer Utilization %"=(\d+)'),
    "til": re.compile(r'"Tiler Utilization %"=(\d+)'),
    "mem": re.compile(r'"In use system memory"=(\d+)'),
}


def _cmd_out(args, timeout=5):
    try:
        done = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
        return done.stdout
    except Exception:
        return ""


def _collapse(per_device):
    """Many cards, one reading. Empty in, None out."""
    per_device = [d for d in per_device if d is not None]
    if not per_device:
        return None
    def total(key):
        vals = [d[key] for d in per_device if d.get(key) is not None]
        return sum(vals) if vals else None
    return {"t": time.monotonic(),
            "dev": max(d["dev"] for d in per_device),
            "mem": total("mem"), "gpu_w": total("gpu_w")}


def _read_int(path):
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------ macOS

class _IOReport:
    """GPU power without sudo: IOReport is the private framework that
    powermetrics itself reads, and its energy counters answer any
    process. Private means it can move between macOS versions, so every
    call is guarded; when anything is missing or NULL, power quietly
    stays off the os line and nothing else changes."""

    SCALE = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9}

    def __init__(self):
        p = ctypes.c_void_p
        self.cf = ctypes.CDLL("/System/Library/Frameworks/"
                              "CoreFoundation.framework/CoreFoundation")
        self.io = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        for lib, name, res, args in (
            (self.cf, "CFStringCreateWithCString", p,
             [p, ctypes.c_char_p, ctypes.c_uint32]),
            (self.cf, "CFStringGetCString", ctypes.c_bool,
             [p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
            (self.cf, "CFDictionaryGetValue", p, [p, p]),
            (self.cf, "CFArrayGetCount", ctypes.c_long, [p]),
            (self.cf, "CFArrayGetValueAtIndex", p, [p, ctypes.c_long]),
            (self.cf, "CFRelease", None, [p]),
            (self.io, "IOReportCopyChannelsInGroup", p,
             [p, p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
            (self.io, "IOReportCreateSubscription", p,
             [p, p, ctypes.POINTER(p), ctypes.c_uint64, p]),
            (self.io, "IOReportCreateSamples", p, [p, p, p]),
            (self.io, "IOReportCreateSamplesDelta", p, [p, p, p]),
            (self.io, "IOReportChannelGetChannelName", p, [p]),
            (self.io, "IOReportChannelGetUnitLabel", p, [p]),
            (self.io, "IOReportSimpleGetIntegerValue", ctypes.c_int64,
             [p, ctypes.POINTER(ctypes.c_int32)]),
        ):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        chans = self.io.IOReportCopyChannelsInGroup(
            self._cfstr("Energy Model"), None, 0, 0, 0)
        if not chans:
            raise OSError("no Energy Model channels")
        subbed = ctypes.c_void_p()
        self._sub = self.io.IOReportCreateSubscription(
            None, chans, ctypes.byref(subbed), 0, None)
        if not self._sub:
            raise OSError("IOReport subscription failed")
        self._subbed = subbed
        self._key = self._cfstr("IOReportChannels")
        self._prev = self.io.IOReportCreateSamples(self._sub, subbed, None)
        self._t_prev = time.monotonic()

    def _cfstr(self, s):
        return self.cf.CFStringCreateWithCString(None, s.encode(),
                                                 0x08000100)

    def _pystr(self, ref):
        buf = ctypes.create_string_buffer(128)
        if ref and self.cf.CFStringGetCString(ref, buf, 128, 0x08000100):
            return buf.value.decode()
        return None

    def watts(self):
        """Average GPU watts since the previous call, or None."""
        cur = self.io.IOReportCreateSamples(self._sub, self._subbed, None)
        t = time.monotonic()
        if not cur or t <= self._t_prev:
            return None
        delta = self.io.IOReportCreateSamplesDelta(self._prev, cur, None)
        w = None
        arr = self.cf.CFDictionaryGetValue(delta, self._key) \
            if delta else None
        for i in range(self.cf.CFArrayGetCount(arr) if arr else 0):
            ch = self.cf.CFArrayGetValueAtIndex(arr, i)
            name = self._pystr(self.io.IOReportChannelGetChannelName(ch))
            if name == "GPU Energy":
                unit = (self._pystr(
                    self.io.IOReportChannelGetUnitLabel(ch)) or "").strip()
                scale = self.SCALE.get(unit)
                if scale:
                    j = self.io.IOReportSimpleGetIntegerValue(ch, None)
                    w = j * scale / (t - self._t_prev)
                break
        self.cf.CFRelease(self._prev)
        if delta:
            self.cf.CFRelease(delta)
        self._prev, self._t_prev = cur, t
        return w


class _IOAccel:
    """The macOS meter: ioreg for utilization and memory, IOReport for
    watts, pmset for thermal pressure. ioreg counts the whole GPU, not
    this process's share of it, which is why an idle baseline is a gate
    on judging at all rather than a number anyone subtracts."""

    src = "ioreg"

    def __init__(self):
        if not shutil.which("ioreg"):
            raise OSError("no ioreg")
        try:
            self.power = _IOReport()
        except Exception:
            self.power = None  # private API absent or moved: no watts

    def sample(self):
        out = _cmd_out(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
        m = re.search(r'"PerformanceStatistics" = \{(.*)\}', out)
        if not m:
            return None
        s = {"t": time.monotonic()}
        for key, rx in RE_TELE.items():
            mm = rx.search(m.group(1))
            s[key] = int(mm.group(1)) if mm else None
        if s["dev"] is None:
            return None
        if self.power:
            try:
                s["gpu_w"] = self.power.watts()
            except Exception:
                self.power = None
        return s

    def throttled(self):
        """True when macOS itself says the machine is under thermal
        pressure. Presentation only; it never votes on placement."""
        out = _cmd_out(["pmset", "-g", "therm"])
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        if m and int(m.group(1)) < 100:
            return True
        m = re.search(r"thermal warning level\s*=?\s*(\d+)", out, re.I)
        return bool(m and int(m.group(1)) > 0)

    def device_name(self):
        return None  # the chip string already names an Apple GPU


# ------------------------------------------------------------ NVIDIA

class _NVML:
    """The NVIDIA meter on Linux: libnvidia-ml.so.1 is the library
    nvidia-smi itself reads, resolvable wherever the driver is
    installed (verified on driver 550.54.14, all eight core symbols).
    utilization.gpu is the percent of the last internal sample period
    (between 1/6 s and 1 s depending on the product, per the NVML
    docs) during which any kernel ran, so 4 Hz polling repeats values;
    medians are judged, so repeats change nothing. Symbols are guarded
    like the IOReport private framework: anything missing quietly
    drops its field and never touches the judgment."""

    src = "nvml"

    # thermal slowdown bits of the clocks throttle reasons mask
    # (sw thermal 0x20, hw thermal 0x40); power caps are normal
    # operation and do not count as throttling here
    THERMAL = 0x20 | 0x40

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        count = ctypes.c_uint()
        if not (hasattr(self.lib, "nvmlDeviceGetCount_v2")
                and self.lib.nvmlDeviceGetCount_v2(
                    ctypes.byref(count)) == 0):
            count.value = 1
        self.hdls = []
        for i in range(max(1, count.value)):
            hdl = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(
                    i, ctypes.byref(hdl)) == 0:
                self.hdls.append(hdl)
        if not self.hdls:
            raise OSError("no nvml device")

    def _one(self, hdl):
        u = self._Util()
        if not (hasattr(self.lib, "nvmlDeviceGetUtilizationRates")
                and self.lib.nvmlDeviceGetUtilizationRates(
                    hdl, ctypes.byref(u)) == 0):
            return None
        s = {"dev": int(u.gpu), "mem": None, "gpu_w": None}
        m = self._Mem()
        if hasattr(self.lib, "nvmlDeviceGetMemoryInfo") \
                and self.lib.nvmlDeviceGetMemoryInfo(
                    hdl, ctypes.byref(m)) == 0:
            s["mem"] = int(m.used)
        p = ctypes.c_uint()
        if hasattr(self.lib, "nvmlDeviceGetPowerUsage") \
                and self.lib.nvmlDeviceGetPowerUsage(
                    hdl, ctypes.byref(p)) == 0:
            s["gpu_w"] = p.value / 1000.0
        return s

    def sample(self):
        return _collapse([self._one(h) for h in self.hdls])

    def _name(self, hdl):
        if not hasattr(self.lib, "nvmlDeviceGetName"):
            return None
        buf = ctypes.create_string_buffer(96)
        if self.lib.nvmlDeviceGetName(hdl, buf, 96) != 0:
            return None
        name = buf.value.decode(errors="replace").strip()
        # display form: the NVIDIA prefix drops so the gpu line and the
        # footer hold the 66 column budget; the raw name is in the log
        return re.sub(r"^NVIDIA\s+", "", name) or None

    def device_name(self):
        return _fleet_name([self._name(h) for h in self.hdls])

    def throttled(self):
        # newer drivers renamed the symbol; try both, judge the same bits
        for sym in ("nvmlDeviceGetCurrentClocksThrottleReasons",
                    "nvmlDeviceGetCurrentClocksEventReasons"):
            if hasattr(self.lib, sym):
                for hdl in self.hdls:
                    r = ctypes.c_ulonglong()
                    if getattr(self.lib, sym)(hdl, ctypes.byref(r)) == 0 \
                            and r.value & self.THERMAL:
                        return True
                return False
        return False


# ------------------------------------------------------------ AMD

class _AMDGPU:
    """The AMD meter on Linux, read straight out of the amdgpu driver's
    sysfs tree: no rocm-smi, no ROCm install, nothing to import. Paths
    and units are the kernel's own (docs.kernel.org/gpu/amdgpu):

        device/gpu_busy_percent   0-100, computed by the SMU firmware
                                  from aggregate IP core activity, the
                                  direct counterpart of NVML's
                                  utilization.gpu
        device/mem_info_vram_used bytes of VRAM in use
        hwmon/hwmon*/power1_average  average SoC power in microwatts

    Only cards that publish gpu_busy_percent count, so presence is the
    capability test and an ASIC whose firmware does not report it is
    simply not a meter. Connector directories (card0-DP-1 and friends)
    are excluded by name so a monitor cannot be counted as a GPU.

    Marketing names live in product_name, which the kernel documents as
    server cards only, so a consumer card reports no name here; the
    engine log already prints a better one and the gpu line uses that."""

    src = "amdgpu"
    ROOT = "/sys/class/drm"

    def __init__(self, root=None):
        self.root = root or self.ROOT
        self.cards = []
        try:
            entries = sorted(os.listdir(self.root))
        except OSError:
            entries = []
        for entry in entries:
            if not re.match(r"^card\d+$", entry):
                continue
            dev = os.path.join(self.root, entry, "device")
            if os.path.exists(os.path.join(dev, "gpu_busy_percent")):
                self.cards.append(dev)
        if not self.cards:
            raise OSError("no amdgpu card reports gpu_busy_percent")

    def _watts(self, dev):
        for leaf in ("power1_average", "power1_input"):
            for path in sorted(glob.glob(
                    os.path.join(dev, "hwmon", "hwmon*", leaf))):
                uw = _read_int(path)
                if uw is not None:
                    return uw / 1000000.0
        return None

    def _one(self, dev):
        busy = _read_int(os.path.join(dev, "gpu_busy_percent"))
        if busy is None:
            return None
        return {"dev": busy,
                "mem": _read_int(os.path.join(dev, "mem_info_vram_used")),
                "gpu_w": self._watts(dev)}

    def sample(self):
        return _collapse([self._one(d) for d in self.cards])

    def device_name(self):
        names = []
        for dev in self.cards:
            try:
                with open(os.path.join(dev, "product_name")) as handle:
                    names.append(handle.read().strip() or None)
            except OSError:
                names.append(None)
        return _fleet_name(names)

    def throttled(self):
        # amdgpu publishes throttle status only inside the binary
        # gpu_metrics blob, whose layout changes per ASIC revision.
        # Reading it wrong would be worse than not reading it, and
        # thermal pressure is presentation only, so this abstains.
        return False


def _fleet_name(names):
    """One display string for a set of cards: 'RTX 4090 x3' when they
    match, both names when they do not, None when nothing answered."""
    names = [n for n in names if n]
    if not names:
        return None
    uniq = sorted(set(names))
    if len(uniq) == 1:
        return uniq[0] if len(names) == 1 else \
            "{} x{}".format(uniq[0], len(names))
    return " + ".join(uniq)


# ------------------------------------------------------------ dispatch

def open_meter(disabled=False):
    """A live backend, or a dict naming why there is none. The os line
    prints that reason, so a run without OS evidence says so instead of
    quietly reading like a fully instrumented one."""
    if disabled:
        return {"off": "disabled"}
    sysname = platform.system()
    if sysname == "Linux":
        for backend in (_NVML, _AMDGPU):
            try:
                meter = backend()
            except Exception:
                continue
            if meter.sample() is not None:
                return meter
        return {"off": "no nvml/amdgpu"}
    if sysname != "Darwin":
        return {"off": "not macos"}
    try:
        meter = _IOAccel()
    except Exception:
        return {"off": "no ioreg"}
    if meter.sample() is None:
        return {"off": "no ioreg data"}
    return meter


def machine_gpu_name():
    """The GPU for the machine fingerprint, or None. Linux only: on
    macOS the chip string already names it."""
    if platform.system() != "Linux":
        return None
    for backend in (_NVML, _AMDGPU):
        try:
            name = backend().device_name()
        except Exception:
            continue
        if name:
            return name
    return None
