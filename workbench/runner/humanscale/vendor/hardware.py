"""Optional hardware observation with bounded cleanup and explicit missing evidence."""
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time

FIELDS = ("layers_on_gpu", "rss", "prefill_tps", "decode_tps")


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def normalize(data, exit_code=None):
    rows = data.get("probes", data.get("samples", [])) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        rows = []
    if not rows and isinstance(data, dict) and any(k in data for k in FIELDS):
        rows = [data]
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        samples.append({"layers_on_gpu": number(row.get("layers_on_gpu", row.get("gpu_layers"))),
                        "rss": number(row.get("rss", row.get("rssBytes"))),
                        "prefill_tps": number(row.get("prefill_tps", row.get("prefill"))),
                        "decode_tps": number(row.get("decode_tps", row.get("decode"))),
                        "state": str(row.get("state", "UNKNOWN")), "i": row.get("i")})
    observed = any(any(s[k] is not None for k in FIELDS) for s in samples)
    states = {s["state"].upper() for s in samples}
    verdict = str(data.get("verdict", "")).upper() if isinstance(data, dict) else ""
    status = "UNKNOWN"
    if observed:
        if exit_code == 4 or states & {"CPU", "CPU FALLBACK"} or "CPU FALLBACK" in verdict:
            status = "CPU FALLBACK"
        elif exit_code == 3 or states & {"PARTIAL", "PARTIAL OFFLOAD"}:
            status = "PARTIAL OFFLOAD"
        elif exit_code == 0 and (states & {"HEALTHY", "ENGAGED", "GPU"} or "ENGAGED THROUGHOUT" in verdict):
            status = "HEALTHY"
    latest = {k: next((s[k] for s in reversed(samples) if s[k] is not None), None) for k in FIELDS}
    return dict(latest, status=status, samples=samples, sample_count=len(samples), rss_unit="bytes")


class HardwareSampler:
    def __init__(self, target, directory, duration):
        self.target, self.directory, self.duration = target, Path(directory), duration
        self.proc = None
        self.handles = []
        self.started_at = None
        self.error = None

    def start(self):
        self.started_at = time.time()
        executable = shutil.which("picchio") if self.target else None
        if not executable:
            self.error = "not requested" if not self.target else "sampler unavailable"
            return self
        try:
            self.handles = [(self.directory / name).open("w") for name in ("hardware.json", "hardware.log")]
            self.proc = subprocess.Popen([executable, "guard", self.target, "--for", str(self.duration),
                                          "--residency", "--json"], stdout=self.handles[0], stderr=self.handles[1])
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        return self

    def stop(self):
        code = None
        try:
            if self.proc:
                if self.proc.poll() is None:
                    self.proc.send_signal(signal.SIGINT)
                try:
                    code = self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    code = self.proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = str(exc)
        finally:
            for handle in self.handles:
                handle.close()
        data = {}
        if self.proc:
            try:
                data = json.loads((self.directory / "hardware.json").read_text())
            except (OSError, ValueError) as exc:
                self.error = str(exc)
        result = normalize(data, code)
        result.update(target=self.target, started_at=self.started_at, ended_at=time.time(),
                      sampler_exit_code=code, note=self.error,
                      artifact="hardware.json" if self.proc else None)
        return result
