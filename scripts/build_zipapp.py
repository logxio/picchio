#!/usr/bin/env python3
"""Build a deterministic, executable Picchio zipapp with stdlib only."""

import argparse
import os
import stat
import tempfile
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = b"from picchio import entrypoint\nentrypoint()\n"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def sources():
    yield "picchio.py", os.path.join(ROOT, "picchio.py"), None
    package = os.path.join(ROOT, "src", "picchio_core")
    for name in sorted(os.listdir(package)):
        if name.endswith(".py"):
            yield "picchio_core/" + name, os.path.join(package, name), None
    yield "__main__.py", None, MAIN


def build(output):
    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".picchio-", suffix=".pyz",
                                dir=os.path.dirname(output))
    try:
        with os.fdopen(fd, "wb") as raw:
            raw.write(b"#!/usr/bin/env python3\n")
            with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as archive:
                for arcname, path, inline in sources():
                    data = inline
                    if data is None:
                        with open(path, "rb") as handle:
                            data = handle.read()
                    info = zipfile.ZipInfo(arcname, FIXED_TIME)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(info, data)
            raw.flush()
            os.fsync(raw.fileno())
        os.chmod(temp, 0o755)
        os.replace(temp, output)
        return output
    except Exception:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        ROOT, "public", "picchio.pyz"))
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
