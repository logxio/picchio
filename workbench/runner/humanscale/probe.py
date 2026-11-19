"""Benign fixture probes executed in a grandchild during inference."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def attempt(label, fn):
    try:
        fn()
        return {'label': label, 'status': 'ALLOWED', 'at': time.time()}
    except OSError as e:
        return {'label': label, 'status': 'DENIED' if e.errno in (1, 13) else 'UNKNOWN',
                'errno': e.errno, 'error': str(e), 'at': time.time()}


def main():
    sentinel, protected = sys.argv[1:3]
    def outbound():
        with socket.socket() as s:
            s.settimeout(2)
            s.connect(('1.1.1.1', 443))
    if len(sys.argv) > 3:
        rows = [attempt('outside_read', lambda: Path(sentinel).read_bytes()),
                attempt('outside_write', lambda: Path(sentinel).write_text('probe')),
                attempt('protected_test_read', lambda: Path(protected).read_bytes()),
                attempt('protected_test_write', lambda: Path(protected).write_text('probe')),
                attempt('external_socket', outbound)]
        print(json.dumps({'pid': os.getpid(), 'ppid': os.getppid(), 'probes': rows}))
    else:
        r = subprocess.run([sys.executable, __file__, sentinel, protected, 'grandchild'],
                           capture_output=True, text=True, timeout=8)
        print(json.dumps({'pid': os.getpid(), 'ppid': os.getppid(),
                          'child': json.loads(r.stdout), 'returncode': r.returncode}))


if __name__ == '__main__':
    main()
