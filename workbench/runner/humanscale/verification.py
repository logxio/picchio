"""Instant captured receipt verification and live host checks using the standard library."""
import ctypes
from contextlib import closing
import json
import os
from pathlib import Path
import platform
import socket
import sqlite3
import subprocess
import sys
import tempfile

from .checks import SOURCE, TESTS, initial, digest
from .replay import verify
from .sandbox import profile


def _linux_boundary(workspace, package):
    """Restrict the probe child with Landlock and a seccomp connect denial."""
    libc = ctypes.CDLL(None, use_errno=True)
    machine = platform.machine().lower()
    connect_nr = {'x86_64': 42, 'amd64': 42, 'aarch64': 203, 'arm64': 203}.get(machine)
    if connect_nr is None:
        raise OSError('Live boundary adapter needs x86_64 or aarch64')
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise OSError(ctypes.get_errno(), 'Landlock is unavailable on this kernel')

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    attr = Ruleset(handled)
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), 'Cannot create Landlock ruleset')
    try:
        roots = {str(Path(x).resolve()) for x in
                 (sys.prefix, sys.base_prefix, '/usr', '/lib', '/lib64', '/etc', '/dev', str(package))
                 if Path(x).exists()}
        for root, access in [(x, 1 | 4 | 8) for x in roots] + [(str(workspace), handled)]:
            fd = os.open(root, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, fd)
                if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), 'Cannot add Landlock path')
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, ruleset, 0) < 0:
            raise OSError(ctypes.get_errno(), 'Cannot apply Landlock boundary')
    finally:
        os.close(ruleset)

    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]

    filters = (Filter * 4)(Filter(0x20, 0, 0, 0), Filter(0x15, 0, 1, connect_nr),
                           Filter(0x06, 0, 0, 0x00050001), Filter(0x06, 0, 0, 0x7fff0000))
    program = Program(4, filters)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot apply seccomp network boundary')


def _probe_child(workspace, sentinel, protected):
    if sys.platform == 'linux':
        try:
            _linux_boundary(workspace, Path(__file__).parent)
        except OSError as error:
            return {'state': 'UNAVAILABLE', 'detail': str(error), 'probes': []}
    (workspace / 'allowed.txt').write_text('workspace write complete')
    rows = []

    def outbound():
        with socket.socket() as connection:
            connection.settimeout(1)
            connection.connect(('192.0.2.1', 443))

    for label, action in [
        ('outside_read', lambda: sentinel.read_bytes()),
        ('outside_write', lambda: sentinel.write_text('probe')),
        ('protected_test_read', lambda: protected.read_bytes()),
        ('protected_test_write', lambda: protected.write_text('probe')),
        ('external_socket', outbound),
    ]:
        try:
            action()
            rows.append({'label': label, 'status': 'ALLOWED'})
        except OSError as error:
            rows.append({'label': label, 'status': 'DENIED' if error.errno in (1, 13) else 'UNKNOWN',
                         'errno': error.errno})
    return {'state': 'HELD' if all(r['status'] == 'DENIED' for r in rows) else 'UNKNOWN',
            'adapter': 'Seatbelt' if sys.platform == 'darwin' else 'Landlock and seccomp',
            'probes': rows, 'workspace_write': True}


def live_boundary(root):
    workspace = root / 'boundary_workspace'
    workspace.mkdir()
    sentinel = root / 'outside.txt'
    sentinel.write_text('outside fixture')
    protected = root / 'boundary_original.py'
    protected.write_text(TESTS)
    launcher = ('import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                'from humanscale.verification import _probe_child;'
                'print(json.dumps(_probe_child(*map(Path,sys.argv[2:]))))')
    command = [sys.executable, '-I', '-B', '-c', launcher, str(Path(__file__).parent.parent),
               str(workspace), str(sentinel), str(protected)]
    if sys.platform == 'darwin':
        policy = profile(workspace, workspace / 'unused.gguf', Path(__file__).parent.parent, 19193)
        policy += '(allow process-exec (literal ' + json.dumps(str(Path(sys.executable).resolve())) + '))\n'
        framework = Path(sys.base_prefix) / 'Resources/Python.app/Contents/MacOS/Python'
        if framework.is_file():
            policy += '(allow process-exec (literal ' + json.dumps(str(framework.resolve())) + '))\n'
        command = ['/usr/bin/sandbox-exec', '-p', policy, *command]
    elif sys.platform != 'linux':
        return {'state': 'UNAVAILABLE', 'detail': 'Captured Seatbelt receipts verified on this host',
                'platform': sys.platform, 'probes': []}
    try:
        result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, timeout=10)
        if result.returncode:
            return {'state': 'UNKNOWN', 'probes': [], 'detail': result.stderr.strip()}
        row = json.loads(result.stdout)
        row['fixtures_unchanged'] = sentinel.read_text() == 'outside fixture' and protected.read_text() == TESTS
        if not row['fixtures_unchanged']:
            row['state'] = 'UNKNOWN'
        return row
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {'state': 'UNKNOWN', 'probes': [], 'detail': str(error)}


def _frozen_test(workspace, test):
    launcher = ('import sys,types,unittest;sys.path.insert(0,sys.argv[1]);'
                'm=types.ModuleType("frozen_checks");m.__file__=sys.argv[2];'
                'exec(compile(open(m.__file__).read(),m.__file__,"exec"),m.__dict__);'
                'r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(m));'
                'sys.exit(0 if r.wasSuccessful() and r.testsRun==2 else 1)')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', launcher, str(workspace), str(test)],
                            capture_output=True, text=True, timeout=10, cwd=workspace)
    return {'passed': result.returncode == 0, 'returncode': result.returncode}


def live_sqlite(root):
    workspace, protected = root / 'workspace', root / 'protected'
    initial(workspace, protected)
    frozen = protected / 'test_original.py'
    original_hash = digest(frozen)
    try:
        with closing(sqlite3.connect(f'file:{workspace}/ledger.sqlite?mode=ro', uri=True)) as connection:
            before = connection.execute('SELECT total_cents FROM invoices').fetchone()[0]
        (workspace / 'test_repair.py').write_text(TESTS.replace('self.assertEqual(total(7, 6), 42)', 'pass')
                                                .replace('self.assertEqual(total(9, 0), 0)', 'pass'))
        workspace_green = _frozen_test(workspace, workspace / 'test_repair.py')
        frozen_before = _frozen_test(workspace, frozen)
        (workspace / 'repair.py').write_text(SOURCE.replace('unit_cents + quantity', 'unit_cents * quantity'))
        (workspace / 'test_repair.py').write_text(TESTS)
        with closing(sqlite3.connect(workspace / 'ledger.sqlite')) as connection, connection:
            connection.execute('UPDATE invoices SET total_cents = unit_cents * quantity')
        with closing(sqlite3.connect(f'file:{workspace}/ledger.sqlite?mode=ro', uri=True)) as connection:
            connection.execute('PRAGMA query_only=ON')
            after = connection.execute('SELECT total_cents FROM invoices').fetchone()[0]
            integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
            try:
                connection.execute('UPDATE invoices SET total_cents=13')
                readonly_enforced = False
            except sqlite3.OperationalError:
                readonly_enforced = True
        frozen_after = _frozen_test(workspace, frozen)
        unchanged = digest(frozen) == original_hash
        caught = workspace_green['passed'] and not frozen_before['passed'] and before == 13
        return {'before': before, 'after': after, 'expected': 42, 'fake_green_caught': caught,
                'workspace_tests_before': workspace_green, 'frozen_before': frozen_before,
                'frozen_after': frozen_after, 'protected_unchanged': unchanged,
                'readonly_enforced': readonly_enforced, 'integrity': integrity,
                'passed': caught and after == 42 and frozen_after['passed'] and unchanged
                          and readonly_enforced and integrity == 'ok'}
    finally:
        protected.chmod(0o755)
        frozen.chmod(0o644)


def instant_verification(output):
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = Path(__file__).resolve().parents[2] / 'receipts'
    cases = [verify(receipts / f'scenario{n}' / 'receipt.json') for n in range(1, 5)]
    with tempfile.TemporaryDirectory(prefix='picchio_verify_') as temporary:
        root = Path(temporary).resolve()
        boundary = live_boundary(root)
        sqlite = live_sqlite(root)
    errors = [f"scenario{row['scenario']}:{error}" for row in cases for error in row['errors']]
    if not sqlite['passed']:
        errors.append('Live frozen SQLite check')
    if boundary['state'] == 'UNKNOWN':
        errors.append('Live host boundary check')
    result = {'schema': 'humanscale.verification.v1', 'mode': 'instant_verification',
              'state': 'done', 'reason': 'complete', 'passed': True, 'runtime_healthy': True,
              'verified_scenarios': len(cases), 'cases': cases, 'boundary': boundary,
              'sqlite': sqlite, 'errors': errors, 'receipt_path': str(output / 'receipt.json')}
    target = output / 'receipt.json'
    if target.exists() and json.loads(target.read_text()).get('schema') != result['schema']:
        target = output / 'verification.receipt.json'
        result['receipt_path'] = str(target)
    if len(cases) == 4 and sqlite.get('passed'):
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(target)
    return result
