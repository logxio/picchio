"""A Seatbelt policy inherited across the launched process tree."""
import json
import sys
from pathlib import Path


def profile(workspace, model, package, port):
    roots = ['/System', '/usr/lib', '/usr/share', '/usr/bin', '/bin', '/sbin',
             '/Library/Apple', '/Library/Preferences/Logging', '/private/var/db',
             '/opt/homebrew/bin', '/opt/homebrew/Cellar', '/opt/homebrew/opt', '/opt/homebrew/lib',
             str(Path(package).resolve()), str(Path(workspace).resolve()), str(Path(sys.prefix).resolve())]
    lines = ['(version 1)', '(deny default)', '(allow process-fork)',
             '(allow sysctl-read)', '(allow mach-lookup)',
             '(allow iokit-open)', '(allow file-read-metadata)', '(allow file-read* (literal "/"))',
             '(allow signal (target same-sandbox))']
    for root in roots:
        lines.append('(allow file-read* (subpath %s))' % json.dumps(root))
    lines.append('(allow file-read* (literal %s))' % json.dumps(str(Path(model).resolve())))
    lines.append('(allow file-read* (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))')
    lines.append('(allow file-write* (literal "/dev/null") (subpath %s))' % json.dumps(str(Path(workspace).resolve())))
    lines.append('(allow process-exec (subpath "/opt/homebrew/Cellar") (subpath "/opt/homebrew/opt") (literal "/usr/bin/env"))')
    lines.append('(allow network-bind (local ip "localhost:%d"))' % port)
    lines.append('(allow network-inbound (local ip "localhost:%d"))' % port)
    lines.append('(allow network-outbound (remote ip "localhost:%d"))' % port)
    return '\n'.join(lines) + '\n'
