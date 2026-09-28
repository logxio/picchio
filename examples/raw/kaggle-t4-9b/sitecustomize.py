import os
import sys
_original_urandom = os.urandom
def _controlled_urandom(n):
    caller = sys._getframe(1)
    if n == 4 and caller.f_code.co_filename.endswith('picchio.py') and caller.f_code.co_name == 'main':
        return bytes.fromhex('11223344')
    return _original_urandom(n)
os.urandom = _controlled_urandom
