"""Read GGUF metadata and tensor dimensions without loading the weights."""
import math
from pathlib import Path
import struct


def inspect_model(path):
    with Path(path).open('rb') as f:
        def read(fmt):
            n=struct.calcsize('<'+fmt);b=f.read(n)
            if len(b)!=n:raise ValueError('Incomplete GGUF header')
            return struct.unpack('<'+fmt,b)[0]
        def string():
            return f.read(read('Q')).decode('utf8')
        def item(kind,keep=False):
            formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
            if kind==8:return string()
            if kind==9:
                subtype,count=read('I'),read('Q')
                if subtype in formats:
                    f.seek(struct.calcsize('<'+formats[subtype])*count,1)
                else:
                    for _ in range(count):item(subtype)
                return {'items':count}
            if kind not in formats:raise ValueError('Unknown GGUF metadata type')
            return read(formats[kind])
        if f.read(4)!=b'GGUF':raise ValueError('Expected GGUF magic')
        version=read('I')
        if version not in (2,3):raise ValueError('Unsupported GGUF version')
        count,nkv=read('Q'),read('Q');meta={}
        for _ in range(nkv):
            key=string();value=item(read('I'))
            if key.startswith('general.') or key.endswith(('.block_count','.context_length')):meta[key]=value
        params=0
        for _ in range(count):
            name=string();ndim=read('I');dims=[read('Q') for _ in range(ndim)]
            read('I');read('Q');params+=math.prod(dims)
        return {'gguf_version':version,'tensor_count':count,'tensor_parameter_count':params,'metadata':meta}
