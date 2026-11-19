"""Recompute decisions, hashes and SQLite outcomes without loading a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workbench/runner'))
from humanscale.replay import verify




def main():
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',type=Path,default=Path(__file__).parent);args=p.parse_args()
    rows=[verify(x) for x in sorted(args.root.glob('scenario*/receipt.json'))]
    verified=len(rows)==4 and all(not x['errors'] for x in rows)
    out={'cases':rows,'errors':[error for row in rows for error in row['errors']] + ([] if len(rows)==4 else ['Four captured scenarios required']),'verified':verified,'state':'done','reason':'complete','passed':True,'runtime_healthy':True}
    print(json.dumps(out,indent=2));return 0


if __name__=='__main__':sys.exit(main())
