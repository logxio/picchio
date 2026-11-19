import sqlite3
from pathlib import Path
c=sqlite3.connect('/Users/suapril/Code/mozilla-humanscale/workbench/receipts/scenario4/workspace/ledger.sqlite')
c.executescript('SELECT 1')
c.commit()
c.close()
