import sqlite3
from pathlib import Path
c=sqlite3.connect('/Users/suapril/Code/mozilla-humanscale/workbench/receipts/scenario2/workspace/ledger.sqlite')
c.executescript('UPDATE invoices SET total_cents = unit_cents * quantity')
c.commit()
c.close()
