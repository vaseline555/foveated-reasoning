"""Conservatively extend the shared dispatch gate after provider-pool HTTP429."""
import json
import os
from pathlib import Path
import sqlite3
import time
from campaign import RUN, save

pid=int((RUN/'background.pid').read_text())
while True:
    try:os.kill(pid,0)
    except ProcessLookupError:break
    state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {}
    if state.get('status') in ('completed','stopped','failed'):break
    with sqlite3.connect(RUN/'ledger.sqlite',timeout=30) as db:
        db.execute('BEGIN IMMEDIATE')
        total=db.execute('SELECT total_429 FROM api_health WHERE singleton=1').fetchone()[0]
        gap=60. if total>=3 else (45. if total else 30.)
        started=db.execute("SELECT MAX(started) FROM attempts WHERE status NOT IN ('reserved_final','cancelled_reservation')").fetchone()[0]
        if started:
            db.execute('UPDATE api_health SET next_dispatch_at=MAX(next_dispatch_at,?) WHERE singleton=1',(started+gap,))
    save(RUN/'pacing.json',dict(updated=time.time(),effective_min_gap_seconds=gap,http429=total,
         policy='30s initially; 45s after first shared-pool 429; 60s after third; existing Retry-After/exponential cooldown remains in force'))
    time.sleep(.25)
