"""Read-only view of persisted method-level dispatch opportunities and waits."""
import sqlite3
import time

def snapshot(database):
    now=time.time()
    with sqlite3.connect(database,timeout=30) as db:
        db.row_factory=sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rr_requests'").fetchone():return None
        pending=[dict(r) for r in db.execute("SELECT method,case_id,stage,queued_at FROM rr_requests WHERE status='waiting' ORDER BY queued_at")]
        for row in pending:row['wait_seconds']=now-row['queued_at']
        methods=[dict(r) for r in db.execute("""SELECT method,COUNT(*) grants,
            AVG(granted_at-queued_at) mean_wait_seconds,MAX(granted_at-queued_at) max_wait_seconds
            FROM rr_requests WHERE status='granted' GROUP BY method""")]
        recent=[dict(r) for r in db.execute("SELECT method,case_id,stage,queued_at,granted_at FROM rr_requests WHERE status='granted' ORDER BY granted_at DESC LIMIT 16")][::-1]
        last=db.execute('SELECT last_method FROM rr_state WHERE singleton=1').fetchone()[0]
    return dict(updated=now,last_method=last,pending=pending,methods=methods,recent_grants=recent,
                definition='Round robin among methods with a pending request. Cache hits need no slot. Retries rejoin the queue. A granted slot can still be rejected by the provider without charge.')
