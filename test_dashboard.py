"""
Test script for dashboard server - verifies all endpoints work.
"""
import sys
import time
import urllib.request
import json
import threading

sys.path.insert(0, r'c:\Users\rodri\Desktop\bot brawl')

from pylaai_real.dashboard_server import DashboardServer

# Start server
server = DashboardServer(port=8766)
server.start(daemon=True)
time.sleep(1)

BASE = "http://localhost:8766"
results = []

def test_endpoint(method, path, data=None):
    try:
        url = BASE + path
        if method == "GET":
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode('utf-8')
                return json.loads(body)
        else:
            req = urllib.request.Request(url, data=json.dumps(data or {}).encode('utf-8'),
                                         headers={"Content-Type": "application/json"},
                                         method=method)
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode('utf-8')
                return json.loads(body)
    except Exception as e:
        return {"_error": str(e)}

# Test endpoints
print("=" * 50)
print("DASHBOARD SERVER ENDPOINT TESTS")
print("=" * 50)

# GET endpoints
for path in ["/", "/dashboard", "/api/live", "/api/history", "/api/rewards",
             "/api/replays", "/api/abtest", "/api/recovery", "/api/brawlers",
             "/api/match-analysis", "/api/ai-pick", "/api/trophy-history",
             "/api/weekly-progress", "/api/bot/status", "/api/system/status",
             "/api/logs", "/api/notifications/config", "/api/notifications/history",
             "/api/config", "/api/antiban/status", "/api/export/stats"]:
    result = test_endpoint("GET", path)
    has_error = "_error" in result or "error" in result
    status = "FAIL" if has_error else "OK"
    print(f"  [{status}] GET {path}")
    if has_error:
        print(f"       -> {result.get('_error') or result.get('error')}")
    results.append(("GET", path, status))

# POST endpoints (without wrapper, some may fail)
for path in ["/api/abtest/start", "/api/abtest/stop", "/api/replay/start",
             "/api/replay/stop", "/api/bot/start", "/api/bot/stop",
             "/api/bot/pause", "/api/bot/resume", "/api/system/toggle",
             "/api/notifications/test", "/api/notifications/config"]:
    result = test_endpoint("POST", path)
    has_error = "_error" in result
    status = "FAIL" if has_error else "OK"
    print(f"  [{status}] POST {path}")
    if has_error:
        print(f"       -> {result.get('_error')}")
    results.append(("POST", path, status))

# Summary
ok = sum(1 for _, _, s in results if s == "OK")
fail = sum(1 for _, _, s in results if s == "FAIL")
print("=" * 50)
print(f"RESULTS: {ok} OK, {fail} FAIL out of {len(results)} endpoints")
print("=" * 50)

server.stop()
