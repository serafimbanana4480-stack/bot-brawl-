S03 E2E Manual Checklist

Purpose: Manual steps to verify emulator lifecycle and bot start endpoints on Windows (BlueStacks/ADB). These steps require a Windows emulator with ADB enabled and accessible from the host running the backend.

Prerequisites:
- Windows emulator (BlueStacks, Nox, or real device) with ADB enabled and connected.
- adb available on PATH or configured via the project get_adb_path.
- Python environment and backend dependencies installed.

Start the backend (example):
- From project root run:
  uvicorn soberana_omega.backend.api.app:app --reload --host 127.0.0.1 --port 8000

Basic route checklist (example curl commands):

1) List emulators (no detection yet)
- curl -sS http://127.0.0.1:8000/emulators | jq '.'

2) Connect emulator via emulator-add/connect flow (calls detect_all)
- curl -X POST -sS http://127.0.0.1:8000/emulators/connect -H "Content-Type: application/json" -d '{"emulator_id": "auto"}' | jq '.'
- Expect diagnostics object with detection outputs and chosen_adb_path field when applicable.

3) Run setup (must be called before starting in this slice)
- curl -X POST -sS http://127.0.0.1:8000/setup | jq '.'
- Expect confirmation that setup completed; if setup fails, check logs for installation diagnostics.

4) Start bot
- curl -X POST -sS http://127.0.0.1:8000/start | jq '.'
- After start, GET /status should reflect running state.

5) Status
- curl -sS http://127.0.0.1:8000/status | jq '.'
- Expect { "status": "running", ... } when bot started.

Notes and diagnostics:
- Route handlers log key diagnostics: chosen_adb_path and installation diagnostics for triage; check server stdout/stderr or configured log files.
- If detection returns an empty list, /emulators/connect should still tolerate that and return a diagnostics object describing the failure.

Windows-specific notes:
- BlueStacks: enable ADB (Settings > Advanced > Enable ADB) and connect via adb connect 127.0.0.1:5555 (port may vary).
- Ensure the backend process can reach adb. If using a bundled ADB, set the path via environment or project configuration.

Troubleshooting:
- If /start refuses to start, confirm /setup has been executed and returned success diagnostics.
- Inspect server logs for chosen_adb_path and detector diagnostics keys mentioned above.

End of checklist.
