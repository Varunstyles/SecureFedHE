# SecureFedHE — Windows Deployment Notes

## Quick Start

```
Double-click setup.bat
```

That's it for a guided install. Everything below is for troubleshooting or manual control.

---

## IP Configuration

Every PC needs its correct local IP in `config.json`. Find yours:

```
ipconfig
```

Look for **IPv4 Address** under your active adapter (usually Wi-Fi or Ethernet). It will look like `192.168.1.xxx`. Edit `config.json` and replace the placeholder IPs in `ring.nodes[]`.

**All 5 PCs must edit the same `config.json` with all 5 IPs before starting.**

---

## Node Start Order

Start in this exact order:

```
PC 1 → python launch.py --id 1
PC 2 → python launch.py --id 2
PC 3 → python launch.py --id 3
PC 4 → python launch.py --id 4
PC 0 → python launch.py --id 0    ← master, starts last
```

Node 0 (master) waits up to 2 minutes for all others to come online, then fires the ring automatically.

---

## Dashboard

The dashboard runs only on Node 0 (master PC). After Node 0 starts:

```
python dashboard\dashboard.py
```

Then open in any browser on any PC on the same network:

```
http://192.168.1.101:8080
```
(replace with your Node 0's actual IP)

To enable Claude explanations, add your API key to `config.json`:
```json
"dashboard": {
    "port": 8080,
    "claude_api_key": "sk-ant-..."
}
```
Or set the environment variable before launching:
```bat
set ANTHROPIC_API_KEY=sk-ant-...
python dashboard\dashboard.py
```

---

## Certificate Distribution

Run `generate_certs.py` **once on Node 0 only**:

```
python generate_certs.py
```

Then copy the entire `certs\` folder to every other PC. After copying:
- Delete `certs\ca.key` from Node 0 (the CA private key — keep it secret or destroy it)

Directory to copy:
```
SecureFedHE\
    certs\
        ca.crt          ← copy to all PCs
        server.crt      ← copy to all PCs
        server.key      ← copy to all PCs
        client.crt      ← copy to all PCs
        client.key      ← copy to all PCs
        ca.key          ← DELETE after distributing
```

---

## Firewall

Windows Firewall will block inter-node traffic by default. `setup.bat` adds the rules automatically, but if you need to do it manually (run as Administrator):

```bat
:: Allow node traffic (port 8000) — run on ALL 5 PCs
netsh advfirewall firewall add rule name="SecureFedHE Node" dir=in action=allow protocol=TCP localport=8000

:: Allow dashboard (port 8080) — run on Node 0 only
netsh advfirewall firewall add rule name="SecureFedHE Dashboard" dir=in action=allow protocol=TCP localport=8080
```

To remove rules later:
```bat
netsh advfirewall firewall delete rule name="SecureFedHE Node"
netsh advfirewall firewall delete rule name="SecureFedHE Dashboard"
```

---

## Path Separators

The codebase uses `pathlib.Path` throughout, which handles Windows backslashes automatically. If you ever need to pass paths manually on the command line, both `/` and `\` work in most contexts.

---

## Single-PC Dev Mode

To test the full system on one PC without certificates or multiple machines:

```bat
:: Terminal 1 — start node in dev mode
python launch.py --id 0 --dev

:: Terminal 2 — start dashboard in dev mode
python dashboard\dashboard.py --dev
```

Then open `http://localhost:8080` and click **Run Demo** to simulate training.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` | Add Python to PATH during install, or use `py` instead of `python` |
| `pip install` fails | Check internet connection; try `pip install -r requirements.txt --verbose` |
| Node says `Certificate not found` | Run `python generate_certs.py` on Node 0, copy `certs\` to all PCs |
| Node says `Connection refused` | Check firewall rules; confirm IPs in `config.json` match `ipconfig` output |
| Dashboard shows all nodes as `unreachable` | Nodes not started yet, or firewall blocking port 8000 |
| `Port already in use` | Another process on port 8000/8080; change port in `config.json` or kill the other process |
| `ModuleNotFoundError` | Virtual environment not activated; run `.venv\Scripts\activate` first |
| Training stops mid-ring | One node crashed; check `logs\node_X.log` on the failing PC |

---

## Logs

Each node writes a structured JSON audit log:

```
logs\node_0.log    ← Node 0 (master)
logs\node_1.log
...
```

View the last 50 lines:
```bat
powershell -command "Get-Content logs\node_0.log -Tail 50"
```

---

## Virtual Environment

`setup.bat` creates `.venv\` in the project folder. To activate it manually:

```bat
.venv\Scripts\activate.bat
```

To deactivate:
```bat
deactivate
```
