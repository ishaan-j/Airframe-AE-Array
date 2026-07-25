"""
final_demo.py — Airframe AE Array console, driven by real tap data.

Serves the Three.js airframe simulation (airframe_ae_console.html, embedded
below unmodified except for a live-feed hook) from a small local HTTP server,
and pushes real acoustic-tap events into it over Server-Sent Events.

Taps come from hardware_code.py's pipeline: read two MAX4466 amplitudes off
an Arduino's serial port, triangulate a position on the metal sheet, and
normalize it to [0,1] sheet coordinates. The browser maps that position onto
the fuselage barrel and runs it through the console's existing TDOA/solve/
classification physics — the same code path the "Trigger test event" button
uses, just with a real (not random) source location.

Falls back to MockSerial (cycles canned tap positions) when
USE_REAL_HARDWARE is False or the Arduino can't be reached.
"""

import http.server
import json
import queue
import threading
import time
import webbrowser

import numpy as np
import serial
import serial.tools.list_ports

# ═══════════════════════════════════════════════════════
# CONFIGURATION — update these to match your setup
# ═══════════════════════════════════════════════════════

USE_REAL_HARDWARE = True  # ← flip to True when Arduino arrives

# Measure your actual tray dimensions in cm
SHEET_WIDTH  = 30.0
SHEET_HEIGHT = 25.0

# Measure where you physically place mics on tray (cm from bottom-left)
MIC_POSITIONS = np.array([
    [3.0,  12.5],   # Mic 1 — left side, middle height
    [27.0, 12.5],   # Mic 2 — right side, middle height
])

SERIAL_BAUD = 115200
HTTP_HOST   = "127.0.0.1"
HTTP_PORT   = 8765

# ═══════════════════════════════════════════════════════
# MOCK SERIAL — used when USE_REAL_HARDWARE = False
# ═══════════════════════════════════════════════════════

class MockSerial:
    """
    Simulates Arduino serial output for testing without hardware.
    Cycles through tap positions across the sheet automatically.
    """
    def __init__(self):
        self.last_event   = time.time()
        self.sequence_idx = 0

        # Each entry: [amp1, amp2] representing different tap positions
        # High amp1 = tap near left mic
        # High amp2 = tap near right mic
        # Equal     = tap in center
        self.tap_sequence = [
            [380, 60],   # far left
            [280, 120],  # left of center
            [200, 200],  # center
            [120, 280],  # right of center
            [60,  380],  # far right
            [320, 80],   # left
            [80,  320],  # right
        ]

    @property
    def in_waiting(self):
        # Computed live (rather than only set as a side effect of readline())
        # so read_tap()'s "if ser.in_waiting" gate actually opens on schedule.
        return 1 if (time.time() - self.last_event) > 2.5 else 0

    def readline(self):
        current_time = time.time()
        if current_time - self.last_event > 2.5:
            self.last_event = current_time
            amps = self.tap_sequence[
                self.sequence_idx % len(self.tap_sequence)
            ]
            self.sequence_idx += 1

            # Add noise to make it realistic
            amps = [max(0, a + np.random.randint(-25, 25))
                    for a in amps]

            ts   = int(current_time * 1000)
            line = f"{ts},{amps[0]},{amps[1]}\n"
            return line.encode("utf-8")

        return b""


# ═══════════════════════════════════════════════════════
# SERIAL CONNECTION
# ═══════════════════════════════════════════════════════

def connect_arduino():
    """
    Auto-detects Arduino port.
    If auto-detect fails, lists available ports and asks you to pick.
    """
    ports = list(serial.tools.list_ports.comports())

    for p in ports:
        desc = p.description.lower()
        dev  = p.device.lower()
        if ("arduino" in desc or
            "ttyacm"  in dev  or
            "ttyusb"  in dev  or
            "usbserial" in dev):
            print(f"Auto-detected Arduino on {p.device}")
            return serial.Serial(p.device, SERIAL_BAUD, timeout=0.05)

    print("Could not auto-detect Arduino. Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")

    if not ports:
        raise Exception("No serial ports found. Check USB connection.")

    idx  = int(input("Enter port number: "))
    port = ports[idx].device
    return serial.Serial(port, SERIAL_BAUD, timeout=0.05)


# ═══════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════

def read_tap(ser):
    """
    Read one line from serial.
    Returns (timestamp, [amp1, amp2]) or (None, None).
    """
    try:
        if ser.in_waiting:
            raw  = ser.readline()
            line = raw.decode("utf-8").strip()
            if "," in line:
                parts = line.split(",")
                if len(parts) == 3:
                    ts   = int(parts[0])
                    amps = [int(parts[1]), int(parts[2])]
                    return ts, amps
    except Exception:
        pass
    return None, None


def triangulate_2mic(amplitudes, mic_positions):
    """
    2-microphone amplitude-based position estimation.

    Physics: sound amplitude falls off with distance.
    Higher amplitude at mic = tap is closer to that mic.
    Gives us left-right position reliably.
    Y is fixed to sheet center (limitation of 2-mic setup).

    For full 2D localization you need 3+ mics.
    """
    amp1, amp2 = float(amplitudes[0]), float(amplitudes[1])
    total = amp1 + amp2

    if total < 1:
        return None

    w1 = amp1 / total
    w2 = amp2 / total

    x = w1 * mic_positions[0][0] + w2 * mic_positions[1][0]
    y = SHEET_HEIGHT / 2  # fixed center — limitation of 2 mics

    x = np.clip(x, 0, SHEET_WIDTH)
    return float(x), float(y)


# ═══════════════════════════════════════════════════════
# LIVE-FEED BROADCAST (server-sent events -> browser console)
# ═══════════════════════════════════════════════════════

_clients = []
_clients_lock = threading.Lock()
_live_state = {"live": False}


def _sse(event_name, data):
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def broadcast(event_name, data):
    payload = _sse(event_name, data)
    with _clients_lock:
        dead = []
        for q in _clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


def serial_loop(ser):
    """Background thread: poll serial, triangulate, broadcast normalized taps."""
    while True:
        ts, amps = read_tap(ser)
        if ts is not None and amps is not None:
            result = triangulate_2mic(amps, MIC_POSITIONS)
            if result:
                sx, sy = result
                nx = float(np.clip(sx / SHEET_WIDTH, 0, 1))
                ny = float(np.clip(sy / SHEET_HEIGHT, 0, 1))
                broadcast("tap", {
                    "ts": ts, "amp1": amps[0], "amp2": amps[1],
                    "nx": nx, "ny": ny,
                })
        time.sleep(0.01)


# ═══════════════════════════════════════════════════════
# HTTP SERVER — serves the console + the /stream SSE feed
# ═══════════════════════════════════════════════════════

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, needed for a long-lived /stream

    def log_message(self, fmt, *args):
        pass  # quiet; serial_loop / main already report status

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = CONSOLE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = queue.Queue(maxsize=64)
            with _clients_lock:
                _clients.append(q)
            try:
                self.wfile.write(_sse("status", dict(_live_state)))
                self.wfile.flush()
                while True:
                    try:
                        payload = q.get(timeout=15)
                    except queue.Empty:
                        payload = b": keep-alive\n\n"
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                with _clients_lock:
                    if q in _clients:
                        _clients.remove(q)
        else:
            self.send_response(404)
            self.end_headers()


# ═══════════════════════════════════════════════════════
# EMBEDDED CONSOLE — airframe_ae_console.html, with one hook added:
# a "Feed" status cell in the header, and a small liveInject()/tapToBarrel()
# pair wired to /stream so a real triangulated tap runs through the exact
# same makeEvent() -> solve() -> present() pipeline as the test-event button.
# Defined here, before the entry point, since server.serve_forever() below
# blocks forever — anything after it in the file would never actually run.
# ═══════════════════════════════════════════════════════
CONSOLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Airframe AE Array — Fault Triangulation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --void:#060B10;
  --deck:#0B131B;
  --panel:#0E1922;
  --panel-2:#122029;
  --rule:#1B2C39;
  --rule-2:#274152;
  --ink:#D6E5ED;
  --ink-2:#7B95A4;
  --ink-3:#4C6474;
  /* C-scan amplitude ramp: cold -> hot */
  --a1:#123F6B;
  --a2:#1B8E96;
  --a3:#3FD2C7;
  --a4:#E8C34E;
  --a5:#FF6B3D;
  --a6:#FF3B5C;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
  --cond:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;background:var(--void);color:var(--ink);
  font-family:var(--sans);font-size:13px;line-height:1.45;
  -webkit-font-smoothing:antialiased;
}
button,select,input{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--a3);outline-offset:2px}

/* ---------- shell ---------- */
/* Pin to the viewport so the footer stays put; inner regions scroll instead. */
.shell{display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* ---------- header ---------- */
header{
  border-bottom:1px solid var(--rule);background:var(--deck);
  display:flex;align-items:stretch;flex-wrap:wrap;
}
.mark{
  padding:10px 16px;border-right:1px solid var(--rule);
  display:flex;flex-direction:column;justify-content:center;min-width:210px;
}
.mark h1{
  margin:0;font-family:var(--cond);font-weight:700;font-size:17px;
  letter-spacing:.14em;text-transform:uppercase;
}
.mark .sub{font-family:var(--mono);font-size:10px;color:var(--ink-3);letter-spacing:.16em;text-transform:uppercase}
.hstat{display:flex;flex:1;flex-wrap:wrap}
.hcell{
  padding:10px 16px;border-right:1px solid var(--rule);min-width:118px;
  display:flex;flex-direction:column;gap:2px;justify-content:center;
}
.hcell .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3)}
.hcell .v{font-family:var(--mono);font-size:13px;color:var(--ink)}
.hcell .v.ok{color:var(--a3)}
.hcell .v.hot{color:var(--a6)}
.pulse{display:inline-block;width:7px;height:7px;background:var(--a3);margin-right:6px;vertical-align:1px;
  animation:beat 2s ease-in-out infinite}
@keyframes beat{0%,100%{opacity:1}50%{opacity:.25}}

/* ---------- body grid ---------- */
.grid{display:grid;grid-template-columns:286px minmax(0,1fr) 330px;flex:1;min-height:0}
/* min-height:0 lets the column shrink inside the grid row so its inner
   scroll areas (event log, event detail) bound themselves to the viewport. */
.col{border-right:1px solid var(--rule);background:var(--panel);display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden auto}
.col:last-child{border-right:0}
.center{background:var(--void);position:relative;min-width:0}

.sect{border-bottom:1px solid var(--rule)}
.sect > h2{
  margin:0;padding:9px 14px 8px;font-family:var(--mono);font-size:9.5px;
  letter-spacing:.2em;text-transform:uppercase;color:var(--ink-3);
  display:flex;justify-content:space-between;align-items:center;gap:8px;
}
.sect > h2 em{font-style:normal;color:var(--ink-2)}
.sect .body{padding:0 14px 14px}

/* ---------- zone rows ---------- */
.zone{display:grid;grid-template-columns:1fr auto;gap:2px 10px;padding:7px 0;border-top:1px solid var(--rule)}
.zone:first-child{border-top:0}
.zone .n{font-family:var(--mono);font-size:11.5px;color:var(--ink)}
.zone .c{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.zone .bar{grid-column:1/3;height:3px;background:var(--rule);position:relative;margin-top:3px}
.zone .bar i{position:absolute;inset:0 auto 0 0;background:var(--a2)}
.zone.hit .n{color:var(--a4)}
.zone.hit .bar i{background:var(--a5)}

/* ---------- controls ---------- */
.ctl{display:flex;flex-direction:column;gap:5px;padding:9px 0;border-top:1px solid var(--rule)}
.ctl:first-child{border-top:0}
.ctl label{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3);
  display:flex;justify-content:space-between;gap:8px}
.ctl label b{color:var(--a3);font-weight:500}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:16px;background:transparent;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:2px;background:var(--rule-2)}
input[type=range]::-moz-range-track{height:2px;background:var(--rule-2)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:3px;height:14px;background:var(--a3);margin-top:-6px;border-radius:0}
input[type=range]::-moz-range-thumb{width:3px;height:14px;background:var(--a3);border:0;border-radius:0}
select{
  background:var(--panel-2);border:1px solid var(--rule-2);padding:6px 8px;
  font-family:var(--mono);font-size:11.5px;width:100%;border-radius:2px;
}

/* ---------- log ---------- */
.logwrap{flex:1;overflow:auto;min-height:80px}
.logrow{
  display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:baseline;
  width:100%;text-align:left;background:none;border:0;border-top:1px solid var(--rule);
  padding:7px 14px;cursor:pointer;font-family:var(--mono);font-size:11px;
}
.logrow:hover{background:var(--panel-2)}
.logrow[aria-current=true]{background:var(--panel-2);box-shadow:inset 2px 0 0 var(--a4)}
.logrow .t{color:var(--ink-3);font-size:10px}
.logrow .d{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.logrow .a{color:var(--a4)}
.empty{padding:14px;font-family:var(--mono);font-size:11px;color:var(--ink-3);line-height:1.6}

/* ---------- stage ---------- */
.ruler{height:38px;border-bottom:1px solid var(--rule);background:var(--deck);position:relative}
.ruler canvas{display:block;width:100%;height:100%}
.stage{position:absolute;inset:38px 0 0 0}
.stage canvas{display:block;width:100%;height:100%;touch-action:none}
.hint{
  position:absolute;left:14px;bottom:12px;font-family:var(--mono);font-size:10px;
  color:var(--ink-3);letter-spacing:.1em;text-transform:uppercase;pointer-events:none
}
.tip{
  position:absolute;pointer-events:none;background:var(--deck);border:1px solid var(--rule-2);
  padding:5px 8px;font-family:var(--mono);font-size:10.5px;white-space:nowrap;
  transform:translate(10px,-50%);opacity:0;transition:opacity .12s;border-radius:2px;z-index:4
}
.tip b{color:var(--a3);font-weight:500}
.phase{
  position:absolute;left:14px;top:12px;font-family:var(--mono);font-size:10px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--a4);pointer-events:none;opacity:0;transition:opacity .2s
}
.phase.on{opacity:1}

/* ---------- event panel ---------- */
.verdict{padding:12px 14px;border-bottom:1px solid var(--rule);background:var(--panel-2)}
.verdict .cls{font-family:var(--cond);font-size:19px;font-weight:700;letter-spacing:.03em;line-height:1.15;color:var(--a5)}
.verdict .conf{font-family:var(--mono);font-size:10.5px;color:var(--ink-2);margin-top:4px}
.verdict .act{
  margin-top:9px;font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--void);background:var(--a4);display:inline-block;padding:3px 7px
}
.verdict .act.sev{background:var(--a6);color:#fff}

.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-family:var(--mono);font-size:11.5px;padding-top:2px}
.kv dt{color:var(--ink-3);letter-spacing:.05em}
.kv dd{margin:0;color:var(--ink);text-align:right}
.kv dd.hi{color:var(--a3)}

table.tri{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
table.tri th{
  text-align:right;font-weight:400;color:var(--ink-3);font-size:9px;letter-spacing:.12em;
  text-transform:uppercase;padding:0 0 5px;border-bottom:1px solid var(--rule)
}
table.tri th:first-child{text-align:left}
table.tri td{padding:5px 0;border-bottom:1px solid var(--rule);text-align:right;color:var(--ink-2)}
table.tri td:first-child{text-align:left;color:var(--ink)}
table.tri tr:last-child td{border-bottom:0}
.vx{display:inline-block;width:7px;height:7px;margin-right:6px;vertical-align:0}

.chart{width:100%;height:74px;display:block}
.chart.spec{height:88px}
.caption{font-family:var(--mono);font-size:9.5px;color:var(--ink-3);letter-spacing:.1em;text-transform:uppercase;
  display:flex;justify-content:space-between;padding-top:5px}

/* ---------- footer ---------- */
footer{
  border-top:1px solid var(--rule);background:var(--deck);display:flex;flex-wrap:wrap;
  align-items:center;gap:8px;padding:9px 14px;
}
.btn{
  background:var(--panel-2);border:1px solid var(--rule-2);padding:7px 13px;cursor:pointer;
  font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;border-radius:2px;
  transition:background .12s,border-color .12s,color .12s;
}
.btn:hover{border-color:var(--a3);color:var(--a3)}
.btn.prime{background:var(--a6);border-color:var(--a6);color:#fff}
.btn.prime:hover{background:#ff5470;border-color:#ff5470;color:#fff}
.btn[aria-pressed=true]{border-color:var(--a3);color:var(--a3);background:#0d2a2c}
.segs{display:flex;border:1px solid var(--rule-2);border-radius:2px;overflow:hidden}
.segs .btn{border:0;border-right:1px solid var(--rule-2);border-radius:0;padding:7px 11px}
.segs .btn:last-child{border-right:0}
.spacer{flex:1}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-family:var(--mono);font-size:9.5px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.legend span{display:flex;align-items:center;gap:6px}
.legend i{width:9px;height:9px;display:block}

@media (max-width:1080px){
  /* Stacked layout: let the whole page scroll again instead of pinning height. */
  .shell{height:auto;overflow:visible}
  .grid{grid-template-columns:1fr;grid-template-rows:auto auto auto}
  .center{order:-1;min-height:440px;border-bottom:1px solid var(--rule)}
  .col{border-right:0;border-bottom:1px solid var(--rule);overflow:visible}
  .logwrap{max-height:190px}
}
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.001ms !important;transition-duration:.001ms !important}
}
</style>
</head>
<body>
<div class="shell">

<header>
  <div class="mark">
    <h1>Airframe AE Array</h1>
    <div class="sub">Acoustic emission &middot; passive SHM</div>
  </div>
  <div class="hstat">
    <div class="hcell"><span class="k">Tail</span><span class="v">N884SG</span></div>
    <div class="hcell"><span class="k">Type</span><span class="v">Narrow-body / Al-Li</span></div>
    <div class="hcell"><span class="k">Cycles</span><span class="v">18 412</span></div>
    <div class="hcell"><span class="k">Nodes online</span><span class="v ok" id="hNodes">74 / 74</span></div>
    <div class="hcell"><span class="k">State</span><span class="v ok" id="hState"><i class="pulse"></i>Armed</span></div>
    <div class="hcell"><span class="k">Feed</span><span class="v" id="hFeed">&mdash; connecting &mdash;</span></div>
    <div class="hcell"><span class="k">Open events</span><span class="v" id="hOpen">0</span></div>
  </div>
</header>

<div class="grid">

  <!-- ================= LEFT ================= -->
  <div class="col">
    <div class="sect">
      <h2>Coverage zones <em id="zoneTot">74 nodes</em></h2>
      <div class="body" id="zones"></div>
    </div>

    <div class="sect">
      <h2>Array settings</h2>
      <div class="body">
        <div class="ctl">
          <label for="thr">Hit threshold <b><span id="thrV">42</span> dB<sub>AE</sub></b></label>
          <input type="range" id="thr" min="30" max="60" step="1" value="42">
        </div>
        <div class="ctl">
          <label for="mode">Propagation mode</label>
          <select id="mode">
            <option value="3120" selected>A0 flexural — 3120 m/s</option>
            <option value="5420">S0 extensional — 5420 m/s</option>
            <option value="2980">A0 cold soak — 2980 m/s</option>
          </select>
        </div>
        <div class="ctl">
          <label for="noise">Timing jitter <b><span id="noiseV">3.0</span> µs</b></label>
          <input type="range" id="noise" min="0" max="12" step="0.5" value="3">
        </div>
      </div>
    </div>

    <div class="sect" style="border-bottom:0;display:flex;flex-direction:column;flex:1;min-height:0">
      <h2>Event log <em id="logCount">0</em></h2>
      <div class="logwrap" id="log">
        <div class="empty">No qualifying hits since the last data download.<br>Trigger a test event to exercise the array.</div>
      </div>
    </div>
  </div>

  <!-- ================= CENTER ================= -->
  <div class="center">
    <div class="ruler"><canvas id="ruler"></canvas></div>
    <div class="stage" id="stage">
      <canvas id="gl"></canvas>
      <div class="phase" id="phase"></div>
      <div class="hint">Drag to orbit &middot; scroll to zoom &middot; hover a node</div>
      <div class="tip" id="tip"></div>
    </div>
  </div>

  <!-- ================= RIGHT ================= -->
  <div class="col">
    <div id="evEmpty" class="sect" style="border-bottom:0">
      <h2>Event detail</h2>
      <div class="empty">
        Each node listens for stress waves in the skin.<br><br>
        When a flaw releases energy, the three nodes forming the mesh cell around it hear it at slightly
        different times. Those differences fix the source; the frequency content names the flaw.
      </div>
    </div>

    <div id="evPanel" hidden>
      <div class="verdict">
        <div class="cls" id="evCls">—</div>
        <div class="conf" id="evConf">—</div>
        <span class="act" id="evAct">—</span>
      </div>

      <div class="sect">
        <h2>Source fix <em id="evId">—</em></h2>
        <div class="body">
          <dl class="kv">
            <dt>Zone</dt><dd id="fxZone">—</dd>
            <dt>Station</dt><dd class="hi" id="fxStation">—</dd>
            <dt>Mesh cell</dt><dd id="fxCell">—</dd>
            <dt>Residual</dt><dd id="fxRes">—</dd>
            <dt>95% radius</dt><dd id="fxEll">—</dd>
            <dt>Solver error</dt><dd id="fxErr">—</dd>
          </dl>
        </div>
      </div>

      <div class="sect">
        <h2>Solving triangle <em>arrival &amp; TDOA</em></h2>
        <div class="body">
          <table class="tri">
            <thead><tr><th>Node</th><th>Range</th><th>Arrival</th><th>&Delta;t</th><th>Amp</th></tr></thead>
            <tbody id="triBody"></tbody>
          </table>
        </div>
      </div>

      <div class="sect">
        <h2>First hit waveform <em id="wfNode">—</em></h2>
        <div class="body">
          <canvas class="chart" id="wave"></canvas>
          <div class="caption"><span>0 µs</span><span id="wfMeta">—</span><span>256 µs</span></div>
        </div>
      </div>

      <div class="sect" style="border-bottom:0">
        <h2>Spectrum <em>flaw signature</em></h2>
        <div class="body">
          <canvas class="chart spec" id="spec"></canvas>
          <div class="caption"><span>0</span><span id="spMeta">—</span><span>400 kHz</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

<footer>
  <button class="btn prime" id="btnInject">Trigger test event</button>
  <button class="btn" id="btnAuto" aria-pressed="false">Auto monitor</button>
  <div class="segs" role="group" aria-label="Camera view">
    <button class="btn" data-view="iso">Iso</button>
    <button class="btn" data-view="top">Top</button>
    <button class="btn" data-view="side">Side</button>
    <button class="btn" data-view="aft">Aft</button>
  </div>
  <button class="btn" id="btnFill" aria-pressed="false">Mesh fill</button>
  <button class="btn" id="btnReplay">Replay</button>
  <div class="spacer"></div>
  <div class="legend">
    <span><i style="background:var(--a1)"></i>Quiet</span>
    <span><i style="background:var(--a3)"></i>Armed</span>
    <span><i style="background:var(--a4)"></i>Hit</span>
    <span><i style="background:var(--a6)"></i>Source</span>
  </div>
</footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
"use strict";
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ===================================================================
   1. AIRFRAME GEOMETRY
   World units: 1 unit = 2 m. Nose at +X, tail at -X, +Y up, +Z right.
   =================================================================== */
const MPU = 2.0;
const FUSE_LEN = 16, FUSE_R = 1.0;
const IN_PER_M = 39.3701;

function fuseRad(t){
  if(t < 0.16){ const u = t/0.16; return FUSE_R*Math.pow(Math.sin(u*Math.PI/2), 0.6); }
  if(t > 0.76){ const u = (t-0.76)/0.24; return FUSE_R*(1 - 0.80*Math.pow(u,1.6)); }
  return FUSE_R;
}
function fuseLift(t){ return t > 0.70 ? 0.62*Math.pow((t-0.70)/0.30, 2.2) : 0; }
function fusePt(t, th, off){
  const r = fuseRad(t) + (off||0);
  return new THREE.Vector3((0.5-t)*FUSE_LEN, fuseLift(t) + r*Math.cos(th), r*Math.sin(th));
}
function wrapPi(a){ while(a > Math.PI) a -= 2*Math.PI; while(a <= -Math.PI) a += 2*Math.PI; return a; }

// bilinear quad panels (wings / stabilisers / fin)
function quadPt(q, s, c, off){
  const le = new THREE.Vector3().lerpVectors(q.rLE, q.tLE, s);
  const te = new THREE.Vector3().lerpVectors(q.rTE, q.tTE, s);
  const p  = new THREE.Vector3().lerpVectors(le, te, c);
  if(off) p.y += off;
  return p;
}
const V = (x,y,z)=>new THREE.Vector3(x,y,z);

const WING_R = { rLE:V( 0.9,-0.60, 0.92), rTE:V(-3.5,-0.60, 0.92), tLE:V(-1.7, 0.12, 7.2), tTE:V(-3.4, 0.12, 7.2) };
const WING_L = { rLE:V( 0.9,-0.60,-0.92), rTE:V(-3.5,-0.60,-0.92), tLE:V(-1.7, 0.12,-7.2), tTE:V(-3.4, 0.12,-7.2) };
const HS_R   = { rLE:V(-5.7, 0.34, 0.50), rTE:V(-7.5, 0.34, 0.50), tLE:V(-6.8, 0.52, 3.05), tTE:V(-7.7, 0.52, 3.05) };
const HS_L   = { rLE:V(-5.7, 0.34,-0.50), rTE:V(-7.5, 0.34,-0.50), tLE:V(-6.8, 0.52,-3.05), tTE:V(-7.7, 0.52,-3.05) };
const FIN    = { rLE:V(-5.2, 0.70, 0), rTE:V(-7.8, 0.70, 0), tLE:V(-7.05, 3.45, 0), tTE:V(-8.05, 3.45, 0) };

/* ===================================================================
   2. SENSOR PANELS
   Each panel maps parameters (a,b) -> surface point, and defines the
   in-plane distance a stress wave actually travels between two points.
   The barrel is a developable cylinder, so its geodesic is exact in
   unrolled (axial, circumferential) coordinates.
   =================================================================== */
const PANELS = {
  barrel: {
    id:"barrel", label:"Fuselage barrel",
    aMin:0.20, aMax:0.72, bMin:-Math.PI, bMax:Math.PI, bWrap:true,
    point:(a,b,off)=>fusePt(a,b,off),
    dist:(a1,b1,a2,b2)=>{
      const du = (a2-a1)*FUSE_LEN*MPU;
      const dw = wrapPi(b2-b1)*FUSE_R*MPU;
      return Math.hypot(du,dw);
    },
    step:(a,b,du,dw)=>[a - du/(FUSE_LEN*MPU), b + dw/(FUSE_R*MPU)]
  }
};
function makeWingPanel(id,label,quad,side){
  const spanLen = quad.rLE.distanceTo(quad.tLE)*MPU;
  const chordLen= quad.rLE.distanceTo(quad.rTE)*MPU;
  return {
    id, label, quad, side,
    aMin:0.16, aMax:0.86, bMin:0.18, bMax:0.82, bWrap:false,
    point:(a,b,off)=>quadPt(quad,a,b,(off||0)+0.055),
    dist:(a1,b1,a2,b2)=>quadPt(quad,a1,b1,0).distanceTo(quadPt(quad,a2,b2,0))*MPU,
    step:(a,b,du,dw)=>[a + du/spanLen, b + dw/chordLen],
    spanLen, chordLen
  };
}
PANELS.wingR = makeWingPanel("wingR","RH wing box",WING_R,"R");
PANELS.wingL = makeWingPanel("wingL","LH wing box",WING_L,"L");

// Wing/body junction points (root-chord midpoints). A stress wave couples
// across these joints, so an event on one panel is also heard by sensors on the
// neighbouring panel near the joint. Path length is source->joint->node.
const WING_JOINT = {
  wingR: new THREE.Vector3().addVectors(WING_R.rLE, WING_R.rTE).multiplyScalar(0.5),
  wingL: new THREE.Vector3().addVectors(WING_L.rLE, WING_L.rTE).multiplyScalar(0.5)
};
const JOINT_LOSS = 4;      // extra dB attenuation crossing the structural discontinuity
const COUPLE_REACH = 3.0;  // world units: cap on the source->joint->node path so only
                           // sensors genuinely near the joint couple (not the whole airframe)

/* --- node roster ------------------------------------------------- */
const RINGS = 7, AROUND = 8, WROWS = 3, WCOLS = 3;
const nodes = [];   // {i,panel,a,b,pos,id,zone}
const tris  = [];   // {v:[i,i,i], panel}

function zoneOf(t){
  if(t < 0.36) return "Fwd barrel";
  if(t < 0.56) return "Mid barrel";
  return "Aft barrel";
}
function barrelId(a,b){
  const fs = Math.round((0.0 + a*FUSE_LEN*MPU)*IN_PER_M);
  const str = 1 + Math.round(Math.abs(b)/Math.PI*24);
  return "FS"+String(fs).padStart(4,"0")+"·S"+String(str).padStart(2,"0")+(b>=0?"R":"L");
}
function wingId(p,a,b){
  const ws = Math.round((0.92 + a*(7.2-0.92))*MPU*IN_PER_M);
  return p.side+"H·WS"+String(ws).padStart(3,"0")+"·C"+(1+Math.round(b*3));
}

const bIdx = [];
for(let i=0;i<RINGS;i++){
  bIdx.push([]);
  const a = 0.20 + (0.72-0.20)*(i/(RINGS-1));
  for(let j=0;j<AROUND;j++){
    const b = wrapPi(j*2*Math.PI/AROUND);
    bIdx[i].push(nodes.length);
    nodes.push({i:nodes.length,panel:PANELS.barrel,a,b,pos:fusePt(a,b,0.012),id:barrelId(a,b),zone:zoneOf(a)});
  }
}
for(let i=0;i<RINGS-1;i++) for(let j=0;j<AROUND;j++){
  const j2=(j+1)%AROUND;
  const A=bIdx[i][j],B=bIdx[i][j2],C=bIdx[i+1][j2],D=bIdx[i+1][j];
  tris.push({v:[A,B,C],panel:PANELS.barrel},{v:[A,C,D],panel:PANELS.barrel});
}
[PANELS.wingR,PANELS.wingL].forEach(p=>{
  const g=[];
  for(let i=0;i<WROWS;i++){
    g.push([]);
    const a = 0.18 + 0.66*(i/(WROWS-1));
    for(let j=0;j<WCOLS;j++){
      const b = 0.20 + 0.60*(j/(WCOLS-1));
      g[i].push(nodes.length);
      nodes.push({i:nodes.length,panel:p,a,b,pos:p.point(a,b,0.012),id:wingId(p,a,b),zone:p.label});
    }
  }
  for(let i=0;i<WROWS-1;i++) for(let j=0;j<WCOLS-1;j++){
    const A=g[i][j],B=g[i][j+1],C=g[i+1][j+1],D=g[i+1][j];
    tris.push({v:[A,B,C],panel:p},{v:[A,C,D],panel:p});
  }
});

/* ===================================================================
   3. SIGNAL MODEL
   =================================================================== */
const FS_SAMP = 2e6, NSAMP = 512;
const ATTEN = 2.6;               // dB per metre in stiffened skin
let   waveV = 3120;              // m/s
let   jitter = 3.0;              // µs 1-sigma
let   thresholdDb = 42;

const CLASSES = [
  {lo:0,   hi:70,  name:"Fastener fretting",        act:"Monitor — trend at next A-check", sev:0},
  {lo:70,  hi:140, name:"Skin–stringer disbond",    act:"Inspect before next flight",      sev:1},
  {lo:140, hi:260, name:"Fatigue crack extension",  act:"Ground — NDT confirmation",       sev:2},
  {lo:260, hi:999, name:"Fibre breakage / composite",act:"Inspect before next flight",     sev:1}
];
function classify(fPeak){ return CLASSES.find(c=>fPeak>=c.lo && fPeak<c.hi) || CLASSES[2]; }

function synthBurst(fc, q, seed){
  const rnd = mulberry(seed);
  const w = new Float32Array(NSAMP);
  const f1 = fc*1e3, f2 = fc*1e3*0.58;
  const tau1 = q/(Math.PI*f1), tau2 = tau1*1.7;
  const rise = 4e-6;
  for(let n=0;n<NSAMP;n++){
    const t = n/FS_SAMP;
    const env = t < rise ? t/rise : 1;
    const s = env*(Math.exp(-t/tau1)*Math.sin(2*Math.PI*f1*t)
                 + 0.55*Math.exp(-t/tau2)*Math.sin(2*Math.PI*f2*t + 1.1));
    w[n] = s + (rnd()-0.5)*0.045;
  }
  let m=0; for(let n=0;n<NSAMP;n++) m=Math.max(m,Math.abs(w[n]));
  for(let n=0;n<NSAMP;n++) w[n]/=m;
  return w;
}
function spectrum(w){
  const N = w.length, half = N>>1, out = new Float32Array(half);
  for(let k=0;k<half;k++){
    let re=0, im=0;
    for(let n=0;n<N;n++){
      const win = 0.5-0.5*Math.cos(2*Math.PI*n/(N-1));
      const ang = -2*Math.PI*k*n/N;
      re += w[n]*win*Math.cos(ang); im += w[n]*win*Math.sin(ang);
    }
    out[k] = Math.hypot(re,im);
  }
  let m=0; for(let k=0;k<half;k++) m=Math.max(m,out[k]);
  for(let k=0;k<half;k++) out[k]/=(m||1);
  return out;
}
function mulberry(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function gauss(rnd){ let u=0,v=0; while(!u)u=rnd(); while(!v)v=rnd();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

/* ===================================================================
   4. LOCALISATION — least-squares TDOA over the panel surface
   Unknowns reduce to (a,b): the emission instant drops out as the mean
   residual, so we minimise the scatter of (t_i - d_i/v).
   =================================================================== */
function residualAt(panel, a, b, hits){
  let sum=0, sum2=0;
  const off=[];
  for(let k=0;k<hits.length;k++){
    const n = nodes[hits[k].n];
    const d = panel.dist(a,b,n.a,n.b);
    const o = hits[k].t - d/waveV;
    off.push(o); sum+=o;
  }
  const mean = sum/hits.length;
  for(let k=0;k<off.length;k++){ const e=off[k]-mean; sum2+=e*e; }
  return {rms:Math.sqrt(sum2/hits.length), t0:mean};
}
function solve(panel, hits){
  let best={rms:Infinity,a:0,b:0,t0:0};
  const NA=110, NB=panel.bWrap?96:70;
  for(let i=0;i<NA;i++){
    const a = panel.aMin + (panel.aMax-panel.aMin)*(i/(NA-1));
    for(let j=0;j<NB;j++){
      const b = panel.bMin + (panel.bMax-panel.bMin)*(j/(NB-1));
      const r = residualAt(panel,a,b,hits);
      if(r.rms < best.rms) best={rms:r.rms,a,b,t0:r.t0};
    }
  }
  // pattern search refinement
  let sa=(panel.aMax-panel.aMin)/NA, sb=(panel.bMax-panel.bMin)/NB;
  for(let it=0; it<60; it++){
    let moved=false;
    const cand=[[best.a+sa,best.b],[best.a-sa,best.b],[best.a,best.b+sb],[best.a,best.b-sb]];
    for(const [ca,cb] of cand){
      const a=Math.min(panel.aMax,Math.max(panel.aMin,ca));
      const b=panel.bWrap?wrapPi(cb):Math.min(panel.bMax,Math.max(panel.bMin,cb));
      const r=residualAt(panel,a,b,hits);
      if(r.rms<best.rms){ best={rms:r.rms,a,b,t0:r.t0}; moved=true; }
    }
    if(!moved){ sa*=0.6; sb*=0.6; }
    if(sa<1e-6 && sb<1e-6) break;
  }
  return best;
}
// which mesh cell owns the fix? (barycentric in unrolled panel coords)
function cellFor(panel, a, b){
  let bestT=null, bestD=Infinity;
  for(const tr of tris){
    if(tr.panel!==panel) continue;
    // Unroll azimuth relative to THIS triangle (not the query b). Unrolling
    // against the query puts the seam at the query's antipode, which stretches
    // any cell crossing it across the whole barrel — and the query would then
    // test "inside" that antipodal cell ~half the time, highlighting the wrong
    // side of the fuselage. A per-cell reference keeps every cell compact.
    const ref = panel.bWrap ? nodes[tr.v[0]].b : 0;
    const to2 = panel.bWrap
        ? (aa,bb)=>[aa*FUSE_LEN*MPU, wrapPi(bb-ref)*FUSE_R*MPU]
        : (aa,bb)=>[aa*(panel.spanLen||1), bb*(panel.chordLen||1)];
    const P = to2(a,b);
    const [A,B,C]=tr.v.map(i=>to2(nodes[i].a,nodes[i].b));
    const d=(B[1]-C[1])*(A[0]-C[0])+(C[0]-B[0])*(A[1]-C[1]);
    if(Math.abs(d)<1e-9) continue;
    const l1=((B[1]-C[1])*(P[0]-C[0])+(C[0]-B[0])*(P[1]-C[1]))/d;
    const l2=((C[1]-A[1])*(P[0]-C[0])+(A[0]-C[0])*(P[1]-C[1]))/d;
    const l3=1-l1-l2;
    const pen=Math.min(0,l1)+Math.min(0,l2)+Math.min(0,l3);
    const score=-pen;
    if(score<bestD){ bestD=score; bestT=tr; }
    if(score===0) break;
  }
  return bestT;
}

/* ===================================================================
   5. THREE.JS SCENE
   =================================================================== */
const C = {
  skin:0x16242F, skinEdge:0x24404F, node:0x1B8E96, nodeArm:0x3FD2C7,
  nodeHit:0xE8C34E, mesh:0x21414F, meshHot:0x3FD2C7, src:0xFF3B5C, fix:0xE8C34E
};

const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({canvas:document.getElementById("gl"),antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x060B10, 26, 62);
const camera = new THREE.PerspectiveCamera(38,1,0.1,200);

scene.add(new THREE.AmbientLight(0x2a4457, 1.05));
const dl = new THREE.DirectionalLight(0x9ec8dd, 0.85); dl.position.set(6,12,8); scene.add(dl);
const rim = new THREE.DirectionalLight(0x2f7d92, 0.6); rim.position.set(-9,-4,-7); scene.add(rim);

const air = new THREE.Group(); scene.add(air);

// --- fuselage shell
(function(){
  const NA=90, NR=34, pos=[], idx=[];
  for(let i=0;i<=NA;i++){ const t=i/NA;
    for(let j=0;j<=NR;j++){ const th=j/NR*Math.PI*2; const p=fusePt(t,th,0); pos.push(p.x,p.y,p.z); }
  }
  for(let i=0;i<NA;i++) for(let j=0;j<NR;j++){
    const A=i*(NR+1)+j, B=A+1, Cc=A+NR+1, D=Cc+1;
    idx.push(A,Cc,B, B,Cc,D);
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.Float32BufferAttribute(pos,3));
  g.setIndex(idx); g.computeVertexNormals();
  air.add(new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:C.skin})));
})();

// --- slab builder for aerofoil panels
function slab(q, th, axis){
  const c=[q.rLE,q.rTE,q.tTE,q.tLE], pos=[], idx=[];
  const off = axis==="y" ? V(0,th,0) : V(0,0,th);
  c.forEach(p=>pos.push(p.x+off.x,p.y+off.y,p.z+off.z));
  c.forEach(p=>pos.push(p.x-off.x,p.y-off.y,p.z-off.z));
  idx.push(0,1,2, 0,2,3, 4,6,5, 4,7,6);
  for(let i=0;i<4;i++){ const j=(i+1)%4; idx.push(i,4+i,4+j, i,4+j,j); }
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.Float32BufferAttribute(pos,3));
  g.setIndex(idx); g.computeVertexNormals();
  // DoubleSide: WING_L / HS_L are mirrored (z-negated), which reverses their
  // winding and turns the box inside-out. Front-face culling would then make
  // the underside vanish and reveal the top-mounted nodes from below. Rendering
  // both faces keeps every slab solid regardless of winding.
  return new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:C.skin,side:THREE.DoubleSide}));
}
air.add(slab(WING_R,0.055,"y"), slab(WING_L,0.055,"y"));
air.add(slab(HS_R,0.04,"y"), slab(HS_L,0.04,"y"), slab(FIN,0.055,"z"));

// --- nacelles + pylons
[3.0,-3.0].forEach(z=>{
  const g=new THREE.CylinderGeometry(0.5,0.42,2.5,20,1,true);
  const m=new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:C.skin,side:THREE.DoubleSide}));
  m.rotation.z=Math.PI/2; m.position.set(0.55,-1.02,z); air.add(m);
  const p=new THREE.Mesh(new THREE.BoxGeometry(1.5,0.55,0.11),
    new THREE.MeshLambertMaterial({color:C.skin}));
  p.position.set(-0.15,-0.72,z); air.add(p);
});

// --- triangular sensor mesh (edges)
const edgeSet=new Set(), ePos=[];
tris.forEach(tr=>{
  const [A,B,Cc]=tr.v;
  [[A,B],[B,Cc],[Cc,A]].forEach(([p,q])=>{
    const k=Math.min(p,q)+"-"+Math.max(p,q);
    if(edgeSet.has(k)) return; edgeSet.add(k);
    const n1=nodes[p], n2=nodes[q];
    // subdivide so barrel edges hug the skin
    const S=6;
    for(let s=0;s<S;s++){
      const u1=s/S, u2=(s+1)/S;
      const P1=lerpOnPanel(n1,n2,u1), P2=lerpOnPanel(n1,n2,u2);
      ePos.push(P1.x,P1.y,P1.z,P2.x,P2.y,P2.z);
    }
  });
});
function lerpOnPanel(n1,n2,u,off){
  const p=n1.panel;
  if(p!==n2.panel) return new THREE.Vector3().lerpVectors(n1.pos,n2.pos,u);
  const a=n1.a+(n2.a-n1.a)*u;
  const b=p.bWrap ? n1.b+wrapPi(n2.b-n1.b)*u : n1.b+(n2.b-n1.b)*u;
  return p.point(a,b,off===undefined?0.014:off);
}
const meshEdges=new THREE.LineSegments(
  new THREE.BufferGeometry().setAttribute("position",new THREE.Float32BufferAttribute(ePos,3)),
  new THREE.LineBasicMaterial({color:C.mesh,transparent:true,opacity:0.85}));
air.add(meshEdges);

// --- mesh fill (toggle)
const fillPos=[];
tris.forEach(tr=>tr.v.forEach(i=>{const p=nodes[i].panel.point(nodes[i].a,nodes[i].b,0.010);fillPos.push(p.x,p.y,p.z);}));
const meshFill=new THREE.Mesh(
  new THREE.BufferGeometry().setAttribute("position",new THREE.Float32BufferAttribute(fillPos,3)),
  new THREE.MeshBasicMaterial({color:0x1B8E96,transparent:true,opacity:0.07,side:THREE.DoubleSide,depthWrite:false}));
meshFill.visible=false; air.add(meshFill);

// --- nodes
const nodeGeo=new THREE.SphereGeometry(0.075,10,8);
const nodeMeshes=nodes.map(n=>{
  const m=new THREE.Mesh(nodeGeo,new THREE.MeshBasicMaterial({color:C.node}));
  m.position.copy(n.pos); m.userData.n=n; air.add(m); return m;
});

// --- highlighted cell + markers + rings
// Tessellate the cell so it follows curved panels (barrel) instead of a flat
// chord that sinks under the fuselage skin. CELL_SUB fill tris per edge,
// CELL_ESEG outline segments per edge.
const CELL_SUB=5, CELL_ESEG=6;
const cellFill=new THREE.Mesh(
  new THREE.BufferGeometry().setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(CELL_SUB*CELL_SUB*9),3)),
  new THREE.MeshBasicMaterial({color:0xE8C34E,transparent:true,opacity:0.20,side:THREE.DoubleSide,depthWrite:false}));
cellFill.visible=false; air.add(cellFill);
const cellEdge=new THREE.Line(
  new THREE.BufferGeometry().setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array((CELL_ESEG*3+1)*3),3)),
  new THREE.LineBasicMaterial({color:0xE8C34E}));
cellEdge.visible=false; air.add(cellEdge);

function ringLine(color,op){
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(3*97),3));
  const l=new THREE.Line(g,new THREE.LineBasicMaterial({color,transparent:true,opacity:op}));
  l.visible=false; air.add(l); return l;
}
const frontRing = ringLine(0xFF3B5C,0.9);
const backRings = [ringLine(0xE8C34E,0.75),ringLine(0xE8C34E,0.6),ringLine(0xE8C34E,0.45)];

function drawRing(line,panel,a,b,rad){
  const arr=line.geometry.attributes.position.array;
  for(let k=0;k<=96;k++){
    const ph=k/96*Math.PI*2;
    const [aa,bb]=panel.step(a,b,rad*Math.cos(ph),rad*Math.sin(ph));
    const A=Math.min(panel.aMax+0.10,Math.max(panel.aMin-0.10,aa));
    const B=panel.bWrap?bb:Math.min(1.02,Math.max(-0.02,bb));
    const p=panel.point(A,B,0.03);
    arr[k*3]=p.x; arr[k*3+1]=p.y; arr[k*3+2]=p.z;
  }
  line.geometry.attributes.position.needsUpdate=true;
  line.visible=true;
}

function marker(color,size,ring){
  const g=new THREE.Group();
  const mat=new THREE.LineBasicMaterial({color});
  const mk=(pts)=>{const gg=new THREE.BufferGeometry().setFromPoints(pts);return new THREE.Line(gg,mat);};
  g.add(mk([V(-size,0,0),V(size,0,0)]),mk([V(0,-size,0),V(0,size,0)]),mk([V(0,0,-size),V(0,0,size)]));
  if(ring){
    const pts=[]; for(let k=0;k<=48;k++){const t=k/48*Math.PI*2;pts.push(V(Math.cos(t)*size*0.8,Math.sin(t)*size*0.8,0));}
    g.add(mk(pts));
  }
  g.visible=false; air.add(g); return g;
}
const srcMark=marker(0xFF3B5C,0.34,true);
const fixMark=marker(0xE8C34E,0.26,false);

/* --- camera control ---------------------------------------------- */
let camT=-0.75, camP=1.16, camD=27;
const target=new THREE.Vector3(0,0.1,0);
let tgtT=camT,tgtP=camP,tgtD=camD;
function applyCam(){
  camera.position.set(
    target.x+camD*Math.sin(camP)*Math.cos(camT),
    target.y+camD*Math.cos(camP),
    target.z+camD*Math.sin(camP)*Math.sin(camT));
  camera.lookAt(target);
}
let dragging=false,px=0,py=0;
const gl=renderer.domElement;
gl.addEventListener("pointerdown",e=>{dragging=true;px=e.clientX;py=e.clientY;gl.setPointerCapture(e.pointerId);});
gl.addEventListener("pointerup",e=>{dragging=false;});
gl.addEventListener("pointermove",e=>{
  if(dragging){
    tgtT = camT -= (e.clientX-px)*0.008;
    tgtP = camP = Math.max(0.16,Math.min(Math.PI-0.16, camP - (e.clientY-py)*0.006));
    px=e.clientX; py=e.clientY; applyCam();
  } else hover(e);
});
gl.addEventListener("pointerleave",()=>{tip.style.opacity=0;});
gl.addEventListener("wheel",e=>{e.preventDefault();
  tgtD = camD = Math.max(11,Math.min(52,camD*(1+Math.sign(e.deltaY)*0.09))); applyCam();},{passive:false});

const VIEWS={iso:[-0.75,1.16,27],top:[-Math.PI/2,0.06,30],side:[-Math.PI/2,1.5708,28],aft:[0.02,1.35,26]};
document.querySelectorAll("[data-view]").forEach(b=>b.addEventListener("click",()=>{
  const v=VIEWS[b.dataset.view]; tgtT=v[0];tgtP=v[1];tgtD=v[2];
}));
function slewTo(pos){
  tgtT = Math.atan2(pos.z,pos.x) + 0.55;
  tgtP = Math.max(0.35,Math.min(2.6, Math.acos(Math.max(-1,Math.min(1,pos.y/Math.max(0.001,pos.length()))))*0.55+0.75));
  tgtD = 20;
}

/* --- hover ------------------------------------------------------- */
const ray=new THREE.Raycaster(); ray.params.Points={threshold:0.2};
const tip=document.getElementById("tip");
const mouse=new THREE.Vector2();
function hover(e){
  const r=gl.getBoundingClientRect();
  mouse.x=((e.clientX-r.left)/r.width)*2-1;
  mouse.y=-((e.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(mouse,camera);
  const hit=ray.intersectObjects(nodeMeshes,false)[0];
  if(hit){
    const n=hit.object.userData.n;
    const st=lastHitMap[n.i];
    tip.innerHTML="<b>"+n.id+"</b> · "+n.zone+(st?"<br>"+st.amp.toFixed(1)+" dB · +"+(st.dt).toFixed(0)+" µs":"<br>quiet");
    tip.style.left=(e.clientX-r.left)+"px"; tip.style.top=(e.clientY-r.top)+"px";
    tip.style.opacity=1;
  } else tip.style.opacity=0;
}

/* --- resize ------------------------------------------------------ */
function resize(){
  const w=stage.clientWidth,h=stage.clientHeight;
  if(!w||!h) return;
  renderer.setSize(w,h,false);
  camera.aspect=w/h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize(); applyCam();

/* ===================================================================
   6. EVENT SIMULATION
   =================================================================== */
let lastHitMap={}, current=null, events=[], evCount=0;
let anim=null, autoTimer=null;

function makeEvent(forced){
  const seed=(Math.random()*1e9)|0, rnd=mulberry(seed);
  const roll=rnd();
  let panel, a, b;
  if(forced){
    panel = forced.panel;
    a = Math.min(panel.aMax, Math.max(panel.aMin, forced.a));
    b = panel.bWrap ? wrapPi(forced.b) : Math.min(panel.bMax, Math.max(panel.bMin, forced.b));
  } else {
    panel = roll<0.60 ? PANELS.barrel : (roll<0.80 ? PANELS.wingR : PANELS.wingL);
    a = panel.aMin + rnd()*(panel.aMax-panel.aMin);
    b = panel.bWrap ? wrapPi(rnd()*2*Math.PI-Math.PI)
                    : panel.bMin + rnd()*(panel.bMax-panel.bMin);
  }
  const fPeak = [38,105,195,300][Math.floor(rnd()*4)] + (rnd()-0.5)*36;
  const A0 = 84 + rnd()*18;

  const hits=[];
  nodes.forEach(n=>{
    if(n.panel!==panel) return;
    const d=panel.dist(a,b,n.a,n.b);
    const amp=A0-ATTEN*d+gauss(rnd)*1.4;
    if(amp<thresholdDb) return;
    const t=d/waveV + gauss(rnd)*jitter*1e-6;
    hits.push({n:n.i,d,amp,t});
  });
  hits.sort((x,y)=>x.t-y.t);
  if(hits.length<3) return null;
  const used=hits.slice(0,Math.min(8,hits.length));

  // Cross-panel responders: the wave carries across the wing/body joint, so
  // sensors on the adjacent structure near the joint also register the hit.
  // They light up for situational awareness but are NOT fed to the TDOA solve,
  // whose geodesic metric is only valid within a single panel.
  const srcPos = panel.point(a,b,0);
  const responders=[];
  const couple=(targetPanel,joint)=>{
    const legIn = srcPos.distanceTo(joint);         // source -> joint (world units)
    nodes.forEach(n=>{
      if(n.panel!==targetPanel) return;
      const path = legIn + joint.distanceTo(n.pos); // source -> joint -> node (world units)
      if(path > COUPLE_REACH) return;               // only sensors near the joint respond
      const d = path*MPU;                           // metres
      const amp = A0 - ATTEN*d - JOINT_LOSS + gauss(rnd)*1.4;
      if(amp<thresholdDb) return;
      const t = d/waveV + gauss(rnd)*jitter*1e-6;
      responders.push({n:n.i,d,amp,t});
    });
  };
  if(panel===PANELS.barrel){ couple(PANELS.wingR,WING_JOINT.wingR); couple(PANELS.wingL,WING_JOINT.wingL); }
  else if(panel===PANELS.wingR){ couple(PANELS.barrel,WING_JOINT.wingR); }
  else if(panel===PANELS.wingL){ couple(PANELS.barrel,WING_JOINT.wingL); }

  const fix=solve(panel,used);
  const cell=cellFor(panel,fix.a,fix.b);
  const err=panel.dist(a,b,fix.a,fix.b);

  const w=synthBurst(fPeak,26,seed);
  const sp=spectrum(w);
  let pk=0,pki=0; for(let k=2;k<sp.length;k++) if(sp[k]>pk){pk=sp[k];pki=k;}
  const fMeas=pki*FS_SAMP/NSAMP/1000;
  const cls=classify(fMeas);

  const t0=used[0].t;
  used.forEach(h=>h.dt=(h.t-t0)*1e6);

  evCount++;
  return {
    idx:evCount, id:"AE-"+String(2100+evCount),
    stamp:new Date(), panel, truth:{a,b}, fix, cell, err, hits:used, allHits:hits, responders,
    wave:w, spec:sp, fPeak:fMeas, cls, A0,
    conf: Math.max(0.42, Math.min(0.99, 1 - fix.rms*1e6/60 - err/8))
  };
}

/* ===================================================================
   7. PRESENTATION
   =================================================================== */
const $=id=>document.getElementById(id);
const fmt=(x,d)=>x.toFixed(d===undefined?1:d);
const TRI_COL=["#FF6B3D","#E8C34E","#3FD2C7"];

function stationOf(panel,a,b){
  if(panel.bWrap){
    const fs=Math.round(a*FUSE_LEN*MPU*IN_PER_M);
    const str=1+Math.round(Math.abs(b)/Math.PI*24);
    return "FS "+fs+" · STR "+str+(b>=0?"R":"L");
  }
  const ws=Math.round((0.92+a*(7.2-0.92))*MPU*IN_PER_M);
  const pc=Math.round(b*100);
  return panel.side+"H WS "+ws+" · "+pc+"% chord";
}

function present(ev){
  current=ev;
  $("evEmpty").hidden=true; $("evPanel").hidden=false;

  $("evCls").textContent=ev.cls.name;
  $("evCls").style.color = ev.cls.sev===2?"var(--a6)":(ev.cls.sev===1?"var(--a5)":"var(--a4)");
  $("evConf").textContent="peak "+fmt(ev.fPeak,0)+" kHz · "+(ev.allHits.length+(ev.responders?ev.responders.length:0))+" nodes hit · confidence "+fmt(ev.conf*100,0)+"%";
  const act=$("evAct"); act.textContent=ev.cls.act; act.className="act"+(ev.cls.sev===2?" sev":"");

  $("evId").textContent=ev.id+" · "+ev.stamp.toISOString().substr(11,8)+"Z";
  $("fxZone").textContent=ev.panel.bWrap?zoneOf(ev.fix.a):ev.panel.label;
  $("fxStation").textContent=stationOf(ev.panel,ev.fix.a,ev.fix.b);
  $("fxCell").textContent=ev.cell?ev.cell.v.map(i=>nodes[i].id.split("·")[0]).join(" / "):"—";
  $("fxRes").textContent=fmt(ev.fix.rms*1e6,1)+" µs";
  $("fxEll").textContent="± "+fmt(ev.fix.rms*waveV*1000*1.96,0)+" mm";
  $("fxErr").textContent=fmt(ev.err*1000,0)+" mm from source";

  const triNodes = ev.cell ? ev.cell.v : ev.hits.slice(0,3).map(h=>h.n);
  const tb=$("triBody"); tb.innerHTML="";
  triNodes.forEach((ni,k)=>{
    const h=ev.hits.find(x=>x.n===ni) || ev.allHits.find(x=>x.n===ni);
    const tr=document.createElement("tr");
    tr.innerHTML="<td><i class='vx' style='background:"+TRI_COL[k]+"'></i>"+nodes[ni].id+"</td>"+
      "<td>"+(h?fmt(h.d,2)+" m":"—")+"</td>"+
      "<td>"+(h?fmt(h.t*1e6,0)+" µs":"no hit")+"</td>"+
      "<td>"+(h&&h.dt!==undefined?"+"+fmt(h.dt,0):"—")+"</td>"+
      "<td>"+(h?fmt(h.amp,1):"—")+"</td>";
    tb.appendChild(tr);
  });

  $("wfNode").textContent=nodes[ev.hits[0].n].id;
  $("wfMeta").textContent="rise 4 µs · "+fmt(ev.hits[0].amp,1)+" dB";
  $("spMeta").textContent="peak "+fmt(ev.fPeak,0)+" kHz → "+ev.cls.name.toLowerCase();
  drawWave(ev); drawSpec(ev);

  lastHitMap={};
  ev.allHits.forEach(h=>lastHitMap[h.n]={amp:h.amp,dt:(h.t-ev.hits[0].t)*1e6});
  (ev.responders||[]).forEach(h=>lastHitMap[h.n]={amp:h.amp,dt:(h.t-ev.hits[0].t)*1e6});
  paintNodes(ev,triNodes);
  paintCell(ev,triNodes);
  $("hOpen").textContent=events.length;
  $("hOpen").className="v hot";
  $("hState").innerHTML='<i class="pulse"></i>Event held';
  $("hState").className="v hot";
  drawRuler();
  renderLog();
  runAnimation(ev,triNodes);
}

function paintNodes(ev,triNodes){
  nodeMeshes.forEach((m,i)=>{
    const h=lastHitMap[i];
    let col=C.node, sc=1;
    if(h){ col=C.nodeHit; sc=1.35; }
    if(triNodes.indexOf(i)>=0){ col=0xFF6B3D; sc=1.9; }
    m.material.color.setHex(col); m.scale.setScalar(sc);
  });
  if(triNodes.length===3) triNodes.forEach((n,k)=>nodeMeshes[n].material.color.set(TRI_COL[k]));
}
function paintCell(ev,triNodes){
  if(triNodes.length!==3){cellFill.visible=false;cellEdge.visible=false;return;}
  const na=nodes[triNodes[0]], nb=nodes[triNodes[1]], nc=nodes[triNodes[2]];
  const P=na.panel;
  // barycentric (u,v,w) over the three nodes -> curved surface point, so a
  // barrel cell hugs the skin instead of cutting a flat chord beneath it.
  const surf=(u,v,off)=>{
    const w=1-u-v;
    const a=u*na.a+v*nb.a+w*nc.a;
    const b=P.bWrap ? na.b + v*wrapPi(nb.b-na.b) + w*wrapPi(nc.b-na.b)
                    : u*na.b+v*nb.b+w*nc.b;
    return P.point(a,b,off);
  };
  // fill — tessellated triangle following the surface
  const fa=cellFill.geometry.attributes.position.array; let fi=0;
  const put=p=>{fa[fi++]=p.x;fa[fi++]=p.y;fa[fi++]=p.z;};
  const N=CELL_SUB;
  for(let i=0;i<N;i++) for(let j=0;j<N-i;j++){
    put(surf(i/N,j/N,0.022)); put(surf((i+1)/N,j/N,0.022)); put(surf(i/N,(j+1)/N,0.022));
    if(i+j<N-1){ put(surf((i+1)/N,j/N,0.022)); put(surf((i+1)/N,(j+1)/N,0.022)); put(surf(i/N,(j+1)/N,0.022)); }
  }
  cellFill.geometry.attributes.position.needsUpdate=true;
  // outline — each edge subdivided along the surface, then closed
  const ea=cellEdge.geometry.attributes.position.array; let ei=0;
  [[na,nb],[nb,nc],[nc,na]].forEach(([p,q])=>{
    for(let s=0;s<CELL_ESEG;s++){ const pt=lerpOnPanel(p,q,s/CELL_ESEG,0.03); ea[ei++]=pt.x;ea[ei++]=pt.y;ea[ei++]=pt.z; }
  });
  ea[ei++]=ea[0]; ea[ei++]=ea[1]; ea[ei++]=ea[2];   // close the loop
  cellEdge.geometry.attributes.position.needsUpdate=true;
  cellFill.visible=true; cellEdge.visible=true;
}

/* --- animation: forward wavefront, then back-propagation --------- */
function runAnimation(ev,triNodes){
  const P=ev.panel;
  const srcP=P.point(ev.truth.a,ev.truth.b,0.05);
  const fixP=P.point(ev.fix.a,ev.fix.b,0.05);
  srcMark.position.copy(srcP); fixMark.position.copy(fixP);
  slewTo(srcP);
  const phase=$("phase");

  if(anim) cancelAnimationFrame(anim.raf);
  frontRing.material.opacity=0.9;
  backRings.forEach((r,k)=>{ r.material.opacity=0.75-k*0.12; r.visible=false; });

  if(REDUCED){
    srcMark.visible=true; fixMark.visible=true;
    frontRing.visible=false;
    phase.textContent="Fix held · "+stationOf(P,ev.fix.a,ev.fix.b);
    phase.classList.add("on"); return;
  }
  const dMax=Math.max(...ev.hits.map(h=>h.d))*1.15;
  const trio=triNodes.map(i=>ev.hits.find(x=>x.n===i)||ev.allHits.find(x=>x.n===i)).filter(Boolean);
  // back-propagation sweeps the clock from the first arrival back to the
  // estimated emission instant; radii v(t_i - tau) meet at the fix.
  const tFirst=Math.min(...trio.map(h=>h.t));
  const start=performance.now();
  srcMark.visible=true; fixMark.visible=false;

  anim={raf:0};
  (function step(now){
    const el=(now-start)/1000;
    if(el<1.15){                       // phase 1 — outgoing wavefront
      phase.textContent="Wavefront · "+fmt(el/1.15*100,0)+"%";
      phase.classList.add("on");
      drawRing(frontRing,P,ev.truth.a,ev.truth.b,(el/1.15)*dMax);
      frontRing.material.opacity=0.9*(1-el/1.15*0.7);
      backRings.forEach(r=>r.visible=false);
    } else if(el<2.6){                 // phase 2 — back-propagation to the fix
      const u=(el-1.15)/1.45;
      phase.textContent="Back-propagation · Δt inversion";
      frontRing.visible=false;
      const tau = tFirst - (tFirst-ev.fix.t0)*u;
      trio.forEach((h,k)=>{
        const rad=(h.t-tau)*waveV;
        if(rad>0.001) drawRing(backRings[k],P,nodes[h.n].a,nodes[h.n].b,rad);
        else backRings[k].visible=false;
        backRings[k].material.color.set(TRI_COL[k]);
      });
      fixMark.visible=u>0.9;
    } else {                           // settled
      phase.textContent="Fix held · "+stationOf(P,ev.fix.a,ev.fix.b);
      backRings.forEach((r,k)=>r.material.opacity=Math.max(0,(0.75-k*0.12)*(1-(el-2.6))));
      if(el>3.6){ backRings.forEach(r=>r.visible=false); phase.classList.remove("on"); return; }
      fixMark.visible=true;
    }
    anim.raf=requestAnimationFrame(step);
  })(start);
}

/* --- charts ------------------------------------------------------ */
function ctxFor(cv){
  const r=cv.getBoundingClientRect(), dpr=Math.min(devicePixelRatio,2);
  cv.width=Math.max(1,r.width*dpr); cv.height=Math.max(1,r.height*dpr);
  const c=cv.getContext("2d"); c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,r.width,r.height);
  return {c,w:r.width,h:r.height};
}
function drawWave(ev){
  const {c,w,h}=ctxFor($("wave")); if(!w) return;
  c.strokeStyle="#1B2C39"; c.lineWidth=1;
  c.beginPath(); c.moveTo(0,h/2); c.lineTo(w,h/2); c.stroke();
  c.strokeStyle="#3FD2C7"; c.lineWidth=1; c.beginPath();
  for(let i=0;i<ev.wave.length;i++){
    const x=i/(ev.wave.length-1)*w, y=h/2-ev.wave[i]*(h/2-4);
    i?c.lineTo(x,y):c.moveTo(x,y);
  }
  c.stroke();
  const thrY=h/2-(Math.pow(10,(thresholdDb-ev.hits[0].amp)/20))*(h/2-4);
  c.strokeStyle="#FF6B3D"; c.setLineDash([3,3]); c.beginPath();
  c.moveTo(0,thrY); c.lineTo(w,thrY); c.stroke(); c.setLineDash([]);
}
function drawSpec(ev){
  const {c,w,h}=ctxFor($("spec")); if(!w) return;
  const kHzMax=400, binHz=FS_SAMP/NSAMP/1000;
  const bands=[[0,70,"#123F6B"],[70,140,"#1B8E96"],[140,260,"#7a5a1e"],[260,400,"#6b2740"]];
  bands.forEach(([lo,hi,col])=>{
    c.fillStyle=col; c.globalAlpha=0.22;
    c.fillRect(lo/kHzMax*w,0,(hi-lo)/kHzMax*w,h); c.globalAlpha=1;
    c.strokeStyle="#1B2C39"; c.beginPath();
    c.moveTo(hi/kHzMax*w,0); c.lineTo(hi/kHzMax*w,h); c.stroke();
  });
  c.fillStyle="#E8C34E";
  for(let k=1;k<ev.spec.length;k++){
    const f=k*binHz; if(f>kHzMax) break;
    const x=f/kHzMax*w, bw=Math.max(1,binHz/kHzMax*w-0.6);
    c.fillRect(x,h-ev.spec[k]*(h-10),bw,ev.spec[k]*(h-10));
  }
  const px=ev.fPeak/kHzMax*w;
  c.strokeStyle="#FF3B5C"; c.lineWidth=1;
  c.beginPath(); c.moveTo(px,0); c.lineTo(px,h); c.stroke();
  c.fillStyle="#FF3B5C"; c.font="10px 'IBM Plex Mono',monospace";
  c.fillText(fmt(ev.fPeak,0)+" kHz", Math.min(w-56,px+4), 11);
}

/* --- fuselage station ruler -------------------------------------- */
function drawRuler(){
  const cv=$("ruler"), {c,w,h}=ctxFor(cv); if(!w) return;
  const FSmax=Math.round(FUSE_LEN*MPU*IN_PER_M);
  const pad=14, span=w-pad*2;
  c.strokeStyle="#274152"; c.lineWidth=1;
  c.beginPath(); c.moveTo(pad,h-13); c.lineTo(w-pad,h-13); c.stroke();
  c.font="9px 'IBM Plex Mono',monospace"; c.textAlign="center";
  for(let fs=0; fs<=FSmax; fs+=100){
    const x=pad+fs/FSmax*span, major=fs%400===0;
    c.strokeStyle=major?"#3d5f73":"#274152";
    c.beginPath(); c.moveTo(x,h-13); c.lineTo(x,h-13-(major?7:4)); c.stroke();
    if(major){ c.fillStyle="#4C6474"; c.fillText("FS "+fs,x,h-3); }
  }
  // sensor rings
  for(let i=0;i<RINGS;i++){
    const a=0.20+(0.72-0.20)*(i/(RINGS-1));
    const x=pad+(a*FUSE_LEN*MPU*IN_PER_M)/FSmax*span;
    c.fillStyle="#1B8E96"; c.fillRect(x-1,h-13,2,-11);
  }
  if(current && current.panel.bWrap){
    const x=pad+(current.fix.a*FUSE_LEN*MPU*IN_PER_M)/FSmax*span;
    c.fillStyle="#FF3B5C"; c.beginPath();
    c.moveTo(x,h-13); c.lineTo(x-5,h-24); c.lineTo(x+5,h-24); c.closePath(); c.fill();
    c.fillStyle="#FF3B5C"; c.textAlign=x>w-90?"right":"left";
    c.fillText(stationOf(current.panel,current.fix.a,current.fix.b), x>w-90?x-8:x+8, h-26);
  }
}

/* --- zones + log ------------------------------------------------- */
function renderZones(){
  const groups={};
  nodes.forEach(n=>{ (groups[n.zone]=groups[n.zone]||[]).push(n); });
  const el=$("zones"); el.innerHTML="";
  Object.keys(groups).forEach(z=>{
    const list=groups[z];
    const hitN=list.filter(n=>lastHitMap[n.i]).length;
    const d=document.createElement("div");
    d.className="zone"+(hitN?" hit":"");
    d.innerHTML="<span class='n'>"+z+"</span><span class='c'>"+list.length+" nodes</span>"+
      "<span class='bar'><i style='width:"+(hitN?Math.round(hitN/list.length*100):100)+"%'></i></span>";
    el.appendChild(d);
  });
  $("zoneTot").textContent=nodes.length+" nodes";
}
function renderLog(){
  const el=$("log");
  if(!events.length){ el.innerHTML="<div class='empty'>No qualifying hits since the last data download.<br>Trigger a test event to exercise the array.</div>"; return; }
  el.innerHTML="";
  events.slice().reverse().forEach(ev=>{
    const b=document.createElement("button");
    b.className="logrow"; b.setAttribute("aria-current", current&&current.id===ev.id?"true":"false");
    b.innerHTML="<span class='t'>"+ev.stamp.toISOString().substr(11,8)+"</span>"+
      "<span class='d'>"+ev.cls.name+"</span><span class='a'>"+fmt(ev.fPeak,0)+"k</span>";
    b.addEventListener("click",()=>present(ev));
    el.appendChild(b);
  });
  $("logCount").textContent=events.length+" held";
}

/* --- actions ----------------------------------------------------- */
function inject(){
  let ev=null,guard=0;
  while(!ev && guard++<12) ev=makeEvent();
  if(!ev){
    $("phase").textContent="No fix — raise sensitivity";
    $("phase").classList.add("on");
    setTimeout(()=>$("phase").classList.remove("on"),2200);
    return;
  }
  events.push(ev); if(events.length>24) events.shift();
  present(ev); renderZones();
}
$("btnInject").addEventListener("click",inject);

/* --- live feed: real taps from final_demo.py's serial/triangulation --- */
function tapToBarrel(nx,ny){
  // nx/ny are normalized [0,1] sheet coordinates from the 2-mic triangulation.
  // nx runs fwd->aft along the barrel; ny (mic setup can only resolve the
  // sheet's vertical center today) maps to the fuselage's top/side arc.
  const P=PANELS.barrel;
  const a=P.aMin+Math.min(1,Math.max(0,nx))*(P.aMax-P.aMin);
  const b=wrapPi((Math.min(1,Math.max(0,ny))-0.5)*Math.PI);
  return {a,b};
}
function liveInject(a,b){
  let ev=null,guard=0;
  while(!ev && guard++<12) ev=makeEvent({panel:PANELS.barrel,a,b});
  if(!ev){
    $("phase").textContent="Tap below threshold — ignored";
    $("phase").classList.add("on");
    setTimeout(()=>$("phase").classList.remove("on"),1500);
    return;
  }
  events.push(ev); if(events.length>24) events.shift();
  present(ev); renderZones();
}
const feedEl=document.getElementById("hFeed");
function setFeed(text,ok){ if(!feedEl) return; feedEl.textContent=text; feedEl.className="v "+(ok?"ok":"hot"); }
(function connectFeed(){
  let es;
  try{ es=new EventSource("/stream"); } catch(err){ setFeed("● OFFLINE",false); return; }
  es.addEventListener("status", e=>{
    const d=JSON.parse(e.data);
    setFeed(d.live?"● LIVE SERIAL":"● MOCK DATA", d.live);
  });
  es.addEventListener("tap", e=>{
    const d=JSON.parse(e.data);
    const {a,b}=tapToBarrel(d.nx,d.ny);
    liveInject(a,b);
  });
  es.onerror=()=>setFeed("● NO CONNECTION",false);
})();
$("btnReplay").addEventListener("click",()=>{ if(current) present(current); });
$("btnFill").addEventListener("click",e=>{
  const on=e.currentTarget.getAttribute("aria-pressed")!=="true";
  e.currentTarget.setAttribute("aria-pressed",on); meshFill.visible=on;
});
$("btnAuto").addEventListener("click",e=>{
  const on=e.currentTarget.getAttribute("aria-pressed")!=="true";
  e.currentTarget.setAttribute("aria-pressed",on);
  if(on){ autoTimer=setInterval(inject, 6500); inject(); }
  else { clearInterval(autoTimer); autoTimer=null; }
});
$("thr").addEventListener("input",e=>{ thresholdDb=+e.target.value; $("thrV").textContent=thresholdDb; });
$("noise").addEventListener("input",e=>{ jitter=+e.target.value; $("noiseV").textContent=jitter.toFixed(1); });
$("mode").addEventListener("change",e=>{
  waveV=+e.target.value;
  if(current){ // re-solve the held event with the new assumed speed
    const ev=current, fix=solve(ev.panel,ev.hits);
    ev.fix=fix; ev.cell=cellFor(ev.panel,fix.a,fix.b);
    ev.err=ev.panel.dist(ev.truth.a,ev.truth.b,fix.a,fix.b);
    present(ev);
  }
});
window.addEventListener("resize",()=>{ drawRuler(); if(current){drawWave(current);drawSpec(current);} });

/* --- main loop --------------------------------------------------- */
let t=0;
(function loop(){
  requestAnimationFrame(loop);
  camT += (tgtT-camT)*0.07; camP += (tgtP-camP)*0.07; camD += (tgtD-camD)*0.07;
  applyCam();
  t+=0.016;
  if(!current){
    // idle: a slow listening shimmer across the array
    nodeMeshes.forEach((m,i)=>{
      const ph=(t*0.55 + nodes[i].a*2.2 + nodes[i].b*0.12)%1;
      const k=Math.max(0,1-Math.abs(ph-0.5)*4);
      m.material.color.setHex(k>0.6?C.nodeArm:C.node);
      m.scale.setScalar(1+k*0.35);
    });
  }
  renderer.render(scene,camera);
})();

renderZones(); drawRuler();
setTimeout(()=>{ drawRuler(); resize(); },120);
})();
</script>
</body>
</html>

"""


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    if USE_REAL_HARDWARE:
        print("Connecting to Arduino...")
        try:
            ser = connect_arduino()
            _live_state["live"] = True
            print("Connected. Streaming live taps to the console...")
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Falling back to mock data...")
            ser = MockSerial()
    else:
        print("Running with mock data.")
        print("Set USE_REAL_HARDWARE = True when Arduino arrives.")
        ser = MockSerial()

    threading.Thread(target=serial_loop, args=(ser,), daemon=True).start()

    server = http.server.ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), Handler)
    url = f"http://{HTTP_HOST}:{HTTP_PORT}/"
    print(f"Serving Airframe AE Array console at {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
        server.shutdown()
