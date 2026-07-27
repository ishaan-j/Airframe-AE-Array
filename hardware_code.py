import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from collections import deque
import time

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
THRESHOLD   = 40    # adjust if too sensitive or missing taps

# ═══════════════════════════════════════════════════════
# MOCK SERIAL — used when USE_REAL_HARDWARE = False
# ═══════════════════════════════════════════════════════

class MockSerial:
    """
    Simulates Arduino serial output for testing without hardware.
    Cycles through tap positions across the sheet automatically.
    """
    def __init__(self):
        self.last_event  = time.time()
        self.sequence_idx = 0
        self.in_waiting  = 0

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

    def readline(self):
        current_time = time.time()
        if current_time - self.last_event > 2.5:
            self.last_event   = current_time
            amps = self.tap_sequence[
                self.sequence_idx % len(self.tap_sequence)
            ]
            self.sequence_idx += 1

            # Add noise to make it realistic
            amps = [max(0, a + np.random.randint(-25, 25))
                    for a in amps]

            ts   = int(current_time * 1000)
            line = f"{ts},{amps[0]},{amps[1]}\n"
            self.in_waiting = len(line)
            return line.encode('utf-8')

        self.in_waiting = 0
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

    # Try auto-detect first
    for p in ports:
        desc = p.description.lower()
        dev  = p.device.lower()
        if ('arduino' in desc or
            'ttyacm'  in dev  or
            'ttyusb'  in dev  or
            'usbserial' in dev):
            print(f"Auto-detected Arduino on {p.device}")
            return serial.Serial(p.device, SERIAL_BAUD, timeout=0.05)

    # Manual fallback
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
            line = raw.decode('utf-8').strip()
            if ',' in line:
                parts = line.split(',')
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

    # Clamp to sheet bounds
    x = np.clip(x, 0, SHEET_WIDTH)
    return float(x), float(y)


def map_to_aircraft(sheet_x, sheet_y):
    """
    Maps sheet coordinates (cm) to aircraft plot coordinates.
    Aircraft plot x: 0.5 → 5.5
    Aircraft plot y: -1.0 → 1.0 (fuselage only)
    """
    norm_x = sheet_x / SHEET_WIDTH
    norm_y = sheet_y / SHEET_HEIGHT

    aircraft_x = 0.5 + norm_x * 5.0
    aircraft_y = -0.5 + norm_y * 1.0

    return aircraft_x, aircraft_y


def classify_impact(aircraft_x, aircraft_y):
    """
    Classifies impact zone by structural risk level.
    Based on known high-stress zones in aircraft structures.
    Returns (label, color).
    """
    # Wing root — highest fatigue stress concentration
    if 1.2 < aircraft_x < 2.6 and abs(aircraft_y) < 0.8:
        return "HIGH RISK — Wing Root Zone", "#ff4444"

    # Door frame cutouts — stress concentration at corners
    elif 2.8 < aircraft_x < 3.6 and abs(aircraft_y) > 0.3:
        return "MEDIUM RISK — Door Frame", "#ff9900"

    # Fuselage lap joints — fatigue-prone in pressurized fuselage
    elif 1.5 < aircraft_x < 5.0 and abs(aircraft_y) < 0.3:
        return "MONITOR — Fuselage Lap Joint", "#ffff00"

    # Tail section
    elif aircraft_x > 5.0:
        return "LOW RISK — Tail Section", "#00ff88"

    else:
        return "EVENT DETECTED", "#00ff88"


# ═══════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════

def build_figure():
    """Builds the full figure with both panels."""
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#0d1117')

    plt.suptitle(
        'Real-Time Aircraft Structural Impact Detection System',
        color='white', fontsize=15, fontweight='bold', y=0.98
    )

    ax1 = fig.add_subplot(1, 2, 1)  # sheet panel
    ax2 = fig.add_subplot(1, 2, 2)  # aircraft panel

    # ── Sheet panel ──────────────────────────────────────
    ax1.set_facecolor('#0d1117')
    ax1.set_xlim(-2, SHEET_WIDTH  + 2)
    ax1.set_ylim(-2, SHEET_HEIGHT + 2)
    ax1.set_aspect('equal')
    ax1.set_title('Live Sensor Array  (Metal Sheet)',
                  color='white', fontsize=12, pad=10)
    ax1.tick_params(colors='#555555')
    for spine in ax1.spines.values():
        spine.set_color('#333333')

    # Sheet rectangle
    sheet = patches.Rectangle(
        (0, 0), SHEET_WIDTH, SHEET_HEIGHT,
        linewidth=2, edgecolor='#4a9eff',
        facecolor='#1a3a5c', alpha=0.35
    )
    ax1.add_patch(sheet)

    # Label sheet dimensions
    ax1.text(SHEET_WIDTH/2, -1.5,
             f'{SHEET_WIDTH:.0f} cm',
             color='#555555', ha='center', fontsize=8)
    ax1.text(-1.5, SHEET_HEIGHT/2,
             f'{SHEET_HEIGHT:.0f} cm',
             color='#555555', va='center',
             fontsize=8, rotation=90)

    # Draw mics
    mic_markers = []
    for i, (mx, my) in enumerate(MIC_POSITIONS):
        m = ax1.plot(mx, my, 's',
                     color='#00ff88', markersize=14,
                     zorder=5, markeredgecolor='white',
                     markeredgewidth=0.5)[0]
        ax1.annotate(
            f'MIC {i+1}', (mx, my),
            color='#00ff88', fontsize=9,
            textcoords="offset points", xytext=(6, 6)
        )
        mic_markers.append(m)

    # Signal strength bars (below sheet)
    bar_ax = fig.add_axes([0.08, 0.08, 0.38, 0.06])
    bar_ax.set_facecolor('#0d1117')
    bar_ax.set_xlim(0, 3)
    bar_ax.set_ylim(0, 512)
    bar_ax.set_title('Sensor Amplitude',
                     color='#888888', fontsize=8)
    bar_ax.tick_params(colors='#555555', labelsize=7)
    bar_ax.set_xticks([0.5, 1.5])
    bar_ax.set_xticklabels(['MIC 1', 'MIC 2'], color='#888888')
    for spine in bar_ax.spines.values():
        spine.set_color('#333333')

    bar1 = bar_ax.bar(0.5, 0, width=0.6,
                      color='#4a9eff', alpha=0.8)[0]
    bar2 = bar_ax.bar(1.5, 0, width=0.6,
                      color='#4a9eff', alpha=0.8)[0]

    # ── Aircraft panel ───────────────────────────────────
    ax2.set_facecolor('#0d1117')
    ax2.set_xlim(-0.5, 7.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect('equal')
    ax2.set_title('Aircraft Structural Map',
                  color='white', fontsize=12, pad=10)
    ax2.tick_params(colors='#555555')
    ax2.axis('off')

    # Fuselage body
    fuselage = patches.Ellipse(
        (3, 0), 6, 1.4,
        linewidth=2, edgecolor='#4a9eff',
        facecolor='#1a3a5c', alpha=0.7, zorder=2
    )
    ax2.add_patch(fuselage)

    # Nose cone
    nose = patches.FancyArrow(
        0.0, 0, -0.3, 0,
        width=0.5, head_width=0.5,
        head_length=0.3,
        color='#1a3a5c',
        edgecolor='#4a9eff',
        linewidth=2, zorder=2
    )
    ax2.add_patch(nose)

    # Wings
    left_wing  = np.array([[1.5,-0.5],[2.5,-0.5],
                            [1.0,-3.2],[0.3,-3.0]])
    right_wing = np.array([[1.5, 0.5],[2.5, 0.5],
                            [1.0, 3.2],[0.3, 3.0]])
    ax2.fill(left_wing[:,0],  left_wing[:,1],
             color='#1a3a5c', alpha=0.7,
             edgecolor='#4a9eff', linewidth=2, zorder=2)
    ax2.fill(right_wing[:,0], right_wing[:,1],
             color='#1a3a5c', alpha=0.7,
             edgecolor='#4a9eff', linewidth=2, zorder=2)

    # Vertical tail
    vtail = np.array([[5.5, 0.0],[6.0, 0.0],
                       [6.2, 1.8],[5.5, 1.2]])
    ax2.fill(vtail[:,0], vtail[:,1],
             color='#1a3a5c', alpha=0.7,
             edgecolor='#4a9eff', linewidth=2, zorder=2)

    # Horizontal tail
    htail_l = np.array([[5.5,-0.3],[6.1,-0.3],
                         [6.3,-1.5],[5.7,-1.5]])
    htail_r = np.array([[5.5, 0.3],[6.1, 0.3],
                         [6.3, 1.5],[5.7, 1.5]])
    ax2.fill(htail_l[:,0], htail_l[:,1],
             color='#1a3a5c', alpha=0.7,
             edgecolor='#4a9eff', linewidth=2, zorder=2)
    ax2.fill(htail_r[:,0], htail_r[:,1],
             color='#1a3a5c', alpha=0.7,
             edgecolor='#4a9eff', linewidth=2, zorder=2)

    # Risk zone overlays
    risk_zones = [
        (patches.Ellipse((1.9, 0), 1.4, 0.8,
                         edgecolor='#ff4444',
                         facecolor='#ff4444',
                         alpha=0.15, linewidth=2,
                         linestyle='--', zorder=3),
         (1.9, -1.8), 'Wing Root\n⚠ HIGH RISK', '#ff4444'),

        (patches.Ellipse((3.1,  0.65), 0.9, 0.35,
                         edgecolor='#ff9900',
                         facecolor='#ff9900',
                         alpha=0.15, linewidth=1.5,
                         linestyle='--', zorder=3),
         (3.8, 1.3), 'Door Frame', '#ff9900'),

        (patches.Ellipse((3.1, -0.65), 0.9, 0.35,
                         edgecolor='#ff9900',
                         facecolor='#ff9900',
                         alpha=0.15, linewidth=1.5,
                         linestyle='--', zorder=3),
         (3.8, -1.3), 'Door Frame', '#ff9900'),
    ]

    for rp, pos, label, color in risk_zones:
        ax2.add_patch(rp)
        ax2.text(pos[0], pos[1], label,
                 color=color, fontsize=7,
                 ha='center', alpha=0.85)

    # Dynamic elements — impact marker and status
    plane_marker  = ax2.plot([], [], 'r*',
                             markersize=30, zorder=10)[0]
    status_text   = ax2.text(
        3, -2.8, '', ha='center',
        fontsize=12, fontweight='bold', color='white'
    )
    mode_text = ax2.text(
        3, 3.2,
        '● MOCK DATA' if not USE_REAL_HARDWARE else '● LIVE SENSORS',
        ha='center', fontsize=9,
        color='#ffff00' if not USE_REAL_HARDWARE else '#00ff88'
    )

    # Event log at bottom
    event_log  = deque(maxlen=4)
    log_text   = fig.text(
        0.5, 0.01, 'Waiting for events...',
        ha='center', color='#555555', fontsize=8
    )

    return (fig, ax1, ax2, bar_ax,
            mic_markers, bar1, bar2,
            plane_marker, status_text, mode_text,
            event_log, log_text)


# ═══════════════════════════════════════════════════════
# MAIN DEMO LOOP
# ═══════════════════════════════════════════════════════

def run_demo(ser):
    (fig, ax1, ax2, bar_ax,
     mic_markers, bar1, bar2,
     plane_marker, status_text, mode_text,
     event_log, log_text) = build_figure()

    # Mutable state
    state = {
        'tap_xy'     : None,
        'plane_xy'   : None,
        'wave_start' : None,
        'amplitudes' : [0, 0],
        'event_count': 0,
        'risk_counts': {'high': 0, 'medium': 0, 'low': 0},
    }

    wave_artists = []
    tap_marker   = ax1.plot([], [], 'r*',
                            markersize=22, zorder=10)[0]

    # Running event counter display
    counter_text = ax1.text(
        SHEET_WIDTH/2, SHEET_HEIGHT + 1,
        'Events: 0', ha='center',
        color='#888888', fontsize=9
    )

    def animate(frame):
        # ── Read from serial ──────────────────────────
        ts, amps = read_tap(ser)

        if ts is not None and amps is not None:
            state['amplitudes'] = amps
            result = triangulate_2mic(amps, MIC_POSITIONS)

            if result:
                sx, sy = result
                state['tap_xy']    = (sx, sy)
                state['plane_xy']  = map_to_aircraft(sx, sy)
                state['wave_start'] = frame
                state['event_count'] += 1

                label, color = classify_impact(*state['plane_xy'])

                # Track risk counts
                if 'HIGH'   in label:
                    state['risk_counts']['high']   += 1
                elif 'MEDIUM' in label:
                    state['risk_counts']['medium'] += 1
                else:
                    state['risk_counts']['low']    += 1

                # Add to event log
                event_log.appendleft(
                    f"#{state['event_count']}  "
                    f"Position: {sx:.0f} cm from left  |  "
                    f"{label}  |  "
                    f"Amps: [{amps[0]}, {amps[1]}]"
                )
                log_text.set_text(
                    '  ·  '.join(list(event_log)[:2])
                )
                log_text.set_color(color)

                counter_text.set_text(
                    f"Events: {state['event_count']}   "
                    f"[High: {state['risk_counts']['high']}  "
                    f"Med: {state['risk_counts']['medium']}  "
                    f"Low: {state['risk_counts']['low']}]"
                )

        # ── Update amplitude bars ──────────────────────
        bar1.set_height(state['amplitudes'][0])
        bar2.set_height(state['amplitudes'][1])

        # Color bars by relative amplitude
        a1, a2 = state['amplitudes']
        if a1 > a2:
            bar1.set_color('#ff6b6b')
            bar2.set_color('#4a9eff')
        else:
            bar1.set_color('#4a9eff')
            bar2.set_color('#ff6b6b')

        # ── Draw propagating wave rings ────────────────
        for c in wave_artists:
            try:
                c.remove()
            except Exception:
                pass
        wave_artists.clear()

        if state['wave_start'] is not None and state['tap_xy']:
            tx, ty  = state['tap_xy']
            elapsed = frame - state['wave_start']

            for ring in range(4):
                radius = max(0, elapsed * 1.2 - ring * 10)
                if 0 < radius < SHEET_WIDTH * 2:
                    alpha = max(0, 0.55 - radius/80 - ring*0.1)
                    colors = ['#4a9eff','#00ffff',
                              '#4a9eff','#ffffff']
                    c = plt.Circle(
                        (tx, ty), radius,
                        fill=False,
                        color=colors[ring],
                        alpha=alpha,
                        linewidth=max(0.5, 2.0 - ring*0.4)
                    )
                    ax1.add_patch(c)
                    wave_artists.append(c)

            # Fade tap marker
            fade = max(0.1, 1.0 - elapsed/60)
            tap_marker.set_data([tx], [ty])
            tap_marker.set_alpha(fade)

            # Light up mics as wave reaches them
            for i, (mx, my) in enumerate(MIC_POSITIONS):
                dist = np.sqrt((mx-tx)**2 + (my-ty)**2)
                wave_radius = elapsed * 1.2
                if wave_radius >= dist:
                    mic_markers[i].set_color('red')
                    mic_markers[i].set_markersize(18)
                else:
                    mic_markers[i].set_color('#00ff88')
                    mic_markers[i].set_markersize(14)

        # ── Update aircraft marker ─────────────────────
        if state['plane_xy']:
            px, py = state['plane_xy']
            plane_marker.set_data([px], [py])

            label, color = classify_impact(px, py)
            status_text.set_text(f'⚠  {label}')
            status_text.set_color(color)
            plane_marker.set_color(color)

        return (wave_artists +
                [tap_marker, plane_marker,
                 status_text, log_text,
                 counter_text, bar1, bar2] +
                mic_markers)

    ani = animation.FuncAnimation(
        fig, animate,
        interval=50,
        blit=False,
        cache_frame_data=False
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    if USE_REAL_HARDWARE:
        print("Connecting to Arduino...")
        try:
            ser = connect_arduino()
            print("Connected. Starting live demo...")
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Falling back to mock data...")
            ser = MockSerial()
    else:
        print("Running with mock data.")
        print("Set USE_REAL_HARDWARE = True when Arduino arrives.")
        ser = MockSerial()

    run_demo(ser)