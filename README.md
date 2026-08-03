# Airframe AE

Real-time acoustic emission detection and structural impact localization for aircraft.

---

## The Problem

Aircraft are inspected for structural damage on the ground every 100–1,000 flight hours. Cracks can initiate and grow the entire time with no one knowing.

## What It Does

Piezoelectric microphone sensors mounted on a structural panel detect the stress waves produced by an impact event. The system triangulates the source location using amplitude differences across the sensor array and maps it onto a live aircraft structural diagram — flagging high-risk zones in real time.

---

## Hardware

- 2× Adafruit MAX4466 electret microphone amplifier modules
- Arduino Uno Q
- Metal panel (analog for aircraft skin)

## Wiring

| MAX4466 | Arduino |
|---------|---------|
| VCC | 3.3V |
| GND | GND |
| OUT (Mic 1) | A0 |
| OUT (Mic 2) | A1 |

---

## Setup

**Arduino**
1. Open `sensor.ino` in Arduino IDE
2. Select board: Arduino Uno Q
3. Upload
4. Confirm data in Serial Monitor at 115200 baud
5. Close Serial Monitor

**Python**
```bash
pip install pyserial numpy matplotlib
```

Set hardware mode in `final_demo.py`:
```python
USE_REAL_HARDWARE = True   # False for simulation mode
```

Run:
```bash
python final_demo.py
```

---

## How It Works

1. Tap registers on the panel
2. Stress wave reaches each mic at slightly different amplitudes
3. Arduino reads analog values and streams over USB serial
4. Python triangulates position from amplitude weighting
5. Impact mapped onto aircraft diagram with risk classification

**Latency:** ~100ms tap to visualization

---

## Risk Zones

| Zone | Risk Level |
|------|------------|
| Wing root | High |
| Door frame | Medium |
| Fuselage lap joint | Monitor |
| Tail section | Low |

---

## Run Without Hardware

Set `USE_REAL_HARDWARE = False` — the system cycles through simulated impact events automatically.

---

## Background

Built on acoustic emission (AE) principles used in structural health monitoring research. Extends the source localization approach demonstrated by Kral et al. (Wichita State, 2013) into a real-time hardware demonstration.
