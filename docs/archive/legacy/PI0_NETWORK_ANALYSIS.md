# Pi0 Network Analysis - LTE/VPN Verbindung

**Kritische Information:** Training Server ↔ Inference PC über LTE VPN (6 MBit)  
**Erstellt:** 07.01.2025  
**Zweck:** Architektur-Bewertung für Remote vs. Local Inference

---

## 📊 Bandbreiten-Analyse

### Datenvolumen für Remote Inference

**Szenario: Policy Server auf Training Server (Remote)**

#### Pro Inference-Zyklus (50Hz):

**Observations → Server:**
```
4 Cameras × 480×640×3 bytes = 3.686.400 bytes ≈ 3.5 MB
State (14 floats × 4 bytes) = 56 bytes
Prompt (tokenized, ~100 bytes) = 100 bytes

Total pro Frame: ~3.5 MB
Bei 50Hz: 3.5 MB × 50 = 175 MB/s
```

**Actions ← Server:**
```
Actions (14 floats × 50 steps × 4 bytes) = 2.800 bytes ≈ 2.8 KB
Minimal im Vergleich zu Images
```

**Total Bandwidth:**
- Ohne Compression: ~175 MB/s = 1.400 Mbit/s
- Mit JPEG Compression (80% Reduktion): ~35 MB/s = 280 Mbit/s
- Mit JPEG + Downsample 320×240: ~9 MB/s = 72 Mbit/s

**Deine Verbindung: 6 Mbit/s = 0.75 MB/s**

### ⚠️ FAZIT: Remote Inference NICHT MÖGLICH

**Rechnung:**
```
Benötigt (minimal): 72 Mbit/s (mit starker Compression)
Verfügbar: 6 Mbit/s
Verhältnis: 12x zu wenig Bandbreite!
```

**Selbst mit extremen Optimierungen:**
- 1 Camera statt 4: 18 Mbit/s (immer noch 3x zu viel)
- 10Hz statt 50Hz: 14 Mbit/s (immer noch 2.3x zu viel)
- Kombination: 3.5 Mbit/s (knapp, aber Latency!)

---

## 💡 EMPFEHLUNG: Local Inference (Alles auf Inference PC)

### Warum Local die bessere Lösung ist

**1. Bandbreite:**
- ✅ Kein Netzwerk-Bottleneck
- ✅ Volle 50Hz Control möglich
- ✅ Alle 4 Cameras nutzbar

**2. Latency:**
- ✅ Kein VPN Overhead (~20-50ms)
- ✅ Kein LTE Jitter
- ✅ Deterministisch <50ms total latency

**3. Zuverlässigkeit:**
- ✅ Keine Abhängigkeit von Mobilfunk
- ✅ Keine VPN Disconnects
- ✅ Lokale Fehlerbehandlung

**4. Feasibility:**
- ✅ RTX 4080 16GB ist ausreichend (braucht nur ~8GB für Inference)
- ✅ Checkpoint Transfer einmalig (~2-5 GB)
- ✅ Einfacherer Setup

---

## 🏗️ NEUE EMPFOHLENE ARCHITEKTUR

### All-in-One auf Inference PC

```
┌─────────────────────────────────────────────────┐
│         INFERENCE PC (RTX 4080 16GB)            │
├─────────────────────────────────────────────────┤
│                                                  │
│  ~/openpi_local/                                │
│  ├── checkpoints/                               │
│  │   └── pi0_lighter_cup_trossen/              │
│  │       └── 20000/  ← Transferred from Server │
│  │                                               │
│  ├── Policy Server (localhost:8000)             │
│  │   ├── GPU Inference (PyTorch on RTX 4080)   │
│  │   └── WebSocket Listener                     │
│  │                                               │
│  └── Robot Client                               │
│      ├── WebSocket → localhost:8000             │
│      ├── LeRobot V0.3.2                         │
│      └── Control Loop (50Hz)                    │
│           ↓                                      │
│      Trossen AI Hardware                        │
│      ├── 2
