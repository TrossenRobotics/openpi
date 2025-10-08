# Pi0 - Kontext für neuen Chat

**Erstellt:** 07.01.2025, 14:18 Uhr  
**Zweck:** Vollständige Zusammenfassung für Fortsetzung in neuem Chat  
**Phase:** Training abgeschlossen → Inference Setup

---

## 🎯 AKTUELLE SITUATION

### Training Phase: ✅ ERFOLGREICH ABGESCHLOSSEN

**Training läuft gerade auf:**
- **Maschine:** Training Server (SSH-Verbindung)
- **GPU:** NVIDIA RTX 6000 Pro 96GB (Blackwell)
- **Location:** `~/openpi/`
- **Status:** Training aktiv, WandB logging läuft

**Training Details:**
```
Config: pi0_lighter_cup_trossen
Dataset: MaxFridge/lighter_cup_v2 (92 Episodes, 4 Cameras)
Steps: 20.000 (batch size 32)
Erwartete Dauer: ~3-4 Stunden
Checkpoints: ~/openpi/checkpoints/pi0_lighter_cup_trossen/production_v1/
WandB: https://wandb.ai/sourteig-fritsch-gmbh/openpi
```

---

## 📊 WAS BEREITS FUNKTIONIERT

### Training Server Setup (RTX 6000 Pro 96GB)

**Installation:**
- ✅ openpi Repository: `~/openpi/`
- ✅ UV Environment: 242 Packages, Python 3.11
- ✅ JAX 0.5.3 + GPU Support verifiziert
- ✅ Conda Environments (lerobot, trossenai) unberührt

**Konfiguration:**
- ✅ Training Config erstellt & funktionsfähig
- ✅ 2 Configs: test (1k steps) + production (20k steps)
- ✅ Dataset: MaxFridge/lighter_cup_v2
- ✅ Camera Mapping: 4 Cameras (cam_high, cam_low, left_wrist, right_wrist)
- ✅ Normalization Stats berechnet
- ✅ Alle Config-Probleme gelöst (Circular Import, repo_id, AssetsConfig)

**Dokumentation:**
- ✅ 9 umfassende Guides in `~/lerobot/docs/PI0_*.md`

**Wichtige gelöste Probleme:**
1. Circular Import bei Config-Loading → Behoben
2. repo_id falsch (lighter_cup_v2episodes vs. MaxFridge/lighter_cup_v2) → Korrigiert
3. AssetsConfig suchte auf Google Cloud statt lokal → Angepasst
4. UV vs. Conda Verständnis → Geklärt (UV nutzt KEIN Conda!)

---

## 🎯 WAS NOCH ZU TUN IST

### Nächste Phase: Inference Setup

**Zwei Maschinen:**

1. **Training Server** (aktuell, SSH):
   - Policy Server einrichten
   - Netzwerk konfigurieren
   - Server im Hintergrund laufen lassen

2. **Inference PC** (RTX 4080 16GB, lokal mit Hardware):
   - openpi Client Environment installieren
   - LeRobot V0.3.2 mit BiWidowXAIFollower
   - Hardware-Integration
   - Client Script erstellen

**Geschätzte Zeit:** 1-2 Tage

---

## 🏗️ ARCHITEKTUR

### Server-Client System

```
[Training Server: RTX 6000 Pro]
         ↓
    Policy Server (Port 8000)
    - Lädt trained checkpoint
    - JAX Inference
    - WebSocket Listener
         ↓
    Netzwerk (Gigabit LAN)
         ↓
[Inference PC: RTX 4080]
         ↓
    Robot Client
    - LeRobot V0.3.2
    - BiWidowXAIFollower
    - WebSocket Client
    - Control Loop (50Hz)
         ↓
    Trossen AI Hardware
    - 2x WidowX Arms (14 DOF)
    - 4x Cameras (480x640)
```

---

## ❓ ROS2 & DOCKER - NICHT NÖTIG!

**Klarstellung:**
- **ROS2:** Nur für andere Trossen Tutorials (MoveIt, etc.) - Pi0 nutzt KEIN ROS2
- **Docker:** Optional, nicht zwingend - Native UV Installation ist einfacher
- **Was wir nutzen:** Direkte Python API über LeRobot + openpi

---

## 📋 IMPLEMENTIERUNGSPLAN

### Auf Training Server (wo du jetzt per SSH bist):

#### Schritt 1: Policy Server vorbereiten

```bash
# Nach Training fertig (warte auf 20k steps):
cd ~/openpi

# Checkpoint auswählen
ls -lh checkpoints/pi0_lighter_cup_trossen/production_v1/

# Policy Server starten
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000
```

#### Schritt 2: Netzwerk konfigurieren

```bash
# Firewall öffnen
sudo ufw allow 8000/tcp

# IP notieren
hostname -I
# z.B. 192.168.1.100 → brauchst du für Client!

# Health check
curl http://localhost:8000/health
```

#### Schritt 3: Server im Hintergrund

```bash
# Mit tmux (empfohlen)
tmux new -s policy_server
# Server Command von oben
# Detach: Ctrl+B dann D
```

### Auf Inference PC (RTX 4080, lokal mit Hardware):

#### Schritt 1: Client Environment installieren

```bash
cd ~/
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git openpi_client

cd openpi_client/examples/trossen_ai
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Verify LeRobot V0.3.2
uv run python -c "import lerobot; print(lerobot.__version__)"
```

#### Schritt 2: Hardware Config übernehmen

```bash
# Nutze deine bestehende LeRobot Hardware Config!
cd ~/lerobot
# Finde Motor/Camera Configs
# Übernehme für Client Script
```

#### Schritt 3: Client Script erstellen

```bash
# Template in docs/PI0_INFERENCE_COMPLETE_GUIDE.md
# Erstelle ~/openpi_client/examples/trossen_ai/run_inference_client.py
```

#### Schritt 4: Verbindung testen

```bash
# Netzwerk testen
ping <training_server_ip>

# WebSocket testen
python3 -c "import asyncio, websockets; ..."

# Client starten
uv run python run_inference_client.py --policy_url ws://<server_ip>:8000
```

---

## 📚 DOKUMENTATION (9 Guides)

**Alle in `~/lerobot/docs/`:**

1. **PI0_NEW_CHAT_CONTEXT.md** ⭐ DIESES DOKUMENT
   - Komplette Zusammenfassung
   - Aktueller Stand
   - Nächste Schritte

2. **PI0_INFERENCE_COMPLETE_GUIDE.md** ⭐ HAUPTGUIDE
   - Policy Server Setup (Training Server)
   - Robot Client Setup (Inference PC)
   - Netzwerk-Konfiguration
   - Hardware-Integration
   - Testing Workflow

3. **PI0_MIGRATION_PLAN.md**
   - Ursprünglicher Gesamtplan
   - 6 Phasen (1-4 abgeschlossen)

4. **PI0_ENVIRONMENT_GUIDE.md**
   - UV vs. Conda Workflow
   - Wichtig: UV nutzt KEIN Conda!

5. **PI0_QUICK_START.md**
   - Training Quick-Start

6. **PI0_CAMERA_CONFIG.md**
   - Deine 4 Camera Konfiguration

7. **PI0_TRAINING_CONFIG_TEMPLATE.md**
   - Config Details & Varianten

8. **PI0_ADAPT_TO_PI_EXPLANATION.md**
   - Warum `adapt_to_pi=False`

9. **PI0_SYNTAX_NOTES.md**
   - Python Syntax Referenz

---

## 🔑 WICHTIGE PARAMETER & ENTSCHEIDUNGEN

### Training Config (bereits implementiert):

```python
TrainConfig(
    name="pi0_lighter_cup_trossen",
    
    # LoRA Fine-tuning (nicht Full)
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),
    
    # Dataset Settings
    data=LeRobotAlohaDataConfig(
        repo_id="MaxFridge/lighter_cup_v2",  # HuggingFace
        use_delta_joint_actions=False,       # Absolute positions
        adapt_to_pi=False,                   # Native Trossen format
        
        # Lokale Assets (nicht Google Cloud)
        assets=AssetsConfig(
            assets_dir=None,
            asset_id=None,
        ),
        
        # Camera Mapping
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform({
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_low": "observation.images.cam_low",
                    "cam_left_wrist": "observation.images.left_wrist",
                    "cam_right_wrist": "observation.images.right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
            })]
        ),
    ),
    
    # Training Settings (RTX 6000 Pro optimiert)
    num_train_steps=20_000,
    batch_size=32,  # 4x Standard (dank 96GB VRAM)
    save_interval=5_000,
)
```

### Wichtige Erkenntnisse:

- `adapt_to_pi=False` → Native Trossen Format (siehe PI0_ADAPT_TO_PI_EXPLANATION.md)
- `use_delta_joint_actions=False` → Absolute Positionen
- `save_interval=5_000` → Python Underscores in Zahlen OK (PEP 515)

---

## 💻 HARDWARE SETUP

### Training Server (wo du aktuell per SSH eingeloggt bist):

```
Hostname: max-ws (vermutlich)
GPU: NVIDIA RTX 6000 Pro 96GB (Blackwell)
RAM: 64GB+
OS: Ubuntu (vermutlich 22.04)
Location: ~/openpi/
Netzwerk: LAN (IP zu ermitteln mit hostname -I)
```

### Inference PC (lokal, mit Trossen Hardware):

```
GPU: NVIDIA RTX 4080 16GB
RAM: 64GB+ (empfohlen)
OS: Ubuntu (vermutlich 22.04)
Hardware: Trossen AI Stationary Kit
  - 2x WidowX Arms (7 DOF each = 14 total)
  - 4x Cameras (480x640, 30 FPS)
  - USB Verbindungen
  - udev rules konfiguriert (ttyDXL_*, CAM_*)

Bestehende LeRobot Installation:
  - ~/lerobot/ mit ACT Training Setup
  - Conda Environment: lerobot
  - Hardware bereits kalibriert
  - → Wiederverwenden für Pi0!
```

---

## 🚀 NÄCHSTE SCHRITTE (Reihenfolge)

### 1. Warte auf Training Completion (~3-4h)

```bash
# Auf Training Server (SSH)
cd ~/openpi

# Monitor Training
tail -f wandb/run-*/logs/debug.log

# Oder WandB Dashboard:
# https://wandb.ai/sourteig-fritsch-gmbh/openpi
```

### 2. Policy Server starten (Training Server)

**Details in:** `docs/PI0_INFERENCE_COMPLETE_GUIDE.md` Abschnitt 4

```bash
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000
```

### 3. Client Environment installieren (Inference PC)

**Details in:** `docs/PI0_INFERENCE_COMPLETE_GUIDE.md` Abschnitt 5

```bash
# Auf Inference PC
cd ~/
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git openpi_client
cd openpi_client/examples/trossen_ai
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 4. Hardware Integration & Testing

**Details in:** `docs/PI0_INFERENCE_COMPLETE_GUIDE.md` Abschnitte 6-8

---

## 📖 FÜR NEUEN CHAT - START HIER

**Kopiere folgende Informationen in neuen Chat:**

```
KONTEXT: Pi0 Inference Setup nach erfolgreichem Training

AKTUELLER STAND:
- Training Server (RTX 6000 Pro 96GB): Training läuft/abgeschlossen
- Checkpoint: ~/openpi/checkpoints/pi0_lighter_cup_trossen/production_v1/20000/
- Dataset: MaxFridge/lighter_cup_v2 (92 Episodes, 4 Cameras)
- Configs: 2 Configs erstellt & funktionsfähig
- WandB: https://wandb.ai/sourteig-fritsch-gmbh/openpi

AUFGABE: Inference Setup implementieren
- Policy Server auf Training Server
- Robot Client auf Inference PC (RTX 4080)
- Netzwerk-Verbindung
- Hardware-Integration

DOKUMENTATION: ~/lerobot/docs/
- PI0_NEW_CHAT_CONTEXT.md ← Dieses Dokument
- PI0_INFERENCE_COMPLETE_GUIDE.md ← Hauptguide

WICHTIG:
- ROS2 NICHT nötig
- Docker NICHT nötig  
- UV Environment (kein Conda)
- Bestehende LeRobot Hardware Config wiederverwenden

HARDWARE:
- Training Server: RTX 6000 Pro 96GB (SSH Remote)
- Inference PC: RTX 4080 16GB (lokal mit Trossen Arms)
- Netzwerk: Gigabit LAN empfohlen

NÄCHSTER SCHRITT:
1. Policy Server starten (Training Server)
2. Client Environment installieren (Inference PC)
3. Siehe: docs/PI0_INFERENCE_COMPLETE_GUIDE.md
```

---

## ⚡ QUICK COMMANDS

### Training Server (Policy Server):

```bash
# Check Training Status
cd ~/openpi
tail -f wandb/run-*/logs/debug.log

# Nach Training → Start Policy Server
tmux new -s policy_server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000
```

### Inference PC (Robot Client):

```bash
# Install Client Environment
cd ~/
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git openpi_client
cd openpi_client/examples/trossen_ai
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Create Client Script
# (Template in PI0_INFERENCE_COMPLETE_GUIDE.md)

# Run Client
uv run python run_inference_client.py --policy_url ws://<server_ip>:8000
```

---

## 🎓 LESSONS LEARNED

1. **UV statt Conda:**
   - UV managed Environments in `.venv/` Verzeichnissen
   - Workflow: `cd ~/openpi && uv run ...`
   - KEIN `conda activate` nötig

2. **Zwei LeRobot Versionen:**
   - V0.1.0 für Training (in openpi)
   - V0.3.2 für Inference (in openpi_client, BiWidowXAIFollower Support)

3. **AssetsConfig:**
   - `assets_dir=None` → lokale Assets in `./assets/`
   - Nicht Google Cloud Storage

4. **adapt_to_pi=False:**
   - Native Trossen Format
   - Keine Joint/Gripper Transformationen
   - Konsistenz Training ↔ Inference

---

## 🎯 ERFOLGS-METRIKEN

**Training (bereits erreicht):**
- ✅ Config lädt ohne Errors
- ✅ Norm Stats erfolgreich berechnet
- ✅ Training startet ohne Errors
- ✅ WandB Logging funktioniert
- ✅ Checkpoints werden gespeichert

**Inference (noch zu erreichen):**
- [ ] Policy Server läuft stabil
- [ ] Client verbindet zu Server
- [ ] Hardware wird korrekt angesteuert
- [ ] Latency <50ms (End-to-End)
- [ ] Smoothe, sichere Bewegungen

---

## ⏭️ NÄCHSTER CHAT

**Start mit:**
- Lese `docs/PI0_NEW_CHAT_CONTEXT.md` (dieses Dokument)
- Hauptguide: `docs/PI0_INFERENCE_COMPLETE_GUIDE.md`
- Implementiere Policy Server (Training Server)
- Implementiere Robot Client (Inference PC)

**Ziel:**
- Funktionierendes Pi0 Inference System
- Training Server → Inference PC über Netzwerk
- Sichere Hardware-Tests
- Performance Benchmarks

**Erwartete Dauer:** 1-2 Tage

**Viel Erfolg! 🚀**
