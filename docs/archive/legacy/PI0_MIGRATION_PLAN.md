# Pi0 Migration Plan - Trossen AI Stationary Kit

**Erstellt:** 07.01.2025  
**Zweck:** Umfassender Plan für den Umstieg von ACT Transformer auf Pi0  
**Hardware:** Trossen AI Stationary Kit (kompatibel mit Aloha Legacy)

---

## 📑 Inhaltsverzeichnis

1. [Executive Summary](#executive-summary)
2. [Architektur-Übersicht](#architektur-übersicht)
3. [Voraussetzungen](#voraussetzungen)
4. [Phase 1: Setup & Installation](#phase-1-setup--installation)
5. [Phase 2: Daten-Vorbereitung](#phase-2-daten-vorbereitung)
6. [Phase 3: Training-Konfiguration](#phase-3-training-konfiguration)
7. [Phase 4: Training](#phase-4-training)
8. [Phase 5: Inference Setup](#phase-5-inference-setup)
9. [Phase 6: Hardware-Deployment](#phase-6-hardware-deployment)
10. [Troubleshooting](#troubleshooting)
11. [Referenzen](#referenzen)

---

## Executive Summary

### Was ist Pi0?

**Pi0** ist ein Vision-Language-Action (VLA) Foundation Model von Physical Intelligence:
- Pre-trained auf 10k+ Stunden Robot-Daten
- Flow-Matching basiert (statt CVAE wie ACT)
- Unterstützt Language Conditioning
- Zero-shot Generalisierung möglich

### Warum Pi0 statt ACT?

**Vorteile:**
- ✅ Kein Posterior Collapse Problem (Flow-Matching statt VAE)
- ✅ Pre-trained Model → weniger Daten nötig
- ✅ Bessere Generalisierung durch Foundation Model Ansatz
- ✅ Language Conditioning für flexible Task-Spezifikation
- ✅ State-of-the-art Performance in Benchmarks

**Nachteile:**
- ❌ Langsamere Inference (~50-100ms vs. ~10ms bei ACT)
- ❌ Höherer GPU Memory Bedarf (22.5GB für LoRA Fine-tuning)
- ❌ Komplexere Infrastruktur (Server-Client Architektur)
- ❌ Weniger dokumentiert/erprobt auf Trossen Hardware

### Migrations-Aufwand

- **Einfach:** Datensammlung bleibt gleich (LeRobot Format)
- **Mittel:** Neue Repositories klonen, UV Setup, Config anpassen
- **Komplex:** Server-Client Infrastruktur, Hardware-Integration

**Geschätzte Zeit:** 2-3 Tage für vollständiges Setup + erstes Training

---

## Architektur-Übersicht

### System-Komponenten

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  LeRobot Dataset (HuggingFace)                               │
│           ↓                                                   │
│  Normalization Stats Computation                             │
│           ↓                                                   │
│  openpi Training (LoRA Fine-tuning)                          │
│    - Base Model: Pi0 (pre-trained)                           │
│    - LeRobot V0.1.0                                          │
│    - UV Environment #1                                        │
│           ↓                                                   │
│  Trained Checkpoint                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌────────────────────┐        │
│  │  Policy Server   │◄────────┤  Trained Checkpoint │        │
│  │  (GPU Machine)   │         └────────────────────┘        │
│  │                  │                                        │
│  │  - Port 8000     │                                        │
│  │  - WebSocket     │                                        │
│  └────────┬─────────┘                                        │
│           │                                                   │
│           │ Actions                                          │
│           │ via WebSocket                                    │
│           ↓                                                   │
│  ┌──────────────────┐                                        │
│  │  Robot Client    │                                        │
│  │  (on Robot)      │                                        │
│  │                  │                                        │
│  │  - LeRobot V0.3.2│                                        │
│  │  - BiWidowXAI    │                                        │
│  │  - UV Env #2     │                                        │
│  └────────┬─────────┘                                        │
│           │                                                   │
│           ↓                                                   │
│  ┌──────────────────┐                                        │
│  │  Trossen AI HW   │                                        │
│  │  - Motors        │                                        │
│  │  - Cameras       │                                        │
│  └──────────────────┘                                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Zwei LeRobot Versionen

**Wichtig:** Das System nutzt zwei verschiedene LeRobot Versionen!

| Version | Zweck | Location | Environment |
|---------|-------|----------|-------------|
| **V0.1.0** | Training mit openpi | `.venv/` | UV Env #1 (project root) |
| **V0.3.2** | Inference Client | `examples/trossen_ai/.venv/` | UV Env #2 (client dir) |

**Grund:** Isolierte Dependencies, V0.3.2 hat BiWidowXAIFollower Support

---

## Voraussetzungen

### Hardware-Requirements

**Deine Hardware-Konfiguration:**

**Training Server:**
- ✅ NVIDIA RTX 6000 Pro 96GB (Blackwell)
- **Vorteil:** Massive 96GB VRAM ermöglicht:
  - Full Fine-tuning (nicht nur LoRA) möglich
  - Große Batch Sizes (16-32+)
  - Schnelleres Training
  - Mehrere Experimente parallel

**Workstation (Inference):**
- ✅ NVIDIA RTX 4080 (16GB)
- **Vorteil:** Mehr als ausreichend für Inference (nur 8GB nötig)
- Kann auch kleinere Trainings-Experimente lokal testen

**Zusätzliche Requirements:**
- Trossen AI Stationary Kit
- 4 Kameras konfiguriert
- 64GB+ RAM empfohlen (beide Maschinen)
- 200GB+ freier Speicherplatz (Training Server)

### Software-Requirements

- Ubuntu 22.04 (getestet, andere Versionen nicht offiziell unterstützt)
- Python 3.11
- CUDA 12.x
- UV Package Manager
- Git LFS
- Bestehende LeRobot Installation (für Datensammlung)

### Bestehende Ressourcen

**Was du bereits hast:**
- ✅ Trossen AI Hardware Setup
- ✅ LeRobot für Datensammlung
- ✅ ACT Training Erfahrung
- ✅ Gesammelte Datasets

**Was du brauchst:**
- 🆕 openpi Repository
- 🆕 UV Package Manager
- 🆕 Pi0 Base Model Checkpoints (auto-download)
- 🆕 Custom Training Config

---

## Phase 1: Setup & Installation

### 1.1 UV Package Manager installieren

```bash
# Installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

### 1.2 OpenPI Repository klonen

```bash
# Neues Verzeichnis (parallel zu deinem lerobot/)
cd ~/
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git

# Falls bereits geklont ohne submodules:
cd openpi
git submodule update --init --recursive
```

### 1.3 Training Environment Setup

```bash
cd ~/openpi

# Environment erstellen (UV managed)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Verify Installation
uv run python -c "import openpi; print('OpenPI installed successfully')"
```

**Note:** `GIT_LFS_SKIP_SMUDGE=1` verhindert Download großer LFS-Dateien während pip install

### 1.4 Inference Environment Setup (später)

```bash
cd ~/openpi/examples/trossen_ai

# Separate environment für Client
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Verify LeRobot V0.3.2
uv run python -c "import lerobot; print(lerobot.__version__)"
# Should print: 0.3.2
```

### 1.5 Verify GPU Access

```bash
cd ~/openpi
uv run python -c "import jax; print(jax.devices())"
# Should show GPU devices

uv run python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## Phase 2: Daten-Vorbereitung

### 2.1 Bestehende Daten analysieren

**Du sammelst Daten weiterhin wie gewohnt mit LeRobot!**

```bash
# In deinem bestehenden lerobot environment
cd ~/lerobot

# Dataset Structure anschauen
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('dein-dataset-name')
print('Episodes:', ds.num_episodes)
print('Cameras:', list(ds.camera_keys))
print('State dim:', ds.meta.get('state_dim'))
print('Action dim:', ds.meta.get('action_dim'))
"
```

### 2.2 Dataset Format Requirements

Pi0 erwartet folgendes Format:

```python
# Dataset Entry Structure
{
    "observation": {
        "images": {
            "cam_high": (480, 640, 3),           # Top camera
            "cam_low": (480, 640, 3),            # Bottom camera  
            "cam_left_wrist": (480, 640, 3),    # Left wrist camera
            "cam_right_wrist": (480, 640, 3),   # Right wrist camera
        },
        "state": (N,)  # Joint positions + gripper states
    },
    "action": (N,)  # Next joint positions + gripper commands
}
```

### 2.3 Camera Mapping definieren

**Wichtig:** Du musst deine Camera-Namen auf Pi0-erwartete Namen mappen!

```python
# Beispiel Mapping (anpassen an deine Namen!)
camera_mapping = {
    "top": "cam_high",              # Dein Name → Pi0 Name
    "bottom": "cam_low",
    "left": "cam_left_wrist", 
    "right": "cam_right_wrist",
}
```

**Deine aktuelle Camera Config dokumentieren:**
```bash
# TODO: Fülle dies aus mit deinen echten Werten
# Camera 1 Name: _____________
# Camera 2 Name: _____________
# Camera 3 Name: _____________
# Camera 4 Name: _____________
```

### 2.4 Dataset auf HuggingFace hochladen (optional)

```bash
# Falls noch nicht getan
cd ~/lerobot

python lerobot/scripts/push_dataset_to_hub.py \
  --raw-dir data/dein-dataset \
  --repo-id dein-username/dein-dataset-name \
  --raw-format lerobot
```

**Alternativ:** Lokale Datasets werden auch unterstützt

---

## Phase 3: Training-Konfiguration

### 3.1 Custom Training Config erstellen

Erstelle oder modifiziere `~/openpi/src/training/config.py`:

```python
from openpi.training import config as _config
from openpi.training.data import LeRobotAlohaDataConfig
from openpi.training import weight_loaders
from openpi.shared import transforms as _transforms
from openpi.models import pi0

# Dein Custom Config Name
TrainConfig(
    name="pi0_trossen_ai_custom",  # <-- Dein Config Name
    
    # Model Config: LoRA Fine-tuning
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",      # Vision-Language Encoder
        action_expert_variant="gemma_300m_lora"  # Action Decoder
    ),
    
    # Data Config
    data=LeRobotAlohaDataConfig(
        # Dein Dataset
        repo_id="dein-username/dein-dataset-name",  # <-- ANPASSEN!
        
        # Aloha-spezifische Settings
        use_delta_joint_actions=False,  # Absolute actions (empfohlen)
        adapt_to_pi=False,              # Trossen != Pi internal format
        
        # Assets (normalization stats werden hier gespeichert)
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen_custom",  # <-- ANPASSEN!
        ),
        
        # Task Prompt (für Language Conditioning)
        default_prompt="deine task beschreibung hier",  # <-- ANPASSEN!
        
        # Camera Mapping
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        # ANPASSEN an deine Camera Namen!
                        "cam_high": "observation.images.top",
                        "cam_left_wrist": "observation.images.left",
                        "cam_right_wrist": "observation.images.right",
                        "cam_low": "observation.images.bottom",
                    },
                    "state": "observation.state",
                    "actions": "action",
                })
            ]
        ),
    ),
    
    # Load Pre-trained Pi0 Base Model
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    
    # Training Hyperparameters
    num_train_steps=20_000,     # Trossen empfiehlt 20k
    batch_size=8,               # Passt auf 24GB GPU
    
    # Freeze all except LoRA layers
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    
    # No EMA (Trossen recommendation)
    ema_decay=None,
)
```

**Speichere diese Config in:** `~/openpi/src/training/config.py`

### 3.2 Config Validation

```bash
cd ~/openpi

# Verify Config lädt korrekt
uv run python -c "
from openpi.training import config
cfg = config.get_config('pi0_trossen_ai_custom')
print('Config loaded successfully!')
print(f'Dataset: {cfg.data.repo_id}')
print(f'Steps: {cfg.num_train_steps}')
print(f'Batch size: {cfg.batch_size}')
"
```

### 3.3 Normalization Stats berechnen

**Kritischer Schritt:** Berechnet Mean/Std für State und Action Normalisierung

```bash
cd ~/openpi

# Compute stats für dein Dataset
uv run scripts/compute_norm_stats.py \
  --config-name pi0_trossen_ai_custom

# Output wird gespeichert in:
# ~/.cache/openpi/assets/trossen_custom/norm_stats.npz
```

**Verify Stats:**
```bash
uv run python -c "
import numpy as np
stats = np.load('~/.cache/openpi/assets/trossen_custom/norm_stats.npz')
print('State mean shape:', stats['state_mean'].shape)
print('State std shape:', stats['state_std'].shape)
print('Action mean shape:', stats['action_mean'].shape)
print('Action std shape:', stats['action_std'].shape)
"
```

---

## Phase 4: Training

### 4.1 Test Training auf Subset

**Empfehlung:** Teste erst auf kleinem Subset!

```bash
cd ~/openpi

# Modifiziere Config temporär für Test:
# - Reduziere num_train_steps auf 1000
# - Nutze nur 5 Episodes

# Dann starte Test Training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_trossen_ai_custom \
  --exp-name=test_run \
  --overwrite
```

**Was zu erwarten:**
- Training sollte ohne Errors starten
- Loss sollte sinken
- Checkpoints werden gespeichert in `checkpoints/pi0_trossen_ai_custom/test_run/`
- WandB Logging (optional)

### 4.2 Full Training

```bash
cd ~/openpi

# Full Training (alle Daten, 20k steps)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_trossen_ai_custom \
  --exp-name=full_training_v1 \
  --overwrite
```

**Training Settings:**
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Nutzt 90% GPU Memory (statt 75%)
- `--exp-name`: Name deines Training Runs
- `--overwrite`: Überschreibt existierende Checkpoints

**🚀 RTX 6000 Pro Optimierungen:**

Mit deiner 96GB GPU kannst du deutlich aggressiver trainieren:

```bash
# Optimierte Config für RTX 6000 Pro (96GB)
# Modifiziere in config.py:
batch_size=32,  # Statt 8! (4x größer)
num_train_steps=20_000,  # Gleich
# Oder:
batch_size=16,
num_train_steps=40_000,  # Doppelt so lange

# Training Command mit voller GPU Nutzung:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi0_trossen_ai_custom \
  --exp-name=full_training_optimized \
  --overwrite
```

**Vorteile großer Batch Size:**
- Schnellere Konvergenz
- Stabileres Training
- Bessere Generalisierung
- 2-3x schnelleres Training

**Training Monitoring:**
```bash
# Loss sollte kontinuierlich sinken
# Expected: ~0.1-0.5 nach 20k steps (dataset-abhängig)

# Bei Problemen:
# - Zu hohe Loss: Mehr Daten sammeln oder learning rate reduzieren
# - Loss explodiert: Gradient clipping aktivieren, batch size reduzieren
# - OOM Error: Unwahrscheinlich mit 96GB! Falls doch: batch_size reduzieren
```

### 4.3 Checkpoint Management

```bash
# Checkpoints werden gespeichert alle N steps (konfigurierbar)
# Default Location: checkpoints/pi0_trossen_ai_custom/full_training_v1/

# Structure:
# checkpoints/
#   └── pi0_trossen_ai_custom/
#       └── full_training_v1/
#           ├── 5000/      # Checkpoint bei step 5000
#           ├── 10000/
#           ├── 15000/
#           └── 20000/     # Final checkpoint

# Best checkpoint identifizieren (basierend auf validation loss)
ls -lh checkpoints/pi0_trossen_ai_custom/full_training_v1/
```

### 4.4 Training Duration

**Erwartete Trainingszeit:**
- RTX 4090: ~8-12 Stunden für 20k steps
- A100: ~6-8 Stunden
- H100: ~4-6 Stunden

**Abhängig von:**
- Dataset-Größe
- Batch size
- GPU Hardware

---

## Phase 5: Inference Setup

### 5.1 Policy Server starten

```bash
cd ~/openpi

# Server mit deinem trainierten Checkpoint starten
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_trossen_ai_custom \
  --policy.dir=checkpoints/pi0_trossen_ai_custom/full_training_v1/20000

# Server läuft auf Port 8000
# Output: "Policy server listening on ws://0.0.0.0:8000"
```

**Server Options:**
```bash
# Custom Port
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_trossen_ai_custom \
  --policy.dir=checkpoints/pi0_trossen_ai_custom/full_training_v1/20000 \
  --port=8080

# Bind zu specific IP
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_trossen_ai_custom \
  --policy.dir=checkpoints/pi0_trossen_ai_custom/full_training_v1/20000 \
  --host=192.168.1.100
```

### 5.2 Server Health Check

```bash
# In neuem Terminal
curl http://localhost:8000/health

# Expected Response:
# {"status": "healthy", "model_loaded": true}
```

### 5.3 Test Inference (ohne Hardware)

```bash
cd ~/openpi/examples/simple_client

# Dummy inference test
uv run python simple_client.py \
  --policy_url ws://localhost:8000

# Generiert random observations und testet inference
# Zeigt predicted actions
```

---

## Phase 6: Hardware-Deployment

### 6.1 Client Environment Setup

```bash
cd ~/openpi/examples/trossen_ai

# Falls noch nicht getan (siehe Phase 1.4)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 6.2 Hardware Configuration Check

**Wichtige Vorbereitungen:**

```bash
# 1. Motors Calibration
# Nutze deine bestehende LeRobot Calibration!

# 2. Cameras Check
# Verifiziere alle 4 Kameras funktionieren
v4l2-ctl --list-devices

# 3. Port Mapping
# Stelle sicher udev rules korrekt sind (aus deinem LeRobot Setup)
ls -l /dev/ttyDXL_*
ls -l /dev/CAM_*
```

### 6.3 Client Script anpassen

Erstelle `~/openpi/examples/trossen_ai/run_trossen_client.py`:

```python
#!/usr/bin/env python3
"""
Trossen AI Client für Pi0 Inference
Verbindet Hardware mit Policy Server
"""

import asyncio
from lerobot.common.robot_devices.robots.biwidowxai_follower import BiWidowXAIFollower
from openpi.client import PolicyClient

async def main():
    # Policy Server URL
    policy_url = "ws://localhost:8000"  # Anpassen falls Server remote
    
    # Robot initialisieren
    robot = BiWidowXAIFollower(
        # Deine Hardware Config hier
        # (aus deinem bestehenden LeRobot Setup übernehmen)
    )
    
    # Policy Client initialisieren
    client = PolicyClient(policy_url)
    
    # Main control loop
    try:
        robot.connect()
        await client.connect()
        
        print("Connected! Starting inference...")
        
        while True:
            # Get observations from robot
            obs = robot.get_observation()
            
            # Send to policy server, get actions
            actions = await client.infer(obs)
            
            # Execute actions
            robot.send_action(actions)
            
            # Control frequency (z.B. 50Hz)
            await asyncio.sleep(0.02)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        robot.disconnect()
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### 6.4 Safety Checks implementieren

```python
# In Client Script hinzufügen:

def check_action_safety(actions, limits):
    """Verify actions are within safe limits"""
    # Check joint limits
    # Check gripper limits  
    # Check velocity limits
    return safe_actions

def emergency_stop(robot):
    """Emergency stop routine"""
    robot.send_action(robot.get_current_state())  # Hold position
    print("EMERGENCY STOP TRIGGERED")
```

### 6.5 Erste Hardware Tests

**Schritt-für-Schritt:**

1. **Server läuft** (von Phase 5.1)
2. **Safety mode:** Arms in freespace, keine Objekte
3. **Start client:**
   ```bash
   cd ~/openpi/examples/trossen_ai
   uv run python run_trossen_client.py
   ```
4. **Beobachte Verhalten:**
   - Smoothe Bewegungen?
   - Korrekte Richtungen?
   - Reaktion auf Perturbationen?

5. **Bei Problemen:**
   - Emergency Stop (Ctrl+C)
   - Logs analysieren
   - Actions plotten
   - Config anpassen

### 6.6 Performance Benchmarks

```python
# Latency messen
import time

start = time.time()
actions = await client.infer(obs)
latency = time.time() - start

print(f"Inference latency: {latency*1000:.1f}ms")

# Target: < 50ms für 50Hz control
# Falls höher: Optimization nötig
```

---

## Troubleshooting

### Training Issues

**Problem: Loss explodiert**
```bash
# Solution 1: Reduziere Learning Rate
# In config.py: optimizer_lr = 2.5e-5 → 1e-5

# Solution 2: Gradient Clipping
# Add to config: max_grad_norm = 1.0

# Solution 3: Kleinere Batch Size
# batch_size = 8 → 4
```

**Problem: OOM (Out of Memory)**
```bash
# Solution 1: Reduziere Batch Size
# batch_size = 8 → 4 → 2

# Solution 2: Mehr GPU Memory freigeben
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95  # statt 0.9

# Solution 3: Gradient Accumulation
# accumulation_steps = 2  # Effective batch size = 2*4 = 8
```

**Problem: Slow Training**
```bash
# Check GPU Utilization
nvidia-smi -l 1

# Falls niedrig (<80%):
# - Increase batch_size
# - Check data loading (bottleneck?)
# - Enable mixed precision (falls verfügbar)
```

### Inference Issues

**Problem: Server startet nicht**
```bash
# Check Port ist frei
lsof -i :8000

# Falls besetzt:
pkill -f serve_policy
# oder nutze anderen Port: --port=8080
```

**Problem: Client kann nicht verbinden**
```bash
# Check Server läuft
curl http://localhost:8000/health

# Check Firewall (falls Server remote)
sudo ufw allow 8000

# Check Server IP/Port im Client Script
```

**Problem: Hohe Latency (>100ms)**
```bash
# Check Network latency (falls remote)
ping training-server-ip

# Reduce inference steps in config
# num_inference_steps = 10 → 5

# Use DDIM sampling (schneller)
# In model config: ddim_inference = True
```

**Problem: Actions sind abgehackt/unsmooth**
```bash
# Increase action chunk size
# n_action_steps in config erhöhen

# Check control frequency
# 50Hz empfohlen, ggf. reduzieren auf 20Hz

# Temporal smoothing anwenden
# Moving average über letzte N actions
```

---

## Remote Training & Inference Setup

**Deine Setup-Konfiguration:**
- **Training Server:** RTX 6000 Pro 96GB (für Training)
- **Workstation:** RTX 4080 16GB (mit Trossen Hardware)

### Netzwerk Setup

```bash
# 1. Auf Training Server (RTX 6000 Pro):
# Finde IP Adresse
hostname -I
# z.B. 192.168.1.100

# 2. Firewall öffnen für Policy Server
sudo ufw allow 8000/tcp
sudo ufw status

# 3. Policy Server mit externem Bind starten
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_trossen_ai_custom \
  --policy.dir=checkpoints/pi0_trossen_ai_custom/full_training_v1/20000 \
  --host=0.0.0.0 \
  --port=8000
```

### Client auf Workstation

```python
# In run_trossen_client.py:
async def main():
    # Policy Server URL (Training Server IP)
    policy_url = "ws://192.168.1.100:8000"  # <-- Anpassen!
    
    # Rest bleibt gleich...
```

### Netzwerk Performance optimieren

```bash
# 1. Test Latency
ping 192.168.1.100

# Target: <5ms für Local Network

# 2. Test Bandbreite
iperf3 -s  # Auf Server
iperf3 -c 192.168.1.100  # Auf Client

# 3. Bei hoher Latenz:
# - Gigabit Ethernet verwenden (nicht WiFi!)
# - Direkte Verbindung zwischen Maschinen
# - QoS für Port 8000 aktivieren
```

### Checkpoint Transfer

```bash
# Option 1: rsync (empfohlen)
rsync -avz --progress \
  training-server:/path/to/checkpoints/ \
  ~/local-checkpoints/

# Option 2: scp
scp -r training-server:/path/to/checkpoints/ \
  ~/local-checkpoints/

# Option 3: Shared Network Drive (NFS/SMB)
# Server kann direkt auf Checkpoints zugreifen
```

### Alternative: Training auf Workstation

```bash
# Falls Netzwerk-Latency zu hoch:
# Transferiere Checkpoint auf Workstation mit RTX 4080
# Starte Policy Server lokal

# Dann im Client:
policy_url = "ws://localhost:8000"
```

---

## Referenzen

### Offizielle Dokumentation

**Pi0 & Physical Intelligence:**
- [π₀ Paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [π₀ Blog Post](https://www.physicalintelligence.company/blog/pi0)
- [Physical Intelligence GitHub - openpi](https://github.com/Physical-Intelligence/openpi)
- [Physical Intelligence GitHub - aloha](https://github.com/Physical-Intelligence/aloha)

**Trossen Robotics:**
- [Trossen openpi Tutorial](https://docs.trossenrobotics.com/trossen_arm/v1.9/tutorials/openpi.html) - **Wichtigste Referenz!**
- [Trossen AI Stationary Kit Docs](https://docs.trossenrobotics.com/trossen_arm/)
- [TrossenRobotics openpi Fork](https://github.com/TrossenRobotics/openpi)

**LeRobot:**
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Pi0 Docs](https://huggingface.co/docs/lerobot/main/pi0)
- [LeRobot Datasets](https://huggingface.co/lerobot)

### Beispiel Datasets

**Trossen Community:**
- [Bimanual WidowX-AI Handover Cube](https://huggingface.co/datasets/TrossenRoboticsCommunity/bimanual-widowxai-handover-cube)

**DROID (für Referenz):**
- [DROID Dataset](https://droid-dataset.github.io/)

### Tools & Dependencies

**UV Package Manager:**
- [UV Documentation](https://docs.astral.sh/uv/)
- [UV Installation](https://docs.astral.sh/uv/getting-started/installation/)

**JAX (für Training):**
- [JAX Documentation](https://jax.readthedocs.io/)

**PyTorch (für Inference Client):**
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Quick-Start Checkliste

### Pre-Training Checklist

- [ ] **Hardware Check**
  - [ ] Training Server: RTX 6000 Pro 96GB verfügbar
  - [ ] Workstation: RTX 4080 verfügbar  
  - [ ] Trossen AI Hardware kalibriert
  - [ ] 4 Kameras funktionieren

- [ ] **Software Installation**
  - [ ] UV Package Manager installiert
  - [ ] openpi Repository geklont (mit submodules)
  - [ ] Training Environment Setup (Phase 1.3)
  - [ ] GPU Access verifiziert

- [ ] **Daten Vorbereitung**
  - [ ] Datasets mit LeRobot gesammelt
  - [ ] Camera Namen dokumentiert
  - [ ] Dataset auf HuggingFace (optional)
  - [ ] Camera Mapping definiert

- [ ] **Training Config**
  - [ ] Custom Config erstellt in `config.py`
  - [ ] Dataset repo_id angepasst
  - [ ] Camera Mapping angepasst
  - [ ] Task Prompt definiert
  - [ ] Normalization stats berechnet

### Training Checklist

- [ ] **Test Training**
  - [ ] Test auf 5 Episodes läuft
  - [ ] Loss sinkt
  - [ ] Keine Errors
  - [ ] Checkpoints werden gespeichert

- [ ] **Full Training**
  - [ ] Batch size für RTX 6000 Pro optimiert (16-32)
  - [ ] Training gestartet (20k+ steps)
  - [ ] WandB monitoring (optional)
  - [ ] GPU Utilization >80%

- [ ] **Checkpoint Management**
  - [ ] Multiple checkpoints gespeichert
  - [ ] Best checkpoint identifiziert
  - [ ] Checkpoint auf Workstation kopiert (falls remote)

### Inference Checklist

- [ ] **Server Setup**
  - [ ] Policy Server startet ohne Errors
  - [ ] Health check erfolgreich
  - [ ] Test inference funktioniert

- [ ] **Client Setup**
  - [ ] Inference Environment installiert (Phase 1.4)
  - [ ] Client Script erstellt/angepasst
  - [ ] Hardware Config übernommen
  - [ ] Safety checks implementiert

- [ ] **Remote Setup** (falls applicable)
  - [ ] Netzwerk zwischen Server/Workstation getestet
  - [ ] Firewall konfiguriert
  - [ ] Latency <5ms
  - [ ] Server IP im Client konfiguriert

### Hardware Testing Checklist

- [ ] **Safety First**
  - [ ] Arms in freespace positioniert
  - [ ] Emergency stop Routine getestet
  - [ ] Joint limits geprüft
  - [ ] Gripper limits geprüft

- [ ] **Erste Tests**
  - [ ] Server läuft
  - [ ] Client verbindet
  - [ ] Observations werden gesendet
  - [ ] Actions werden empfangen
  - [ ] Movements sind smooth

- [ ] **Performance Benchmarks**
  - [ ] Inference latency gemessen
  - [ ] Latency <50ms für 50Hz control
  - [ ] Success rate dokumentiert
  - [ ] Vergleich mit ACT

---

## Erwartete Ergebnisse

### Training Performance

**Mit deiner RTX 6000 Pro 96GB:**
- Training Time: ~3-5 Stunden (20k steps, batch_size=32)
- Memory Usage: ~40-60GB (abhängig von batch size)
- GPU Utilization: >85%

**Im Vergleich:**
- RTX 4090 (24GB): ~8-12 Stunden (batch_size=8)
- A100 (80GB): ~6-8 Stunden (batch_size=16)

### Inference Performance

**Auf RTX 4080:**
- Latency: 30-50ms (DDIM sampling mit 10 steps)
- Memory Usage: ~6-8GB
- Control Frequency: 20-50Hz möglich

**Network Latency (Remote Setup):**
- Gigabit LAN: <5ms zusätzlich
- Gesamt: 35-55ms total latency
- **Fazit:** Echtzeit-Control möglich!

### Expected vs. ACT

| Metrik | ACT | Pi0 (erwartet) |
|--------|-----|----------------|
| **Success Rate** | Baseline | +10-20% |
| **Inference Latency** | ~10ms | ~40ms |
| **Training Time** | 5h | 3-5h (mit RTX 6000 Pro) |
| **Data Efficiency** | 50+ demos | 20-30 demos |
| **Generalization** | Limited | Better |
| **Multimodal** | Posterior Collapse | Keine Probleme |

---

## Nächste Schritte

1. **✅ Dieses Dokument durchlesen**
2. **📦 Phase 1: Installation**
   - UV, openpi, environments setup
3. **📊 Phase 2: Daten analysieren**
   - Bestehende Datasets prüfen
   - Camera mapping dokumentieren
4. **⚙️ Phase 3: Config erstellen**
   - Training config anpassen
   - Normalization stats berechnen
5. **🚀 Phase 4: Training**
   - Test training
   - Full training mit optimierten settings
6. **🌐 Phase 5: Inference Setup**
   - Server starten
   - Network setup (remote)
7. **🤖 Phase 6: Hardware Tests**
   - Client einrichten
   - Erste vorsichtige Tests
   - Performance benchmarks

---

## Support & Community

**Bei Fragen/Problemen:**

1. **Trossen Docs:** Offizielle Trossen Dokumentation checken
2. **GitHub Issues:** 
   - [TrossenRobotics/openpi Issues](https://github.com/TrossenRobotics/openpi/issues)
   - [Physical-Intelligence/openpi Issues](https://github.com/Physical-Intelligence/openpi/issues)
3. **LeRobot Discord:** Community support
4. **Eigene Experimente:** Dokumentiere deine Findings!

**Dieses Dokument verbessern:**
- Füge deine eigenen Erkenntnisse hinzu
- Dokumentiere spezifische Probleme/Lösungen
- Teile erfolgreiche Konfigurationen

---

## Zusammenfassung

### Was du jetzt hast

✅ **Umfassender Plan** für Pi0 Migration  
✅ **Hardware-spezifische Optimierungen** für RTX 6000 Pro  
✅ **Remote Setup Guide** für Server-Client Architektur  
✅ **Schritt-für-Schritt Checklisten**  
✅ **Troubleshooting Guide**  
✅ **Alle wichtigen Referenzen**

### Warum Pi0 sich lohnt

1. **Kein Posterior Collapse** - Stabiles Training
2. **Pre-trained Model** - Weniger Daten nötig
3. **Bessere Generalisierung** - Foundation Model Ansatz
4. **Language Conditioning** - Flexible Task-Spezifikation
5. **State-of-the-art** - Beste verfügbare Open-Source VLA

### Deine Vorteile

🚀 **RTX 6000 Pro 96GB** - Ultra-schnelles Training  
🎯 **ACT Erfahrung** - Du kennst bereits den Workflow  
🤖 **Trossen Hardware** - Kompatibel mit Pi0  
📊 **Bestehende Daten** - Sofort nutzbar

**Viel Erfolg beim Umstieg auf Pi0! 🎉**
