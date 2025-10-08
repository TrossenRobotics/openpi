# Pi0 Inference - Vollständiger Setup Guide

**Erstellt:** 07.01.2025  
**Zweck:** Komplette Anleitung für Inference-Phase nach erfolgreichem Training  
**Hardware:** Training Server (RTX 6000 Pro 96GB) + Inference PC (RTX 4080 16GB)

---

## 📑 Inhaltsverzeichnis

1. [Stand der Implementierung](#stand-der-implementierung)
2. [Architektur-Übersicht Inference](#architektur-übersicht-inference)
3. [ROS2 & Container - Notwendigkeit](#ros2--container---notwendigkeit)
4. [Training Server: Policy Server Setup](#training-server-policy-server-setup)
5. [Inference PC: Client Setup](#inference-pc-client-setup)
6. [Netzwerk-Konfiguration](#netzwerk-konfiguration)
7. [Hardware-Integration](#hardware-integration)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting-inference)

---

## Stand der Implementierung

### ✅ Training Phase - ABGESCHLOSSEN

**Training Server (RTX 6000 Pro 96GB):**

```
Location: ~/openpi/
Status: Training läuft erfolgreich!

Installation:
✅ openpi Repository geklont
✅ UV Environment: 242 Packages (Python 3.11)
✅ JAX 0.5.3 + GPU Support
✅ Conda Environments unberührt (lerobot, trossenai)

Konfiguration:
✅ 2 Training Configs erstellt:
   - pi0_lighter_cup_test (1k steps, batch 16)
   - pi0_lighter_cup_trossen (20k steps, batch 32)
✅ Dataset: MaxFridge/lighter_cup_v2 (92 Episodes)
✅ 4 Cameras gemappt (cam_high, cam_low, left_wrist, right_wrist)
✅ Normalization Stats berechnet

Training:
✅ Läuft aktuell erfolgreich
✅ WandB Logging: https://wandb.ai/sourteig-fritsch-gmbh/openpi
✅ Checkpoints werden gespeichert: ~/openpi/checkpoints/pi0_lighter_cup_trossen/

Dokumentation:
✅ 8 Guides in ~/lerobot/docs/PI0_*.md
```

### ⏳ Inference Phase - ZU IMPLEMENTIEREN

**Was noch fehlt:**

1. **Policy Server** (auf Training Server)
   - Server-Script konfigurieren
   - Netzwerk freigeben
   - Health checks

2. **Robot Client** (auf Inference PC)
   - openpi Client Environment
   - LeRobot V0.3.2 mit BiWidowXAIFollower
   - Hardware-Integration
   - Client Script

3. **Netzwerk**
   - SSH/WebSocket Verbindung
   - Firewall-Konfiguration
   - Latency-Optimierung

---

## Architektur-Übersicht Inference

### Zwei-Maschinen Setup

```
┌─────────────────────────────────────────────────────┐
│         TRAINING SERVER (SSH Remote)                │
│         RTX 6000 Pro 96GB                           │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ~/openpi/                                          │
│  ├── checkpoints/                                   │
│  │   └── pi0_lighter_cup_trossen/                  │
│  │       └── production_v1/                        │
│  │           └── 20000/  ← Trained Model           │
│  │                                                   │
│  └── Policy Server (Port 8000)                      │
│      ├── Lädt Checkpoint                            │
│      ├── WebSocket Listener                         │
│      └── GPU Inference (JAX)                        │
│                                                      │
│      ↓ WebSocket (Port 8000)                       │
│      ↓ Observations → Actions                       │
│                                                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ LAN/Internet
                       │ (Gigabit Ethernet empfohlen)
                       │
┌──────────────────────┴──────────────────────────────┐
│         INFERENCE PC (Lokal)                        │
│         RTX 4080 16GB                                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ~/openpi_client/                                   │
│  └── Robot Client                                   │
│      ├── LeRobot V0.3.2                            │
│      ├── BiWidowXAIFollower                        │
│      ├── WebSocket Client                           │
│      └── Control Loop (50Hz)                        │
│                                                      │
│      ↓ Motor Commands                               │
│      ↓ Camera Streams                               │
│                                                      │
│  Trossen AI Stationary Kit                         │
│  ├── 2x WidowX Arms (14 DOF total)                 │
│  ├── 4x Cameras (480x640)                          │
│  └── USB Connections                                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## ROS2 & Container - Notwendigkeit

### Kurze Antwort: NICHT NÖTIG für Pi0! ✅

**Was du in den Quellen gesehen hast:**

1. **ROS2** - Für andere Trossen Tutorials (MoveIt, Navigation)
   - **Pi0:** Nutzt KEIN ROS2
   - **Pi0:** Direkte Python API über LeRobot

2. **Container/Docker** - Optional, nicht zwingend
   - **Vorteil:** Isolierte Dependencies
   - **Nachteil:** Komplexer Setup
   - **Für Pi0:** Native Installation mit UV ist einfacher

### Was Pi0 wirklich braucht

**Training Server:**
- ✅ openpi (bereits installiert)
- ✅ UV Environment
- ✅ JAX für Inference

**Inference PC:**
- 🆕 openpi Client Environment
- 🆕 LeRobot V0.3.2
- 🆕 PyTorch (nicht JAX!)
- ✅ Bestehende LeRobot Hardware Setup (behältst du)

**Kein ROS2, kein Docker nötig!** 🎉

---

## Training Server: Policy Server Setup

### Voraussetzungen

- ✅ Training abgeschlossen
- ✅ Checkpoint verfügbar
- ✅ Netzwerk-Zugriff von Inference PC

### 4.1 Checkpoint auswählen

```bash
# Auf Training Server (SSH)
cd ~/openpi

# Zeige verfügbare Checkpoints
ls -lh checkpoints/pi0_lighter_cup_trossen/production_v1/

# Wähle besten Checkpoint (z.B. 20000 = final)
CHECKPOINT_DIR=checkpoints/pi0_lighter_cup_trossen/production_v1/20000
```

### 4.2 Policy Server starten

```bash
# Auf Training Server
cd ~/openpi

# Server mit externem Bind für Remote Access
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000
```

**Wichtige Parameter:**
- `--host=0.0.0.0` - Bindet an alle Netzwerk-Interfaces (für Remote Access)
- `--port=8000` - Standard Port (anpassbar falls belegt)

**Erwartete Output:**
```
Loading checkpoint from checkpoints/pi0_lighter_cup_trossen/production_v1/20000
Policy server listening on ws://0.0.0.0:8000
Ready to accept connections...
```

### 4.3 Firewall konfigurieren

```bash
# Auf Training Server
sudo ufw allow 8000/tcp
sudo ufw status

# Verify Port ist offen
ss -tlnp | grep 8000
```

### 4.4 Server IP Address notieren

```bash
# Auf Training Server
hostname -I

# Notiere die IP (z.B. 192.168.1.100)
# Diese brauchst du für Client-Konfiguration
```

### 4.5 Health Check

```bash
# Auf Training Server (in neuem Terminal)
curl http://localhost:8000/health

# Von Inference PC (später):
curl http://192.168.1.100:8000/health

# Expected Response:
# {"status": "healthy", "model_loaded": true}
```

### 4.6 Server im Hintergrund laufen lassen

```bash
# Mit tmux (empfohlen)
tmux new -s policy_server
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000

# Detach: Ctrl+B dann D
# Reattach: tmux attach -t policy_server

# Oder mit nohup:
nohup uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000 > policy_server.log 2>&1 &

# Logs: tail -f policy_server.log
```

---

## Inference PC: Client Setup

### 5.1 System Requirements Check

```bash
# Auf Inference PC
nvidia-smi
# Sollte RTX 4080 16GB zeigen

# Python Version
python --version
# Python 3.10+ empfohlen

# UV installiert?
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 5.2 openpi Client Repository Setup

```bash
# Auf Inference PC
cd ~/
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git openpi_client

cd openpi_client
git submodule update --init --recursive
```

**Wichtig:** Separates Verzeichnis `openpi_client` (nicht `openpi`), falls du lokal auch trainieren willst

### 5.3 Client Environment Installation

```bash
# Auf Inference PC
cd ~/openpi_client/examples/trossen_ai

# Client Environment mit LeRobot V0.3.2
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Verify LeRobot V0.3.2
uv run python -c "import lerobot; print('LeRobot version:', lerobot.__version__)"
# Expected: 0.3.2

# Verify BiWidowXAIFollower verfügbar
uv run python -c "from lerobot.common.robot_devices.robots.biwidowxai_follower import BiWidowXAIFollower; print('✅ BiWidowXAIFollower available')"
```

### 5.4 Hardware Config aus bestehendem Setup übernehmen

```bash
# Auf Inference PC
# Du hast bereits LeRobot für ACT - nutze diese Config!

# Finde deine Hardware Config
cd ~/lerobot
find . -name "*config*" -o -name "*calibration*" | grep -i trossen

# Kopiere relevante Hardware-Parameter
```

**Wichtige Hardware-Informationen:**
- Motor IDs & Ports
- Camera Device IDs
- Joint Limits
- Gripper Limits
- Control Frequency

### 5.5 Client Script erstellen

Erstelle `~/openpi_client/examples/trossen_ai/run_inference_client.py`:

```python
#!/usr/bin/env python3
"""
Pi0 Inference Client für Trossen AI Hardware
Verbindet Trossen Arms mit Policy Server
"""

import asyncio
import numpy as np
from lerobot.common.robot_devices.robots.biwidowxai_follower import BiWidowXAIFollower
import websockets
import json

class Pi0InferenceClient:
    def __init__(self, policy_server_url, robot_config):
        self.policy_url = policy_server_url
        self.robot_config = robot_config
        self.robot = None
        self.websocket = None
        
    async def connect(self):
        """Connect to robot and policy server"""
        # Initialize robot
        self.robot = BiWidowXAIFollower(**self.robot_config)
        self.robot.connect()
        print("✅ Robot connected")
        
        # Connect to policy server
        self.websocket = await websockets.connect(self.policy_url)
        print(f"✅ Connected to policy server: {self.policy_url}")
        
    async def get_observation(self):
        """Get current observation from robot"""
        # Get state
        state = self.robot.get_state()
        
        # Get images from all 4 cameras
        images = {}
        for cam_name in ['cam_high', 'cam_low', 'left_wrist', 'right_wrist']:
            img = self.robot.get_camera_image(cam_name)
            images[cam_name] = img
            
        return {
            'observation': {
                'state': state,
                'images': images
            }
        }
    
    async def infer(self, observation):
        """Send observation to server, receive actions"""
        # Send observation
        await self.websocket.send(json.dumps({
            'type': 'infer',
            'observation': self._serialize_obs(observation)
        }))
        
        # Receive actions
        response = await self.websocket.recv()
        result = json.loads(response)
        
        return result['actions']
    
    def _serialize_obs(self, obs):
        """Serialize observation for WebSocket transmission"""
        # Convert numpy arrays to lists, etc.
        # Implementation details...
        pass
        
    async def run_control_loop(self, control_freq=50):
        """Main control loop"""
        dt = 1.0 / control_freq
        
        try:
            while True:
                loop_start = asyncio.get_event_loop().time()
                
                # Get observation
                obs = await self.get_observation()
                
                # Get actions from policy server
                actions = await self.infer(obs)
                
                # Execute actions
                self.robot.send_action(actions)
                
                # Maintain control frequency
                elapsed = asyncio.get_event_loop().time() - loop_start
                if elapsed < dt:
                    await asyncio.sleep(dt - elapsed)
                else:
                    print(f"⚠️ Warning: Loop too slow ({elapsed*1000:.1f}ms > {dt*1000:.1f}ms)")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            await self.disconnect()
            
    async def disconnect(self):
        """Cleanup"""
        if self.robot:
            self.robot.disconnect()
        if self.websocket:
            await self.websocket.close()
        print("✅ Disconnected")


async def main():
    # Configuration
    POLICY_SERVER_URL = "ws://192.168.1.100:8000"  # <-- ANPASSEN!
    
    ROBOT_CONFIG = {
        # Deine Hardware Config hier
        # Übernehme aus deinem bestehenden LeRobot Setup
        'leader_arms': {
            'main': ...,  # Leader arm config
        },
        'follower_arms': {
            'main': ...,  # Follower arm config
        },
        'cameras': {
            'cam_high': ...,
            'cam_low': ...,
            'left_wrist': ...,
            'right_wrist': ...,
        },
    }
    
    # Create and run client
    client = Pi0InferenceClient(POLICY_SERVER_URL, ROBOT_CONFIG)
    await client.connect()
    await client.run_control_loop(control_freq=50)


if __name__ == "__main__":
    asyncio.run(main())
```

### 5.6 Hardware Config aus LeRobot extrahieren

```bash
# Auf Inference PC
# Dein bestehendes LeRobot Setup hat bereits alles konfiguriert!

cd ~/lerobot

# Finde Robot Config
grep -r "BiWidowXAI\|WidowX" . --include="*.py" --include="*.yaml" | head -20

# Oder schaue in deine ACT Training Configs:
cat configs/your_training_config.yaml
```

**Du brauchst:**
- Motor Port Mappings (ttyDXL_*)
- Camera Device IDs (CAM_*)
- Joint Limits
- Gripper Limits
- Control Parameters

---

## Netzwerk-Konfiguration

### 6.1 Netzwerk-Topologie

**Empfohlenes Setup:**

```
Training Server (192.168.1.100)
    ↓
Gigabit Ethernet Switch
    ↓
Inference PC (192.168.1.101)
```

**Minimale Requirements:**
- Latency: <5ms (für 50Hz control mit ~40ms inference)
- Bandbreith: >100 Mbps (für 4 Kamera-Streams)
- Stable Connection (kein WiFi!)

### 6.2 Latency Test

```bash
# Von Inference PC zu Training Server
ping 192.168.1.100

# Target: <5ms average
# Falls höher:
# - Use wired connection
# - Check network config
# - Disable power saving on network interfaces
```

### 6.3 Bandwidth Test

```bash
# Auf Training Server
iperf3 -s

# Auf Inference PC
iperf3 -c 192.168.1.100

# Target: >500 Mbps for smooth operation
```

### 6.4 WebSocket Connection Test

```bash
# Auf Inference PC (nachdem Policy Server läuft)
python3 -c "
import asyncio
import websockets

async def test():
    uri = 'ws://192.168.1.100:8000'
    async with websockets.connect(uri) as ws:
        print('✅ WebSocket connection successful!')
        
asyncio.run(test())
"
```

---

## Hardware-Integration

### 7.1 Bestehende Calibration nutzen

**Du hast bereits alles kalibriert für ACT!**

```bash
# Auf Inference PC
cd ~/lerobot

# Deine bestehende Calibration:
# - Motor Calibration Files
# - Camera Calibration
# - udev rules (ttyDXL_*, CAM_*)

# Diese BEHALTEN und wiederverwenden!
```

**Kein Re-Calibration nötig!** ✅

### 7.2 Camera Setup überprüfen

```bash
# Auf Inference PC
v4l2-ctl --list-devices

# Check alle 4 Kameras:
ls -l /dev/CAM_*
# Sollte zeigen:
# CAM_HIGH
# CAM_LOW
# CAM_LEFT_WRIST (oder ähnlich)
# CAM_RIGHT_WRIST
```

### 7.3 Motor Setup überprüfen

```bash
# Auf Inference PC
ls -l /dev/ttyDXL_*

# Sollte zeigen (abhängig von deinem Setup):
# ttyDXL_master_left
# ttyDXL_master_right
# ttyDXL_puppet_left
# ttyDXL_puppet_right
```

### 7.4 LeRobot Hardware Test

```bash
# Auf Inference PC
conda activate lerobot  # Deine bestehende env
cd ~/lerobot

# Test ob Hardware noch funktioniert
python -c "
from lerobot.common.robot_devices.robots.biwidowxai_follower import BiWidowXAIFollower

# Quick test (mit deiner Config)
robot = BiWidowXAIFollower(...)
robot.connect()
print('✅ Robot hardware OK!')
robot.disconnect()
"
```

---

## Testing & Validation

### 8.1 Offline Test (ohne Hardware)

**Auf Inference PC:**

```bash
cd ~/openpi_client/examples/trossen_ai

# Test mit Dummy Observations
uv run python test_client.py \
  --policy_url ws://192.168.1.100:8000 \
  --mode dummy

# Sollte random observations senden und actions empfangen
```

### 8.2 Hardware Test (vorsichtig!)

**Safety First Checklist:**
- [ ] Arms in freespace (nichts in Reichweite)
- [ ] Emergency stop bereit (Ctrl+C)
- [ ] Jemand observiert die Arms
- [ ] Joint limits in Config gesetzt
- [ ] Reduzierte Geschwindigkeit zunächst

```bash
# Auf Inference PC
cd ~/openpi_client/examples/trossen_ai

# Start Client
uv run python run_inference_client.py \
  --policy_url ws://192.168.1.100:8000 \
  --control_freq 20  # Start mit niedrigerer Frequenz!

# Beobachte:
# - Smoothe Bewegungen?
# - Korrekte Richtungen?
# - Keine gefährlichen Bewegungen?
```

### 8.3 Performance Benchmarks

```python
# Im Client Script Latency messen:
import time

obs_start = time.time()
obs = await self.get_observation()
obs_time = time.time() - obs_start

infer_start = time.time()
actions = await self.infer(obs)
infer_time = time.time() - infer_start

exec_start = time.time()
self.robot.send_action(actions)
exec_time = time.time() - exec_start

total = obs_time + infer_time + exec_time

print(f"Observation: {obs_time*1000:.1f}ms")
print(f"Inference: {infer_time*1000:.1f}ms")
print(f"Execution: {exec_time*1000:.1f}ms")
print(f"Total: {total*1000:.1f}ms")

# Target für 50Hz (20ms period):
# Total < 20ms → Möglich
# Total 20-50ms → Reduziere auf 20-30Hz
# Total >50ms → Optimize oder niedrigere Frequenz
```

---

## Troubleshooting Inference

### Server-Probleme

**Problem: Server startet nicht**
```bash
# Check Port belegt?
lsof -i :8000

# Kill bestehenden Process
pkill -f serve_policy

# Nutze anderen Port
--port=8080
```

**Problem: Checkpoint lädt nicht**
```bash
# Verify Checkpoint existiert
ls -lh ~/openpi/checkpoints/pi0_lighter_cup_trossen/production_v1/20000/

# Check Permissions
chmod -R 755 ~/openpi/checkpoints/

# Try absolute path
--policy.dir=/home/max/openpi/checkpoints/...
```

### Client-Probleme

**Problem: WebSocket Connection Failed**
```bash
# Check Server läuft
curl http://192.168.1.100:8000/health

# Check Firewall
# Auf Training Server:
sudo ufw status
sudo ufw allow 8000/tcp

# Check Network
ping 192.168.1.100
```

**Problem: Hardware not found**
```bash
# Check udev rules
ls -l /dev/ttyDXL_*
ls -l /dev/CAM_*

# Falls fehlt: Reload udev
sudo udevadm control --reload
sudo udevadm trigger

# Check Permissions
sudo usermod -a -G dialout $USER
# Logout/Login required!
```

**Problem: Hohe Latency**
```bash
# Measure end-to-end
# Observation → Inference → Action → Execute

# Optimize:
# 1. Reduce image resolution (falls möglich)
# 2. Compression für Netzwerk-Transfer
# 3. Batch observations (falls sinnvoll)
# 4. Lower control frequency (50Hz → 20Hz)
```

---

## Optimierungen

### 9.1 Inference Latency reduzieren

**Auf Training Server (Policy Server):**
```python
# In serve_policy.py Config:
num_inference_steps=5  # Statt 10 (DDIM sampling)

# Oder nutze kleineres Model (trade-off quality vs. speed)
```

### 9.2 Network Bandwidth optimieren

```python
# Im Client: Image Compression
import cv2

def compress_image(img):
    # JPEG Compression für Netzwerk-Transfer
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    return encoded

# Oder: Reduziere Resolution
img_resized = cv2.resize(img, (320, 240))  # Statt 640x480
```

### 9.3 Control Loop Optimization

```python
# Asynchrone Image Capture
async def capture_cameras_async():
    tasks = [
        asyncio.to_thread(robot.get_camera_image, 'cam_high'),
        asyncio.to_thread(robot.get_camera_image, 'cam_low'),
        asyncio.to_thread(robot.get_camera_image, 'left_wrist'),
        asyncio.to_thread(robot.get_camera_image, 'right_wrist'),
    ]
    images = await asyncio.gather(*tasks)
    return dict(zip(['cam_high', 'cam_low', 'left_wrist', 'right_wrist'], images))
```

---

## Quick Reference Commands

### Training Server (Policy Server):

```bash
# Start Server
cd ~/openpi
tmux new -s policy_server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lighter_cup_trossen \
  --policy.dir=checkpoints/pi0_lighter_cup_trossen/production_v1/20000 \
  --host=0.0.0.0 \
  --port=8000

# Check Status
curl http://localhost:8000/health
tmux attach -t policy_server
```

### Inference PC (Robot Client):

```bash
# Start Client
cd ~/openpi_client/examples/trossen_ai
uv run python run_inference_client.py \
  --policy_url ws://192.168.1.100:8000 \
  --control_freq 50

# Monitor Performance
# (Add logging to script)
```

---

## Nächste Schritte für neuen Chat

### Zusammenfassung mitnehmen:

1. **Training Phase:** ✅ Abgeschlossen & erfolgreich
   - Checkpoint: `~/openpi/checkpoints/pi0_lighter_cup_trossen/production_v1/20000/`
   - WandB: https://wandb.ai/sourteig-fritsch-gmbh/openpi

2. **Policy Server:** Training Server Setup (Abschnitt 4)

3. **Robot Client:** Inference PC Setup (Abschnitt 5)

4. **Integration:** Netzwerk + Hardware (Abschnitte 6-7)

5. **Testing:** Validation Workflow (Abschnitt 8)

### Implementierungs-Reihenfolge:

1. **Auf Training Server:**
   - Policy Server starten
   - Firewall konfigurieren
   - Health check

2. **Auf Inference PC:**
   - Client Environment installieren
   - Hardware Config übernehmen
   - Client Script erstellen

3. **Netzwerk:**
   - Verbindung testen
   - Latency messen
   - WebSocket funktionsfähig

4. **Hardware:**
   - Vorsichtige erste Tests
   - Performance messen
   - Optimieren

**Geschätzte Zeit:** 1-2 Tage für komplettes Inference Setup

---

## Wichtige Erkenntnisse

**ROS2:** ❌ NICHT nötig  
**Docker:** ❌ NICHT nötig  
**UV:** ✅ Beides Machines nutzen UV  
**LeRobot V0.3.2:** ✅ Nur auf Inference PC  
**Bestehende Hardware Config:** ✅ Wiederverwenden!  

**Viel Erfolg bei der Inference-Phase! 🚀**
