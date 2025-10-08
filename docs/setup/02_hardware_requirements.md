---
title: "Hardware-Anforderungen & Optimierung"
category: setup
tags: [hardware, gpu, ram, optimization, rtx]
difficulty: beginner
last_updated: 2025-01-08
status: stable
related_docs:
  - 01_installation.md
  - ../training/21_configuration.md
---

# Hardware-Anforderungen & Optimierung

## Zusammenfassung (TL;DR)

**Minimum:** RTX 4080 16GB, 32GB RAM, 100GB Storage  
**Empfohlen:** RTX 6000 Pro 96GB, 64GB+ RAM, 200GB+ Storage  
**Für Training:** GPU mit ≥24GB VRAM für LoRA Fine-tuning  
**Für Inference:** GPU mit ≥8GB VRAM ausreichend

---

## GPU-Anforderungen

### Für Training

**Minimum (LoRA Fine-tuning):**
- NVIDIA GPU mit ≥24GB VRAM
- CUDA Compute Capability ≥7.0
- CUDA 12.x Unterstützung

**Empfohlen:**
- NVIDIA RTX 6000 Pro (96GB) - Optimal!
- NVIDIA A100 (80GB)
- NVIDIA H100 (80GB)
- NVIDIA RTX 4090 (24GB) - Minimum

**Budget-Optionen:**
- RTX 4080 (16GB) - Nur für sehr kleine Batch Sizes
- RTX 3090 (24GB) - Funktioniert, aber langsamer

### Für Inference

**Minimum:**
- NVIDIA GPU mit ≥8GB VRAM
- CUDA Compute Capability ≥7.0

**Empfohlen:**
- RTX 4080 (16GB) - Gut für lokale Inference
- RTX 4090 (24GB) - Noch besser
- Jede Training-GPU funktioniert auch für Inference

---

## Ihr Setup

### Training Server: RTX 6000 Pro 96GB

**Spezifikationen:**
- VRAM: 96 GB GDDR6
- Architecture: Ada Lovelace (Blackwell)
- CUDA Cores: 18,176
- Tensor Cores: 568 (4th Gen)
- Memory Bandwidth: 1,152 GB/s
- TDP: 300W

**Optimale Konfiguration für Pi0:**

```python
# Training Config Optimierungen
TrainConfig(
    batch_size=32,  # 4x größer als Standard!
    num_workers=8,  # CPU nicht Bottleneck
    # Memory Fraction
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.95  # Nutzt 90GB von 96GB
)
```

**Erwartete Performance:**
- Training Speed: ~10-15 steps/sec (20k steps in 3-4h)
- Memory Usage: 40-60GB (viel Headroom!)
- GPU Utilization: >95%
- Parallel Experiments möglich

**Vorteile vs. Standard-GPUs:**

| Metrik | RTX 4090 (24GB) | RTX 6000 Pro (96GB) | Vorteil |
|--------|----------------|---------------------|---------|
| Batch Size | 8 | 32 | 4x größer |
| Training Time | ~12h | ~3h | 4x schneller |
| Memory Safety | Eng | Viel Headroom | Sicherer |
| Parallel Runs | Nein | Ja (2-3x) | Flexibler |

### Inference PC: RTX 4080 16GB

**Spezifikationen:**
- VRAM: 16 GB GDDR6X
- Architecture: Ada Lovelace
- CUDA Cores: 9,728
- Tensor Cores: 304 (4th Gen)
- Memory Bandwidth: 736 GB/s
- TDP: 320W

**Optimale Konfiguration für Pi0 Inference:**

```python
# Policy Server Config
# Memory Usage: ~6-8GB (viel Reserve)
# Inference Latency: ~30-50ms
# Control Frequency: 30-50Hz möglich
```

**Erwartete Performance:**
- Inference Latency: 30-50ms (DDIM sampling)
- Memory Usage: 6-8GB von 16GB
- GPU Utilization: 40-60% (nicht voll ausgelastet)
- Headroom für andere Tasks

---

## RAM-Anforderungen

### Training

**Minimum:**
- 32 GB RAM
- Mit num_workers=2

**Empfohlen:**
- 64 GB RAM
- Mit num_workers=4-8

**Optimal:**
- 128+ GB RAM
- Mit num_workers=8+
- Parallel Training möglich

**Ihr Setup (Training Server):**
```bash
# Check RAM
free -h
# Total: 113 GB verfügbar
# Optimal für num_workers=8!
```

**RAM Usage Breakdown:**
- System: ~2-5 GB
- Python Environment: ~2 GB
- Data Loading (per worker): ~2-3 GB
- Training Process: ~10-20 GB
- GPU Buffers: ~5-10 GB

**Berechnung für num_workers:**
```
Pro Worker: ~2.5 GB
num_workers=2: 5 GB
num_workers=4: 10 GB
num_workers=8: 20 GB

Verfügbar: 113 GB
→ num_workers=8 problemlos möglich! ✅
```

### Inference

**Minimum:**
- 16 GB RAM

**Empfohlen:**
- 32+ GB RAM

**Ihr Setup (Inference PC):**
- Vermutlich 64GB
- Mehr als ausreichend

---

## Storage-Anforderungen

### Training Server

**Minimum:**
- 100 GB freier Speicherplatz
- SSD empfohlen

**Empfohlen:**
- 200+ GB freier Speicherplatz
- NVMe SSD für schnelleres Data Loading

**Breakdown:**
```
OpenPI Repository: ~5 GB
.venv Environment: ~8 GB
Datasets (cached): ~10-50 GB
Checkpoints: ~20-100 GB (5GB pro Checkpoint bei 20k steps)
WandB Logs: ~5-10 GB
Gesamt: ~50-200 GB
```

**Check aktueller Speicherplatz:**
```bash
cd ~/openpi
df -h .
# Empfohlen: >200GB frei
```

### Inference PC

**Minimum:**
- 50 GB freier Speicherplatz

**Breakdown:**
```
OpenPI Client: ~5 GB
Checkpoints: ~5-10 GB (transferiert von Training Server)
Logs: ~1-2 GB
Gesamt: ~10-20 GB
```

---

## CPU-Anforderungen

### Für Training

**Minimum:**
- 4 Cores / 8 Threads
- Intel i5 oder AMD Ryzen 5

**Empfohlen:**
- 8+ Cores / 16+ Threads
- Intel i7/i9 oder AMD Ryzen 7/9

**Optimal (Ihr Setup):**
- 12+ Cores / 24+ Threads
- Für num_workers=8

**CPU Usage während Training:**
- Base System: ~10-20%
- Data Loading (num_workers=8): ~50-80%
- Gesamt: ~60-100% von einzelnen Cores
- Aber: Gesamt CPU-Last bleibt niedrig (<30%)

### Für Inference

**Minimum:**
- 4 Cores / 8 Threads

**Empfohlen:**
- 6+ Cores / 12+ Threads
- Für smooth Control Loop

---

## Network-Anforderungen

### Remote Inference (Training Server ↔ Inference PC)

**Minimum:**
- 100 Mbit/s LAN
- <10ms Latency

**Empfohlen:**
- 1 Gbit/s (Gigabit) LAN
- <5ms Latency
- Direkte Verbindung (kein WiFi!)

**Ihr Setup:**
```bash
# Von Inference PC:
ping <training_server_ip>
# Target: <5ms

iperf3 -c <training_server_ip>
# Target: >500 Mbit/s
```

**Warum wichtig:**
```
4 Cameras @ 640x480 @ 30 FPS:
= 3.5 MB pro Frame
= 105 MB/s raw

Mit JPEG Compression (80%):
= 21 MB/s = 168 Mbit/s

Minimum LAN: 100 Mbit/s → Zu wenig!
Empfohlen: 1 Gbit/s → Gut! ✅
```

**Siehe auch:** [../inference/32_network.md](../inference/32_network.md)

---

## Batch Size Optimierung

### Für RTX 6000 Pro (96GB)

**Empfohlene Batch Sizes:**

```python
# LoRA Fine-tuning (pi0_lighter_cup_trossen)
batch_size=32  # Optimal! Nutzt ~50GB

# Full Fine-tuning (falls gewünscht)
batch_size=16  # Nutzt ~70-80GB

# Experimental: Noch größer
batch_size=64  # Möglich! Nutzt ~90GB
```

**Berechnung:**
```
Base Memory: ~20 GB (Model + Optimizer)
Pro Batch Item: ~1-1.5 GB

batch_size=8:  ~30 GB → Verschwendung auf 96GB GPU!
batch_size=16: ~40 GB → Immer noch viel frei
batch_size=32: ~50 GB → Optimal ✅
batch_size=64: ~90 GB → Möglich, aber eng
```

**Trade-offs:**

| Batch Size | Memory | Speed | Convergence | Empfehlung |
|------------|--------|-------|-------------|------------|
| 8 | 30GB | Slow | Good | ❌ GPU nicht ausgelastet |
| 16 | 40GB | Medium | Good | ✓ OK, aber nicht optimal |
| 32 | 50GB | Fast | Better | ✅ Optimal! |
| 64 | 90GB | Fastest | Best | ⚠️ Eng, aber möglich |

### Für RTX 4080 (16GB) - Falls Training lokal

```python
# LoRA Fine-tuning
batch_size=4  # Sicher
batch_size=8  # Limit, könnte OOM

# Nicht empfohlen für volle Trainings auf 16GB!
```

---

## Memory Fraction Tuning

### XLA_PYTHON_CLIENT_MEM_FRACTION

**Was ist das:**
- Kontrolliert wie viel GPU Memory JAX nutzen darf
- Standard: 0.75 (75%)
- Verhindert OOM bei Multi-Process

**Empfohlene Werte:**

```bash
# RTX 6000 Pro (96GB) - Viel Headroom
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95  # 90GB

# RTX 4090 (24GB) - Wenig Headroom
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90  # 21.6GB

# RTX 4080 (16GB) - Sehr eng
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # 13.6GB
```

**Im Training Command:**
```bash
cd ~/openpi

# Training starten
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi0_lighter_cup_trossen \
  --exp-name=production_v1 \
  --overwrite
```

---

## Monitoring

### GPU Monitoring

```bash
# Real-time Monitoring
watch -n 1 nvidia-smi

# Oder: Detaillierter
nvidia-smi dmon -s pucvmet -d 2

# Log GPU Stats
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 10 > gpu_log.csv
```

**Wichtige Metriken:**
- **GPU Util:** Target >90% während Training
- **Memory:** Sollte nicht >95% sein (OOM Risk)
- **Temp:** Sollte <85°C bleiben
- **Power:** Sollte nahe TDP sein (volle Auslastung)

### RAM Monitoring

```bash
# Real-time
watch -n 2 free -h

# Mit Details
htop

# Oder: top
top
```

### CPU Monitoring

```bash
# CPU Usage
mpstat 1

# Per-Core Usage
htop
# Drücke 't' für Thread-View
```

---

## Optimierungsstrategien

### Strategie 1: Maximize GPU Utilization

```python
# Problem: GPU nur 60% ausgelastet
# Lösung: Erhöhe batch_size

batch_size=32  # Von 16
num_workers=8  # Von 4
```

### Strategie 2: Reduce Training Time

```python
# Nutze größere Batch Size
batch_size=32

# Mehr Data Loader Workers
num_workers=8

# Höheres Memory Fraction
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Result: 3-4x schnelleres Training! ✅
```

### Strategie 3: Memory-Constrained Training

```python
# Falls OOM:
batch_size=16  # Reduzieren
num_workers=4  # Reduzieren
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # Konservativer
```

### Strategie 4: Parallel Experiments

**Nur mit RTX 6000 Pro (96GB) möglich:**

```bash
# Terminal 1: Experiment A
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py config_A

# Terminal 2: Experiment B
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py config_B

# Beide nutzen je ~43GB, gesamt ~86GB von 96GB ✅
```

---

## Hardware-Upgrade Empfehlungen

### Wenn Training zu langsam

**Upgrade-Path:**
1. RTX 4080 (16GB) → RTX 4090 (24GB) - 2x batch size
2. RTX 4090 (24GB) → RTX 6000 Ada (48GB) - 2x batch size again
3. RTX 6000 Ada (48GB) → RTX 6000 Pro (96GB) - Parallel runs möglich

### Wenn Inference zu langsam

**Optimierungen statt Upgrade:**
- Reduce DDIM sampling steps (10 → 5)
- Lower control frequency (50Hz → 30Hz)
- Image compression für Network-Transfer

**Falls Upgrade nötig:**
- RTX 4060 (8GB) → RTX 4080 (16GB)
- Aber: RTX 4080 ist bereits sehr gut für Inference

---

## Kosten-Nutzen Analyse

### GPU Costs (ungefähr)

| GPU | Preis | VRAM | Training Speed | Empfehlung |
|-----|-------|------|----------------|------------|
| RTX 4080 | €1.200 | 16GB | Slow | ⚠️ Zu wenig für Training |
| RTX 4090 | €1.800 | 24GB | Medium | ✓ Minimum für Training |
| RTX 6000 Ada | €6.800 | 48GB | Fast | ✓✓ Gut für Training |
| RTX 6000 Pro | €12.000 | 96GB | Very Fast | ✅ Optimal! (Ihr Setup) |
| A100 | €10.000+ | 80GB | Very Fast | ✓✓ Server-GPU |

**ROI Berechnung:**
```
Zeit gespart pro Training Run:
RTX 4090: 12 Stunden
RTX 6000 Pro: 3 Stunden
Ersparnis: 9 Stunden

Bei 10 Experiments: 90 Stunden = 3.75 Tage gespart!
```

---

## Nächste Schritte

1. **Hardware überprüfen:** `nvidia-smi`, `free -h`, `df -h`
2. **Optimierungen anwenden:** batch_size und num_workers anpassen
3. **Monitoring einrichten:** WandB + nvidia-smi logging
4. **Training starten:** [../training/22_training_execution.md](../training/22_training_execution.md)

---

## Siehe auch

- [01_installation.md](01_installation.md) - Environment Setup
- [../training/21_configuration.md](../training/21_configuration.md) - Training Config
- [../training/23_monitoring.md](../training/23_monitoring.md) - Performance Monitoring

---

## Changelog

- **2025-01-08:** Initial Version
- **2025-01-08:** RTX 6000 Pro Optimierungen hinzugefügt
- **2025-01-08:** Batch Size Kalkulator ergänzt
