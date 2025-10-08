# Pi0 Training Config Template - Dein Setup

**Für Dataset:** `lighter_cup_v2episodes` (92 Episodes)  
**Hardware:** RTX 6000 Pro 96GB  
**Erstellt:** 07.01.2025

---

## 📋 Vollständige Training Config

### Datei: `~/openpi/src/training/config.py`

Füge diese Config in die Liste der TrainConfigs ein:

```python
from openpi.training.data import AssetsConfig
import openpi.shared.transforms as _transforms
from openpi.models import pi0
from openpi.training import weight_loaders

# Füge diese Config zu den bestehenden Configs hinzu:
TrainConfig(
    name="pi0_lighter_cup_trossen",  # Dein Custom Config Name
    
    # Model: LoRA Fine-tuning (nur bestimmte Layer werden trainiert)
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",       # Vision-Language Encoder
        action_expert_variant="gemma_300m_lora"  # Action Decoder
    ),
    
    # Data Configuration
    data=LeRobotAlohaDataConfig(
        # ANPASSEN: Dein Dataset (lokal oder HuggingFace)
        repo_id="lighter_cup_v2episodes",  # Lokal: nur Folder-Name
        # ODER: "dein-username/lighter-cup-v2" wenn auf HuggingFace
        
        # Aloha-kompatible Settings
        use_delta_joint_actions=False,  # Absolute positions (empfohlen)
        adapt_to_pi=False,              # Trossen != Pi internal runtime
        
        # Asset Config (für Normalization Stats)
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen_lighter_cup",  # Unique ID für dein Experiment
        ),
        
        # Task Description (Language Conditioning)
        default_prompt="pick up the lighter and place it in the cup",
        
        # KRITISCH: Camera Mapping
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        # Pi0 Name → Dein Dataset Name
                        "cam_high": "observation.images.cam_high",
                        "cam_low": "observation.images.cam_low",
                        "cam_left_wrist": "observation.images.left_wrist",
                        "cam_right_wrist": "observation.images.right_wrist",
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
    
    # Training Hyperparameters - OPTIMIERT für RTX 6000 Pro 96GB
    num_train_steps=20_000,     # Trossen Empfehlung
    batch_size=32,              # 4x größer als Standard (dank 96GB VRAM!)
    
    # Freeze all except LoRA layers (schnelleres Training)
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    
    # No EMA (Trossen Empfehlung)
    ema_decay=None,
    
    # Optional: Checkpoint Interval
    save_interval=5_000,  # Speichere alle 5k steps
)
```

---

## 🎯 Config Variations

### Variante 1: Schnelles Test-Training

Für erste Tests mit kleinemDataset:

```python
TrainConfig(
    name="pi0_lighter_cup_test",
    # ... alles wie oben ...
    num_train_steps=1_000,      # Nur 1k steps
    batch_size=16,              # Kleinere batch size
    save_interval=500,          # Häufigere checkpoints
)
```

### Variante 2: Longer Training

Für mehr Daten oder bessere Performance:

```python
TrainConfig(
    name="pi0_lighter_cup_extended",
    # ... alles wie oben ...
    num_train_steps=40_000,     # Doppelt so lange
    batch_size=32,              # Volle Power
    save_interval=10_000,       # Weniger häufige checkpoints
)
```

### Variante 3: Full Fine-tuning (statt LoRA)

**NUR mit 96GB GPU möglich:**

```python
TrainConfig(
    name="pi0_lighter_cup_full",
    # Model OHNE LoRA
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b",       # Kein "_lora"
        action_expert_variant="gemma_300m"  # Kein "_lora"
    ),
    # ... rest wie oben ...
    batch_size=16,              # Kleinere batch size (braucht mehr Memory)
    
    # Keine freeze_filter (alle Parameter werden trainiert)
    freeze_filter=None,
)
```

---

## 📂 Wo die Config speichern?

### Option 1: Direkt in openpi/src/training/config.py

```bash
# Öffne die Datei
code ~/openpi/src/training/config.py

# Scrolle zu den anderen TrainConfig Definitionen
# Füge deine Config hinzu (siehe oben)
# Speichern!
```

### Option 2: Separate Config-Datei (sauberer)

```bash
# Erstelle neue Datei
touch ~/openpi/src/training/trossen_configs.py

# Füge Config ein:
```

```python
# ~/openpi/src/training/trossen_configs.py

from openpi.training.config import TrainConfig
from openpi.training.data import LeRobotAlohaDataConfig, AssetsConfig
import openpi.shared.transforms as _transforms
from openpi.models import pi0
from openpi.training import weight_loaders

# Deine Custom Config hier einfügen...
# (siehe oben)
```

```bash
# Dann in config.py importieren:
# from openpi.training.trossen_configs import *
```

---

## ✅ Config Validation

Nach dem Speichern testen:

```bash
cd ~/openpi

# Verify Config lädt
uv run python -c "
from openpi.training import config
cfg = config.get_config('pi0_lighter_cup_trossen')
print('✅ Config loaded successfully!')
print(f'Dataset: {cfg.data.repo_id}')
print(f'Batch size: {cfg.batch_size}')
print(f'Steps: {cfg.num_train_steps}')
"
```

**Erwartete Output:**
```
✅ Config loaded successfully!
Dataset: lighter_cup_v2episodes
Batch size: 32
Steps: 20000
```

---

## 🚀 Nächste Schritte

### 1. Normalization Stats berechnen

```bash
cd ~/openpi

# Compute normalization statistics
uv run scripts/compute_norm_stats.py \
  --config-name pi0_lighter_cup_trossen

# Stats werden gespeichert in:
# ~/.cache/openpi/assets/trossen_lighter_cup/norm_stats.npz
```

**Was passiert:**
- Lädt dein Dataset
- Berechnet Mean/Std für State und Action
- Speichert Stats für Training

**Dauer:** ~2-5 Minuten (abhängig von Dataset-Größe)

### 2. Test Training

```bash
cd ~/openpi

# Erstelle Test Config (1k steps) oder nutze pi0_lighter_cup_test
# Dann:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi0_lighter_cup_test \
  --exp-name=first_test \
  --overwrite

# Mit vollem 96GB Power:
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

**Erwartete Dauer:** ~10-15 Minuten für 1k steps

### 3. Full Training

```bash
cd ~/openpi

# Full Training mit optimierten Settings
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi0_lighter_cup_trossen \
  --exp-name=production_v1 \
  --overwrite

# Läuft im Hintergrund mit tmux/screen empfohlen
```

**Erwartete Dauer:** ~3-5 Stunden für 20k steps (RTX 6000 Pro)

---

## 🔍 Monitoring

### WandB (optional aber empfohlen)

```bash
# Falls noch nicht konfiguriert:
wandb login

# Training logged automatisch zu WandB
# Dashboard: https://wandb.ai/<your-username>/openpi-training
```

### GPU Monitoring

```bash
# In separatem Terminal:
watch -n 1 nvidia-smi

# Check:
# - GPU Utilization >80%
# - Memory Usage
# - Temperature
```

### Loss Tracking

```bash
# Checkpoints werden gespeichert in:
ls -lh ~/openpi/checkpoints/pi0_lighter_cup_trossen/production_v1/

# Structure:
# checkpoints/
#   └── pi0_lighter_cup_trossen/
#       └── production_v1/
#           ├── 5000/
#           ├── 10000/
#           ├── 15000/
#           └── 20000/
```

---

## ⚠️ Troubleshooting

### Problem: Config lädt nicht

```bash
# Check Syntax
cd ~/openpi
uv run python -c "from openpi.training import config"

# Falls Fehler: Syntax-Fehler in config.py
# Prüfe Klammern, Kommas, Einrückung
```

### Problem: Dataset nicht gefunden

```bash
# Falls Dataset lokal:
# Stelle sicher es ist in ~/lerobot/lighter_cup_v2episodes/

# Falls auf HuggingFace:
# Nutze vollständigen Namen: "username/repo-name"
# Authentifizierung: huggingface-cli login
```

### Problem: OOM während Training

```bash
# Reduziere batch_size:
batch_size=16,  # statt 32
# oder:
batch_size=8,   # falls 16 auch zu viel
```

### Problem: Training sehr langsam

```bash
# Check GPU Utilization
nvidia-smi

# Falls niedrig (<70%):
# - Erhöhe batch_size
# - Check data loading nicht bottleneck
```

---

## 📊 Expected Performance

### Mit deinem Setup (RTX 6000 Pro, 92 Episodes, Batch 32)

**Training:**
- Steps/sec: ~2-3
- Time per 1k steps: ~5-8 Minuten
- Total für 20k steps: ~3-4 Stunden

**Memory Usage:**
- Expected: 40-60GB VRAM (von 96GB)
- Headroom: 35-55GB für andere Tasks

**Loss Expectations:**
- Initial: ~1.0-2.0
- After 5k: ~0.5-1.0
- After 10k: ~0.3-0.7
- After 20k: ~0.1-0.5

---

## 🎯 Quick Reference

```bash
# 1. Config erstellen/bearbeiten
code ~/openpi/src/training/config.py

# 2. Normalization Stats
cd ~/openpi && uv run scripts/compute_norm_stats.py --config-name pi0_lighter_cup_trossen

# 3. Test Training (1k steps, ~15 min)
cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_lighter_cup_test --exp-name=test1 --overwrite

# 4. Full Training (20k steps, ~3-4h)
cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_lighter_cup_trossen --exp-name=prod_v1 --overwrite
```

---

## 📝 Nächste Dokumentation

Nach erfolgreichem Training siehe:
- `docs/PI0_MIGRATION_PLAN.md` - Phase 5: Inference Setup
- `docs/PI0_SETUP_STATUS.md` - Aktuelle Status Updates
