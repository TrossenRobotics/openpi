---
title: "Installation & Environment Setup"
category: setup
tags: [installation, uv, environment, dependencies, setup]
difficulty: beginner
last_updated: 2025-01-08
status: stable
related_docs:
  - 02_hardware_requirements.md
  - 03_git_workflow.md
  - ../concepts/11_uv_vs_conda.md
---

# Installation & Environment Setup

## Zusammenfassung (TL;DR)

OpenPI nutzt **UV** (nicht Conda) für Package Management. Installation dauert ~10-15 Minuten und erstellt ein isoliertes `.venv/` Environment in `~/openpi/`. Ihre bestehenden Conda Environments (lerobot, trossenai) bleiben vollständig unberührt!

**Quick Start:**
```bash
# UV installieren
curl -LsSf https://astral.sh/uv/install.sh | sh

# OpenPI klonen
git clone --recurse-submodules https://github.com/Sourteig/openpi.git ~/openpi
cd ~/openpi

# Environment erstellen
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Verifizieren
uv run python -c "import openpi; print('✅ OpenPI installed!')"
```

---

## Voraussetzungen

### System-Anforderungen

**Minimum:**
- Ubuntu 22.04 (oder kompatibel)
- Python 3.11+ 
- 20 GB freier Speicherplatz
- CUDA 12.x (für GPU Support)
- Git mit LFS

**Empfohlen:**
- Ubuntu 22.04 LTS
- Python 3.11
- 50+ GB freier Speicherplatz
- NVIDIA GPU mit >= 16 GB VRAM
- Git 2.30+

### Vor der Installation prüfen

```bash
# Python Version
python3 --version  # Sollte 3.10+ sein

# CUDA
nvidia-smi  # Sollte GPU(s) zeigen

# Git
git --version  # Sollte 2.30+ sein

# Speicherplatz
df -h ~  # Mindestens 20GB frei
```

---

## Schritt 1: UV Package Manager installieren

### 1.1 Was ist UV?

**UV ist ein moderner Python Package Manager:**
- Schneller als pip/conda (geschrieben in Rust)
- Erstellt isolierte Virtual Environments
- Kompatibel mit pip packages
- Projekt-basiertes Dependency Management

**Wichtig:** UV ≠ Conda! Siehe [concepts/11_uv_vs_conda.md](../concepts/11_uv_vs_conda.md)

### 1.2 UV Installation

```bash
# Installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# Shell neu laden
source ~/.bashrc  # oder ~/.zshrc

# Verifizieren
uv --version
# Sollte zeigen: uv X.Y.Z
```

**Alternative Installation:**
```bash
# Via pip (falls bevorzugt)
pip install uv

# Via cargo (Rust)
cargo install uv
```

---

## Schritt 2: Repository klonen

### 2.1 Von Ihrem Fork klonen

```bash
# In home directory
cd ~

# Klonen mit Submodules
git clone --recurse-submodules https://github.com/Sourteig/openpi.git openpi

# Ins Verzeichnis wechseln
cd openpi

# Submodules prüfen
git submodule status
# Sollte aloha und libero zeigen
```

### 2.2 Falls bereits geklont (ohne submodules)

```bash
cd ~/openpi

# Submodules nachträglich laden
git submodule update --init --recursive

# Verifizieren
ls third_party/
# Sollte aloha/ und libero/ enthalten
```

---

## Schritt 3: Training Environment Setup

### 3.1 Environment erstellen

```bash
cd ~/openpi

# UV sync (erstellt .venv/ und installiert Dependencies)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Package im editable mode installieren
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Was passiert:**
- Erstellt `~/openpi/.venv/` Directory
- Installiert Python 3.11
- Installiert ~240 Packages:
  - JAX 0.5.3 (für Training)
  - PyTorch (für Inference)
  - Transformers, Diffusers
  - LeRobot V0.1.0
  - Alle OpenPI Dependencies

**Dauer:** 5-10 Minuten (abhängig von Internet-Speed)

### 3.2 Warum GIT_LFS_SKIP_SMUDGE=1?

```bash
# OHNE dieses Flag:
# - Git LFS versucht große Dateien zu laden (Checkpoints, Models)
# - Kann sehr langsam sein
# - Nicht nötig für Installation

# MIT diesem Flag:
# - Git LFS überspringt große Dateien
# - Schnellere Installation
# - Dateien werden bei Bedarf später geladen
```

---

## Schritt 4: Installation verifizieren

### 4.1 Python Environment Test

```bash
cd ~/openpi

# Zeige Python Path
uv run python -c "import sys; print(sys.executable)"
# Sollte zeigen: /home/max/openpi/.venv/bin/python

# OpenPI Import Test
uv run python -c "import openpi; print('✅ OpenPI OK')"

# JAX GPU Test
uv run python -c "import jax; print('Devices:', jax.devices())"
# Sollte GPU device zeigen: [CudaDevice(id=0)]
```

### 4.2 Installed Packages Check

```bash
cd ~/openpi

# Package Liste
uv pip list | grep -E "jax|torch|openpi|lerobot"

# Erwartete Ausgabe:
# jax                    0.5.3
# torch                  2.x.x
# openpi                 0.1.0 (editable)
# lerobot                0.1.0
```

### 4.3 Config Load Test

```bash
cd ~/openpi

# Test ob Configs laden
uv run python -c "
from openpi.training import config
cfg = config.get_config('pi0_base')
print(f'✅ Config OK: {cfg.name}')
"
```

---

## Schritt 5: Conda Isolation verifizieren

### 5.1 Ihre Conda Environments sind sicher

```bash
# Liste Conda Environments
conda env list

# Sollte zeigen (UNVERÄNDERT):
# base                 * /home/max/miniconda3
# lerobot                /home/max/miniconda3/envs/lerobot
# trossenai              /home/max/miniconda3/envs/trossenai
# etc...
```

### 5.2 Test: ACT Training noch funktioniert

```bash
# Aktiviere lerobot (Conda)
conda activate lerobot

# Test LeRobot
python -c "import lerobot; print('✅ LeRobot Conda env OK')"

# Zurück zu base
conda deactivate
```

**✅ Alles isoliert! OpenPI (UV) und LeRobot (Conda) koexistieren problemlos.**

---

## Schritt 6: Optionale Komponenten

### 6.1 WandB (Empfohlen für Monitoring)

```bash
cd ~/openpi

# WandB installieren (bereits in Dependencies)
# Login (für Logging)
uv run wandb login

# Geben Sie Ihren API Key ein
# Key finden: https://wandb.ai/settings
```

### 6.2 Git LFS (Für große Checkpoints)

```bash
# Falls noch nicht installiert
sudo apt install git-lfs

# Git LFS aktivieren
cd ~/openpi
git lfs install

# Track große Dateien
git lfs track "*.ckpt"
git lfs track "*.pth"
git lfs track "checkpoints/**/*.npz"
```

### 6.3 Development Tools (Optional)

```bash
cd ~/openpi

# Pre-commit Hooks (Code Quality)
uv pip install pre-commit
uv run pre-commit install

# Jupyter (für Notebooks)
uv pip install jupyter ipykernel
uv run python -m ipykernel install --user --name=openpi
```

---

## Arbeiten mit dem Environment

### UV Commands (statt conda activate)

```bash
# KEIN "conda activate" nötig!
# Nutze "uv run" für Commands:

cd ~/openpi

# Python Script ausführen
uv run python scripts/train.py

# Interactive Python
uv run python

# Jupyter Notebook
uv run jupyter notebook

# Beliebiges Command
uv run <command>
```

### Environment Location

```bash
# Environment befindet sich hier:
ls -la ~/openpi/.venv/

# Struktur:
.venv/
├── bin/
│   └── python  # Python 3.11
├── lib/
│   └── python3.11/
│       └── site-packages/  # Alle Packages
└── pyvenv.cfg
```

---

## Troubleshooting

### Problem: "uv: command not found"

```bash
# Lösung 1: Shell neu laden
source ~/.bashrc

# Lösung 2: PATH manuell setzen
export PATH="$HOME/.local/bin:$PATH"

# Lösung 3: Neu installieren
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Problem: CUDA nicht gefunden

```bash
# Check CUDA Installation
nvcc --version

# Falls fehlt:
# Siehe: https://developer.nvidia.com/cuda-downloads

# Check CUDA Path
echo $CUDA_HOME
# Sollte zeigen: /usr/local/cuda oder ähnlich

# Falls nicht gesetzt:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Problem: "Out of Memory" während Installation

```bash
# Lösung: Weniger parallel Builds
UV_CONCURRENT_BUILDS=1 uv sync

# Oder: Swap erhöhen
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Problem: JAX findet GPU nicht

```bash
cd ~/openpi

# Check JAX Installation
uv run python -c "import jax; print(jax.default_backend())"
# Sollte "gpu" zeigen

# Falls "cpu":
# Reinstall mit GPU Support
uv pip install --upgrade "jax[cuda12]"
```

### Problem: Submodules fehlen

```bash
cd ~/openpi

# Submodules nachträglich laden
git submodule update --init --recursive

# Falls Fehler:
git submodule sync
git submodule update --init --recursive --force
```

---

## Environment Management

### Environment neu erstellen

```bash
cd ~/openpi

# Altes Environment löschen
rm -rf .venv/

# Neu erstellen
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Dependencies updaten

```bash
cd ~/openpi

# Update alle Packages
uv pip install --upgrade -r pyproject.toml

# Oder: Nur specific Package
uv pip install --upgrade jax
```

### Dependencies anzeigen

```bash
cd ~/openpi

# Alle installierten Packages
uv pip list

# Dependency Tree
uv pip show openpi

# Outdated Packages
uv pip list --outdated
```

---

## Nächste Schritte

**Jetzt wo Installation abgeschlossen:**

1. **Hardware überprüfen** → [02_hardware_requirements.md](02_hardware_requirements.md)
2. **UV vs Conda verstehen** → [../concepts/11_uv_vs_conda.md](../concepts/11_uv_vs_conda.md)
3. **Git Workflow einrichten** → [03_git_workflow.md](03_git_workflow.md)
4. **Daten vorbereiten** → [../training/20_data_preparation.md](../training/20_data_preparation.md)

---

## Siehe auch

- [UV Documentation](https://docs.astral.sh/uv/)
- [OpenPI Original README](../../README.md)
- [../reference/41_troubleshooting.md](../reference/41_troubleshooting.md#installation-issues)

---

## Changelog

- **2025-01-08:** Initial Version
- **2025-01-08:** UV Installation Details hinzugefügt
- **2025-01-08:** Troubleshooting Section erweitert
