# Pi0 Quick Start Guide - Sofort loslegen! 🚀

**Zielgruppe:** Du bist bereit, mit Pi0 zu starten  
**Zeit bis zum ersten Training:** ~10 Minuten  
**Erstellt:** 07.01.2025

---

## ✅ Was bereits fertig ist

- ✅ openpi Repository geklont (`~/openpi`)
- ✅ UV Environment installiert (isoliert, Conda unberührt)
- ✅ JAX mit GPU Support funktioniert
- ✅ Deine Camera-Konfiguration analysiert (4 Cameras kompatibel!)
- ✅ Training Config Template ready

**Deine Conda Environments sind SICHER und UNBERÜHRT!** ✅

---

## 🎯 Die nächsten 3 Schritte

### Schritt 1: Training Config erstellen (5 Min)

Kopiere die Config aus `docs/PI0_TRAINING_CONFIG_TEMPLATE.md` in openpi:

```bash
# In neuem Terminal (oder diesem)
cd ~/openpi

# Öffne config.py
nano src/training/config.py
# ODER:
code src/training/config.py
```

**Was hinzufügen:**
- Scrolle ganz nach unten
- Füge die Config aus dem Template ein (siehe `docs/PI0_TRAINING_CONFIG_TEMPLATE.md`)
- Speichern & schließen

**Validation:**
```bash
cd ~/openpi
uv run python -c "from openpi.training import config; cfg = config.get_config('pi0_lighter_cup_trossen'); print('✅ Config OK!')"
```

### Schritt 2: Normalization Stats berechnen (2-5 Min)

```bash
cd ~/openpi

# Berechne Stats für dein Dataset
uv run scripts/compute_norm_stats.py \
  --config-name pi0_lighter_cup_trossen
```

**Was passiert:**
- Lädt `lighter_cup_v2episodes` Dataset
- Berechnet Mean/Std für States & Actions
- Speichert in `~/.cache/openpi/assets/trossen_lighter_cup/norm_stats.npz`

**Completion Check:**
```bash
ls -lh ~/.cache/openpi/assets/trossen_lighter_cup/
# Sollte norm_stats.npz zeigen
```

### Schritt 3: Test Training starten (10-15 Min)

```bash
cd ~/openpi

# Kurzes Test-Training (1k steps)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi0_lighter_cup_test \
  --exp-name=test_run_1 \
  --overwrite
```

**Was du sehen solltest:**
```
Loading checkpoint from gs://openpi-assets/checkpoints/pi0_base/params
Loading dataset: lighter_cup_v2episodes
Batch size: 16
Starting training...
Step 0: loss=1.234
Step 100: loss=0.987
...
```

**Erfolgs-Indikatoren:**
- Loss sinkt kontinuierlich
- GPU Utilization >70% (check mit `nvidia-smi`)
- Checkpoints werden gespeichert

---

## 📊 Monitoring während Training

### Terminal 1: Training läuft
```bash
cd ~/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py...
```

### Terminal 2: GPU Monitor
```bash
watch -n 1 nvidia-smi
```

**Erwartete Werte:**
- GPU Util: 80-95%
- Memory: 40-60GB / 96GB
- Temp: <85°C

### Terminal 3: Checkpoint Check
```bash
watch -n 10 'ls -lh ~/openpi/checkpoints/pi0_lighter_cup_test/test_run_1/'
```

---

## ⚡ Full Training starten (nach erfolgreichem Test)

Falls Test Training erfolgreich war:

```bash
cd ~/openpi

# Production Training (20k steps, ~3-4 Stunden)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi0_lighter_cup_trossen \
  --exp-name=production_v1 \
  --overwrite

# Im Hintergrund (empfohlen für lange Trainings):
nohup uv run scripts/train.py \
  pi0_lighter_cup_trossen \
  --exp-name=production_v1 \
  --overwrite > training.log 2>&1 &

# Logs anschauen:
tail -f training.log
```

---

## 🛡️ Safety Checks

### Vor jedem Training:

```bash
# 1. Richtiges Verzeichnis?
pwd
# Sollte sein: /home/max/openpi

# 2. Richtige Python Environment?
uv run python -c "import sys; print(sys.executable)"
# Sollte sein: /home/max/openpi/.venv/bin/python

# 3. GPU verfügbar?
uv run python -c "import jax; print(jax.devices())"
# Sollte sein: [CudaDevice(id=0)]

# 4. Config existiert?
uv run python -c "from openpi.training import config; config.get_config('pi0_lighter_cup_trossen')"
# Sollte keine Errors werfen
```

### Deine Conda Env check:

```bash
# In separatem Terminal
conda env list
# lerobot, trossenai etc. sollten alle noch da sein ✅

conda activate lerobot
python -c "import lerobot; print('✅ LeRobot Conda env intact!')"
# Sollte funktionieren!
```

---

## 📁 File Structure Übersicht

```
~/lerobot/                           # Dein bestehendes Setup
├── lighter_cup_v2episodes/          # ✅ Dein Dataset
├── .venv/                           # ✅ Unberührt
└── conda: lerobot env               # ✅ Unberührt

~/openpi/                            # Neues Pi0 Setup
├── .venv/                           # UV Environment (Python 3.11)
│   └── LeRobot V0.1.0              # Für Training
├── src/training/config.py           # 📝 Hier Config hinzufügen
├── scripts/
│   ├── compute_norm_stats.py       # Schritt 2
│   └── train.py                    # Schritt 3
└── checkpoints/                     # Training Outputs
    └── pi0_lighter_cup_*/
```

---

## 🎓 Lern-Ressourcen

### Trossen Official Tutorial
https://docs.trossenrobotics.com/trossen_arm/v1.9/tutorials/openpi.html

### Pi0 Paper
https://www.physicalintelligence.company/download/pi0.pdf

### Deine Dokumentation
- `docs/PI0_MIGRATION_PLAN.md` - Gesamter Plan
- `docs/PI0_CAMERA_CONFIG.md` - Deine Camera Config
- `docs/PI0_TRAINING_CONFIG_TEMPLATE.md` - Config Details

---

## ❓ FAQ

### F: Wird mein bestehendes LeRobot/ACT Setup beeinträchtigt?

**A:** Nein! Komplett isoliert:
- openpi nutzt UV (in ~/openpi/.venv)
- LeRobot nutzt Conda (in conda env `lerobot`)
- Keine Überschneidungen

### F: Muss ich Daten neu sammeln?

**A:** Nein! Deine bestehenden LeRobot Datasets funktionieren direkt.

### F: Kann ich parallel ACT und Pi0 trainieren?

**A:** Ja! Verschiedene Environments, verschiedene Verzeichnisse.
- ACT: `conda activate lerobot && cd ~/lerobot`
- Pi0: `cd ~/openpi && uv run ...`

### F: Was wenn ich Fehler bekomme?

**A:** Siehe `docs/PI0_MIGRATION_PLAN.md` Troubleshooting Sektion, oder:
1. Check welches Environment aktiv ist
2. Check welches Verzeichnis (`pwd`)
3. Logs analysieren
4. GPU Memory check (`nvidia-smi`)

### F: Wie lange dauert Training?

**A:** Mit deiner RTX 6000 Pro 96GB:
- Test (1k steps): ~10-15 Min
- Full (20k steps): ~3-4 Stunden
- Extended (40k steps): ~6-8 Stunden

### F: Wo finde ich die Checkpoints?

**A:** `~/openpi/checkpoints/<config_name>/<exp_name>/<step>/`

---

## 🚀 Los geht's!

```bash
# Terminal 1: Training
cd ~/openpi
uv run scripts/train.py pi0_lighter_cup_test --exp-name=first_test --overwrite

# Terminal 2: Monitor
watch -n 1 nvidia-smi

# Terminal 3: Checkpoints
watch -n 10 'ls -lh ~/openpi/checkpoints/*/first_test/'
```

**Viel Erfolg! 🎉**

---

## 📞 Support

Bei Fragen:
1. Trossen Docs durchsuchen
2. openpi GitHub Issues
3. Eigene Notizen in `docs/PI0_TRAINING_LOG.md` machen
