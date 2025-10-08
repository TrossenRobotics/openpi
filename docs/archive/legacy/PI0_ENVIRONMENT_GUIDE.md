# Pi0 Environment Guide - UV vs. Conda

**Wichtige Klarstellung:** Pi0 nutzt KEIN Conda! ⚠️

---

## 🔄 UV statt Conda

### Was ist UV?

**UV ist ein moderner Python Package Manager** (Alternative zu Conda/pip):
- Erstellt isolierte Virtual Environments (`.venv/` Verzeichnisse)
- Schneller als Conda
- Projekt-basiert (nicht global)
- Nutzt `uv run` statt `conda activate`

### UV vs. Conda - Der Unterschied

| Aspekt | Conda | UV (Pi0) |
|--------|-------|----------|
| **Environment Type** | Named envs (`conda env list`) | Projekt `.venv/` Verzeichnis |
| **Aktivierung** | `conda activate <name>` | `uv run <command>` |
| **Location** | `~/miniconda3/envs/<name>/` | `~/openpi/.venv/` |
| **Isolation** | Global verfügbar | Nur im Projekt-Verzeichnis |
| **Commands** | `python script.py` | `uv run python script.py` |

---

## 🎯 Dein Setup

### Conda Environments (UNVERÄNDERT):

```bash
conda env list

# Ausgabe:
# base                 * /home/max/miniconda3
# lerobot                /home/max/miniconda3/envs/lerobot
# trossenai              /home/max/miniconda3/envs/trossenai
# trossen_ai_data_collection_ui_env  /home/max/miniconda3/envs/...
# trossen_mujoco_env     /home/max/miniconda3/envs/trossen_mujoco_env

# ✅ Alle unberührt!
```

### UV Environment (NEU):

```bash
# NICHT in conda env list!
# Location: ~/openpi/.venv/

ls -la ~/openpi/.venv/
# drwxr-xr-x  .venv/
#   ├── bin/
#   │   └── python  # Python 3.11
#   ├── lib/
#   │   └── python3.11/
#   │       └── site-packages/
#   │           ├── jax/
#   │           ├── torch/
#   │           ├── openpi/
#   │           └── ...
```

---

## 💻 Wie arbeiten mit UV?

### Du brauchst KEIN `conda activate` für Pi0!

**Falsch (funktioniert nicht):**
```bash
conda activate openpi  # ❌ Existiert nicht!
```

**Richtig:**
```bash
# Einfach uv run nutzen:
cd ~/openpi
uv run python -c "import openpi"
uv run scripts/train.py ...
```

### Workflow-Beispiele

#### Für ACT (Conda):
```bash
conda activate lerobot
cd ~/lerobot
python scripts/train.py ...
```

#### Für Pi0 (UV):
```bash
# KEIN conda activate!
cd ~/openpi
uv run scripts/train.py ...
```

---

## 🔍 Welches Environment bin ich gerade?

### Check Current Environment:

```bash
# Zeige aktives Conda env (wenn vorhanden)
echo $CONDA_DEFAULT_ENV

# Zeige Python Path
which python

# Wenn in Conda env (z.B. lerobot):
# Output: /home/max/miniconda3/envs/lerobot/bin/python

# Wenn UV nutzen willst:
cd ~/openpi
uv run python -c "import sys; print(sys.executable)"
# Output: /home/max/openpi/.venv/bin/python
```

---

## ⚙️ Wie UV funktioniert

### Automatische Environment Detection:

```bash
cd ~/openpi

# UV schaut automatisch nach .venv/ im aktuellen Verzeichnis
uv run python ...
# ↓
# Nutzt ~/openpi/.venv/bin/python

cd ~/some_other_project
uv run python ...
# ↓
# Würde ~/some_other_project/.venv/ nutzen (falls vorhanden)
```

### Manuelles Environment angeben:

```bash
# Falls du explizit sein willst:
cd ~/openpi
uv run --python .venv/bin/python scripts/train.py ...

# Oder von überall:
cd ~
uv run --directory ~/openpi scripts/train.py ...
```

---

## 🛡️ Isolation zwischen Projekten

```
📁 ~/lerobot/
  ├── .venv/  (optional, falls du hier auch venv nutzt)
  └── 🐍 Conda: lerobot environment
      ↓
      Nutzt: conda activate lerobot
      
📁 ~/openpi/
  ├── .venv/  (UV managed)
  └── 🚫 KEIN Conda environment
      ↓
      Nutzt: uv run ...
```

**Komplett isoliert! Keine Konflikte möglich!** ✅

---

## 📋 Quick Reference

### ACT Training (Conda):
```bash
conda activate lerobot
cd ~/lerobot
python src/train_val.py ...
```

### Pi0 Training (UV):
```bash
# KEIN conda activate!
cd ~/openpi
uv run scripts/train.py pi0_lighter_cup_test --exp-name=test1
```

### Beide parallel möglich:
```bash
# Terminal 1: ACT
conda activate lerobot && cd ~/lerobot && python src/train_val.py ...

# Terminal 2: Pi0
cd ~/openpi && uv run scripts/train.py ...
```

---

## ❓ FAQ

### F: Muss ich eine neue Conda Umgebung erstellen?

**A:** **NEIN!** UV nutzt kein Conda. Nutze einfach `uv run` Befehle.

### F: Wie deaktiviere ich die UV Environment?

**A:** Es gibt nichts zu deaktivieren! `uv run` nutzt das Environment nur für diesen einen Befehl.

### F: Kann ich conda und UV mischen?

**A:** Ja, aber besser getrennt halten:
- ACT/LeRobot: Nutze Conda
- Pi0/openpi: Nutze UV
- Verschiedene Terminals für verschiedene Projekte

### F: Wo sind die UV Packages installiert?

**A:** `~/openpi/.venv/lib/python3.11/site-packages/`

### F: Kann ich das UV Environment löschen?

**A:** Ja! Einfach `rm -rf ~/openpi/.venv/` und neu erstellen mit `uv sync`

---

## ✅ Zusammenfassung

**Es gibt KEINE neue Conda Umgebung für Pi0!**

**Stattdessen:**
- UV verwaltet ein `.venv/` Verzeichnis in `~/openpi/`
- Nutze `uv run` statt `conda activate`
- Komplett isoliert von deinen Conda Environments
- Einfacher Workflow: `cd ~/openpi && uv run ...`

**Deine Conda Environments bleiben wie sie sind!** ✅

---

## 🚀 Nächste Schritte

```bash
# 1. Config testen (Single quotes wegen Bash)
cd ~/openpi
uv run python -c 'from openpi.training import config; print(config.get_config("pi0_lighter_cup_test").name)'

# 2. Norm Stats berechnen
uv run scripts/compute_norm_stats.py --config-name pi0_lighter_cup_test

# 3. Training starten
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_lighter_cup_test --exp-name=test1 --overwrite
```

**Keine conda activate nötig! Einfach `uv run` nutzen.** ✅
