---
title: "UV vs Conda - Package Manager Vergleich"
category: concepts
tags: [uv, conda, environment, package-manager]
difficulty: beginner
last_updated: 2025-01-08
status: stable
related_docs:
  - ../setup/01_installation.md
  - 10_pi0_architecture.md
---

# UV vs Conda - Package Manager Vergleich

## Zusammenfassung (TL;DR)

**OpenPI nutzt UV, NICHT Conda!** UV ist ein moderner Python Package Manager (ähnlich pip), der isolierte Environments in `.venv/` Verzeichnissen erstellt. Ihre bestehenden Conda Environments (lerobot, trossenai) bleiben vollständig unberührt und funktionieren parallel.

**Wichtig:** Kein `conda activate` für OpenPI - nutzen Sie `uv run` stattdessen!

---

## Was ist UV?

### UV Grundlagen

**UV** ist ein moderner Python Package Manager:
- Geschrieben in Rust (extrem schnell)
- Kompatibel mit pip/PyPI packages
- Erstellt projekt-basierte Virtual Environments
- Keine globale Installation von Packages

**Entwickelt von:** Astral (gleiche Firma wie Ruff, uv)  
**Website:** https://docs.astral.sh/uv/

### UV vs Conda - Der Hauptunterschied

```
┌─────────────────────────────────────┐
│  CONDA (Was Sie kennen)             │
├─────────────────────────────────────┤
│  - Global benannte Environments     │
│  - conda env list zeigt alle        │
│  - conda activate <name>            │
│  - Location: ~/miniconda3/envs/     │
│  - Verwaltet Python + Packages      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  UV (Für OpenPI)                    │
├─────────────────────────────────────┤
│  - Projekt-basierte .venv/          │
│  - Nicht in conda env list          │
│  - uv run <command>                 │
│  - Location: ~/openpi/.venv/        │
│  - Verwaltet nur Python Packages    │
└─────────────────────────────────────┘
```

---

## Ihr Setup: Conda UND UV parallel

### Ihr aktueller Stand

**Conda Environments (UNVERÄNDERT):**
```bash
$ conda env list
base                    * /home/max/miniconda3
lerobot                   /home/max/miniconda3/envs/lerobot
trossenai                 /home/max/miniconda3/envs/trossenai
trossen_ai_data_...       /home/max/miniconda3/envs/...
trossen_mujoco_env        /home/max/miniconda3/envs/trossen_mujoco_env
```

**UV Environment (NEU):**
```bash
# NICHT in conda env list!
# Location: ~/openpi/.venv/

$ ls ~/openpi/.venv/
bin/  lib/  pyvenv.cfg
```

**✅ Beide koexistieren problemlos! Keine Konflikte!**

---

## Workflow-Vergleich

### ACT Training (mit Conda)

```bash
# Terminal 1: ACT Training
conda activate lerobot
cd ~/lerobot
python lerobot/scripts/train.py ...

# Environment ist aktiv:
$ which python
/home/max/miniconda3/envs/lerobot/bin/python
```

### Pi0 Training (mit UV)

```bash
# Terminal 2: Pi0 Training
# KEIN conda activate!
cd ~/openpi
uv run scripts/train.py ...

# UV nutzt automatisch .venv/:
$ uv run python -c "import sys; print(sys.executable)"
/home/max/openpi/.venv/bin/python
```

### Parallel nutzen - Kein Problem!

```bash
# Terminal 1: ACT (Conda)
conda activate lerobot
cd ~/lerobot
python scripts/collect_data.py  # Daten sammeln

# Terminal 2: Pi0 (UV)
cd ~/openpi
uv run scripts/train.py ...     # Training

# Terminal 3: System
# Beide laufen parallel, keine Interferenz!
```

---

## Command Vergleich

### Environment aktivieren

| Task | Conda | UV |
|------|-------|-----|
| Environment "aktivieren" | `conda activate lerobot` | ❌ Nicht nötig! |
| Python ausführen | `python script.py` | `uv run python script.py` |
| Script ausführen | `python -m module` | `uv run python -m module` |
| Package installieren | `conda install <pkg>` | `uv pip install <pkg>` |
| Package liste | `conda list` | `uv pip list` |

### Beispiele Side-by-Side

**ACT (Conda):**
```bash
conda activate lerobot
python -c "import lerobot; print('OK')"
pip install numpy
python scripts/train.py
```

**Pi0 (UV):**
```bash
# Kein activate!
uv run python -c "import openpi; print('OK')"
uv pip install numpy
uv run scripts/train.py
```

---

## Environment Isolation

### Wie funktioniert Isolation?

**Conda:**
```bash
$ conda activate lerobot
$ which python
/home/max/miniconda3/envs/lerobot/bin/python

$ python -c "import sys; print(sys.path[0])"
/home/max/miniconda3/envs/lerobot/lib/python3.10/site-packages
```

**UV:**
```bash
$ cd ~/openpi
$ uv run python -c "import sys; print(sys.executable)"
/home/max/openpi/.venv/bin/python

$ uv run python -c "import sys; print(sys.path[0])"
/home/max/openpi/.venv/lib/python3.11/site-packages
```

**Resultat:** Komplett getrennte Namespaces! ✅

---

## Vorteile & Nachteile

### Vorteile von UV

✅ **Schneller:** 10-100x schneller als pip/conda  
✅ **Projekt-basiert:** .venv/ bleibt im Projekt  
✅ **Deterministisch:** Lock-File (uv.lock) für Reproduzierbarkeit  
✅ **Kompatibel:** Nutzt PyPI, funktioniert mit allen pip Packages  
✅ **Keine Aktivierung:** `uv run` ist einfacher  

### Nachteile von UV

❌ **Nur Python:** Keine system packages (gcc, cmake, etc.)  
❌ **Weniger bekannt:** Neuere Technologie  
❌ **Kein Channel System:** Wie conda channels (conda-forge, etc.)  

### Vorteile von Conda

✅ **System Packages:** Kann CUDA, gcc, etc. installieren  
✅ **Mature:** Lange bewährt, viele Guides  
✅ **Channels:** conda-forge, bioconda, etc.  
✅ **Bekannter Workflow:** `conda activate`  

### Nachteile von Conda

❌ **Langsam:** Environment creation dauert lange  
❌ **Groß:** Environments sind 2-5GB  
❌ **Dependency Hell:** Solver kann lange dauern  
❌ **Global:** Environments sind nicht projekt-lokal  

---

## Häufige Missverständnisse

### Missverständnis 1: "UV ist ein Conda-Ersatz"

**Falsch!** UV ersetzt pip, nicht Conda.

```
UV = pip + venv (kombiniert)
Conda = Eigenes Ökosystem

Beide können parallel existieren!
```

### Missverständnis 2: "Ich muss conda deactivate vor uv run"

**Falsch!** `uv run` ignoriert aktive Conda Environments.

```bash
# Selbst wenn conda env aktiv ist:
$ conda activate lerobot
(lerobot) $ cd ~/openpi
(lerobot) $ uv run python -c "import sys; print(sys.executable)"
/home/max/openpi/.venv/bin/python  # Nutzt UV, nicht Conda!
```

### Missverständnis 3: "UV Environment in conda env list"

**Falsch!** UV Environments erscheinen NICHT in `conda env list`.

```bash
$ conda env list
# Zeigt nur: base, lerobot, trossenai, etc.

$ ls ~/openpi/.venv/
# UV Environment ist hier, aber nicht in Conda sichtbar
```

### Missverständnis 4: "UV braucht Conda"

**Falsch!** UV ist komplett unabhängig von Conda.

```bash
# UV funktioniert auch ohne Conda installiert:
$ uv run python --version
Python 3.11.x  # UV managed Python
```

---

## Wann welches Tool?

### Nutze Conda für:

- ✓ ACT Training (Ihre bestehende Setup)
- ✓ LeRobot Data Collection
- ✓ System Dependencies (CUDA, gcc)
- ✓ Wenn Sie conda gewohnt sind
- ✓ Cross-language Projects (Python + R + Julia)

### Nutze UV für:

- ✓ OpenPI / Pi0 Training
- ✓ Neue Python-only Projekte
- ✓ Schnelle Entwicklung
- ✓ CI/CD Pipelines
- ✓ Wenn Performance wichtig ist

---

## Migration: Conda → UV (Optional)

Falls Sie ein Conda Environment in UV umwandeln wollen:

### Schritt 1: Export Conda Requirements

```bash
conda activate lerobot
pip list --format=freeze > requirements.txt
```

### Schritt 2: Create UV Environment

```bash
cd ~/new_project
uv venv
uv pip install -r requirements.txt
```

### Schritt 3: Test

```bash
uv run python -c "import lerobot; print('OK')"
```

**Aber:** Für OpenPI/Pi0 ist dies NICHT nötig! UV Environment existiert bereits.

---

## Troubleshooting

### Problem: "uv: command not found"

```bash
# Lösung: UV neu installieren
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Problem: "Wrong Python version"

```bash
# UV Python Version prüfen
uv run python --version

# Falls falsch: Environment neu erstellen
cd ~/openpi
rm -rf .venv/
uv sync
```

### Problem: "Package not found"

```bash
# UV nutzt PyPI, nicht Conda channels
# Manche Conda-only Packages sind nicht verfügbar

# Lösung: Alternatives Package suchen
# Oder: Diese Aufgabe in Conda Environment erledigen
```

### Problem: "Mixing conda and uv"

```bash
# Dies funktioniert NICHT:
conda activate lerobot
cd ~/openpi
python scripts/train.py  # ❌ Nutzt conda python, nicht uv!

# Stattdessen:
cd ~/openpi
uv run scripts/train.py  # ✅ Korrekt!
```

---

## Quick Reference

### UV Commands

```bash
# Environment erstellen
uv venv

# Sync Dependencies (aus pyproject.toml)
uv sync

# Package installieren
uv pip install <package>

# Package deinstallieren
uv pip uninstall <package>

# Package Liste
uv pip list

# Python ausführen
uv run python <script>

# Command ausführen
uv run <any-command>

# Python Version anzeigen
uv run python --version
```

### Conda Commands (Referenz)

```bash
# Environment erstellen
conda create -n myenv python=3.10

# Environment aktivieren
conda activate myenv

# Package installieren
conda install <package>
# Oder: pip install <package>

# Package Liste
conda list

# Environment Liste
conda env list

# Environment deaktivieren
conda deactivate
```

---

## Beste Praktiken

### DO ✅

```bash
# Nutze uv run für OpenPI
cd ~/openpi
uv run scripts/train.py

# Nutze conda activate für ACT
conda activate lerobot
cd ~/lerobot
python scripts/train.py

# Check welches Python aktiv ist
which python
uv run python -c "import sys; print(sys.executable)"
```

### DON'T ❌

```bash
# Nicht: uv und conda mischen
conda activate lerobot
cd ~/openpi
python scripts/train.py  # ❌ Falsch!

# Nicht: UV packages in Conda installieren
conda activate lerobot
uv pip install something  # ❌ Macht keinen Sinn

# Nicht: Conda Environment in OpenPI verwenden
conda activate lerobot
cd ~/openpi
./scripts/train.py  # ❌ Nutzt falsches Python
```

---

## Zusammenfassung

**UV und Conda können problemlos parallel existieren:**
- ✅ Conda für ACT/LeRobot (bestehender Workflow)
- ✅ UV für OpenPI/Pi0 (neuer Workflow)
- ✅ Keine Konflikte
- ✅ Separate Namespaces

**Key Takeaways:**
1. UV nutzt `.venv/` in Projekt-Verzeichnissen
2. Kein `conda activate` für OpenPI nötig
3. Nutze `uv run` für alle OpenPI Commands
4. Ihre Conda Environments bleiben unberührt

---

## Nächste Schritte

- **Installation abschließen:** [../setup/01_installation.md](../setup/01_installation.md)
- **Pi0 Architektur verstehen:** [10_pi0_architecture.md](10_pi0_architecture.md)
- **Training konfigurieren:** [../training/21_configuration.md](../training/21_configuration.md)

---

## Siehe auch

- [UV Documentation](https://docs.astral.sh/uv/)
- [Conda Documentation](https://docs.conda.io/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [../setup/01_installation.md](../setup/01_installation.md)

---

## Changelog

- **2025-01-08:** Initial Version
- **2025-01-08:** Command Vergleiche erweitert
- **2025-01-08:** Troubleshooting Section hinzugefügt
