---
title: "Git Workflow - Fork Management"
category: setup
tags: [git, fork, repository, version-control]
difficulty: beginner
last_updated: 2025-01-08
status: stable
related_docs:
  - 01_installation.md
  - ../concepts/10_pi0_architecture.md
---

# Git Workflow - Fork Management

## Zusammenfassung (TL;DR)

Ihr OpenPI-Projekt läuft in einem Fork von TrossenRobotics/openpi. Alle Ihre Änderungen (Dokumentation, Code, Configs) werden in Ihrem eigenen Repository https://github.com/Sourteig/openpi.git gespeichert, während Sie gelegentlich Updates vom Original-Repository ziehen können.

**Wichtig:** Ab jetzt laufen ALLE Änderungen in Ihren Fork, nicht mehr ins Hersteller-Repository.

---

## Voraussetzungen

- Git installiert (`git --version`)
- GitHub Account (Sourteig)
- SSH-Key für GitHub konfiguriert (empfohlen)

---

## Aktuelle Situation

### Was Sie haben

```bash
Location: ~/openpi/
Origin: https://github.com/TrossenRobotics/openpi.git
Branch: main (oder feature branches)
Status: Lokale Änderungen an Dokumentation
```

### Was wir aufbauen

```
┌─────────────────────────────────────────┐
│  Physical-Intelligence/openpi           │
│  (Upstream Original)                    │
└──────────────┬──────────────────────────┘
               │ fork
               ↓
┌─────────────────────────────────────────┐
│  TrossenRobotics/openpi                 │
│  (Hersteller Fork)                      │
└──────────────┬──────────────────────────┘
               │ fork
               ↓
┌─────────────────────────────────────────┐
│  Sourteig/openpi                        │
│  (Ihr Fork - IHRE Änderungen!)          │
└──────────────┬──────────────────────────┘
               │ clone
               ↓
┌─────────────────────────────────────────┐
│  ~/openpi/                              │
│  (Lokales Working Directory)            │
└─────────────────────────────────────────┘
```

---

## Schritt 1: Fork auf GitHub erstellen

### 1.1 Fork erstellen

1. Gehen Sie zu: https://github.com/TrossenRobotics/openpi
2. Klicken Sie auf "Fork" (oben rechts)
3. Owner: Sourteig
4. Repository Name: openpi (beibehalten)
5. Description: "My customized Pi0 training setup for Trossen AI"
6. **Wichtig:** Haken bei "Copy the main branch only" ✓
7. Klicken Sie "Create fork"

Ihr Fork ist jetzt verfügbar unter: `https://github.com/Sourteig/openpi`

### 1.2 Fork-Einstellungen

**Optional aber empfohlen:**

```
Settings → General:
- [x] Issues aktivieren (für Ihre eigenen Notizen)
- [ ] Wiki deaktivieren (nutzen Sie docs/)
- [x] Discussions aktivieren (optional)

Settings → Branches:
- Default branch: main
- Branch protection rules: Später einrichten
```

---

## Schritt 2: Lokales Repository umkonfigurieren

### 2.1 Aktuellen Status sichern

```bash
cd ~/openpi

# Aktuellen Status ansehen
git status

# Falls uncommitted Änderungen vorhanden:
git add docs/
git commit -m "docs: restructured documentation for RAG optimization"

# Aktuellen Remote ansehen
git remote -v
# Sollte zeigen:
# origin  https://github.com/TrossenRobotics/openpi.git (fetch)
# origin  https://github.com/TrossenRobotics/openpi.git (push)
```

### 2.2 Remote auf Ihren Fork umstellen

```bash
cd ~/openpi

# Origin auf Ihren Fork ändern
git remote set-url origin https://github.com/Sourteig/openpi.git

# TrossenRobotics als "upstream" hinzufügen
git remote add upstream https://github.com/TrossenRobotics/openpi.git

# Verify
git remote -v
# Sollte jetzt zeigen:
# origin    https://github.com/Sourteig/openpi.git (fetch)
# origin    https://github.com/Sourteig/openpi.git (push)
# upstream  https://github.com/TrossenRobotics/openpi.git (fetch)
# upstream  https://github.com/TrossenRobotics/openpi.git (push)
```

### 2.3 Ersten Push zu Ihrem Fork

```bash
cd ~/openpi

# Aktuellen Branch ansehen
git branch
# * main (oder anderer Branch)

# Push zu Ihrem Fork
git push -u origin main

# Falls Fehler "rejected":
git push -u origin main --force-with-lease  # Vorsichtig!
```

**✅ Erfolg:** Ihre Änderungen sind jetzt in `https://github.com/Sourteig/openpi`

---

## Schritt 3: Workflow für tägliche Arbeit

### 3.1 Feature Branch Workflow (Empfohlen)

**Für jede neue Funktion/Änderung einen eigenen Branch:**

```bash
cd ~/openpi

# Neuer Feature Branch
git checkout -b feature/neue-dokumentation
# Oder: git checkout -b fix/training-bug
# Oder: git checkout -b experiment/neue-config

# Arbeiten...
# Dateien ändern, erstellen, etc.

# Änderungen committen
git add .
git commit -m "docs: add new training guide"

# Zu GitHub pushen
git push -u origin feature/neue-dokumentation

# Wenn fertig: Pull Request auf GitHub erstellen
# (von feature/neue-dokumentation nach main)
```

### 3.2 Commit-Konventionen

**Nutzen Sie Conventional Commits:**

```bash
# Format: <type>(<scope>): <subject>

# Beispiele:
git commit -m "docs: restructure training documentation"
git commit -m "feat: add camera calibration script"
git commit -m "fix: correct normalization stats calculation"
git commit -m "refactor: simplify config structure"
git commit -m "test: add unit tests for data loader"
git commit -m "chore: update dependencies"

# Types:
# feat:     Neue Feature
# fix:      Bug Fix
# docs:     Dokumentation
# style:    Formatierung (kein Code-Change)
# refactor: Code-Umstrukturierung
# test:     Tests hinzufügen
# chore:    Build/Tools/Dependencies
```

### 3.3 Änderungen zusammenführen

```bash
# Zurück zu main
git checkout main

# Feature Branch mergen
git merge feature/neue-dokumentation

# Push zu GitHub
git push origin main

# Optional: Feature Branch löschen
git branch -d feature/neue-dokumentation
git push origin --delete feature/neue-dokumentation
```

---

## Schritt 4: Updates vom Upstream holen

### 4.1 Upstream Updates checken

```bash
cd ~/openpi

# Upstream Updates fetchen
git fetch upstream

# Änderungen ansehen
git log main..upstream/main

# Oder kompakt:
git log --oneline --graph --all
```

### 4.2 Upstream Changes mergen

**Option 1: Rebase (sauberer):**
```bash
cd ~/openpi

# Sicherstellen dass main aktuell ist
git checkout main
git pull origin main

# Upstream changes fetchen
git fetch upstream

# Rebase auf upstream/main
git rebase upstream/main

# Push (force required nach rebase)
git push origin main --force-with-lease
```

**Option 2: Merge (einfacher):**
```bash
cd ~/openpi
git checkout main
git pull origin main
git fetch upstream
git merge upstream/main
git push origin main
```

**Wann welche Option?**
- **Rebase:** Wenn Sie saubere, lineare History wollen
- **Merge:** Wenn Sie Merge-Commits bevorzugen (sicherer)

### 4.3 Konflikte lösen

Falls Konflikte auftreten:

```bash
# Git zeigt Konflikte an
git status

# Konflikte manuell in Dateien lösen
# (Suchen Sie nach <<<<<<< und >>>>>>>)

# Nach dem Lösen:
git add <gelöste-datei>
git rebase --continue  # Falls rebase
# ODER
git merge --continue   # Falls merge

# Falls alles schiefgeht:
git rebase --abort  # Rebase abbrechen
git merge --abort   # Merge abbrechen
```

---

## Schritt 5: .gitignore anpassen

### 5.1 Ihre spezifischen Ignorierungen

```bash
cd ~/openpi
nano .gitignore  # oder code .gitignore
```

**Fügen Sie hinzu:**

```gitignore
# Projekt-spezifisch (Ihre Ergänzungen)
# =====================================

# Training Outputs
/checkpoints/**/*
!/checkpoints/.gitkeep

# WandB Logs
/wandb/**/*
!/wandb/.gitkeep

# Cache
/.cache/
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment
/.venv/
/venv/
*.egg-info/

# IDE
/.vscode/
/.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporäre Docs (behalten Sie nur finalisierte Docs)
/docs/archive/legacy/PI0_*.md

# Persönliche Notizen
/notes/
/scratch/
TODO.md
```

### 5.2 Wichtige Dateien trotzdem tracken

**.gitkeep Dateien erstellen:**

```bash
cd ~/openpi
mkdir -p checkpoints wandb
touch checkpoints/.gitkeep wandb/.gitkeep
git add checkpoints/.gitkeep wandb/.gitkeep
git commit -m "chore: add .gitkeep for empty directories"
```

---

## Schritt 6: Repository-Struktur für Fork

### 6.1 README anpassen

Ihr Fork sollte ein eigenes README haben:

```bash
cd ~/openpi
cp README.md README_ORIGINAL.md
nano README.md
```

**Fügen Sie am Anfang hinzu:**

```markdown
# OpenPI - Sourteig Fork

> **Note:** Dies ist ein Fork von [TrossenRobotics/openpi](https://github.com/TrossenRobotics/openpi) mit Anpassungen für unser spezifisches Setup.

## Unterschiede zum Original

- ✨ Erweiterte Dokumentation (siehe [docs/](docs/))
- 🎯 Optimierte Training-Configs für RTX 6000 Pro
- 📊 Zusätzliche WandB Monitoring-Tools
- 🤖 Angepasst für Trossen AI Stationary Kit

## Original README

Siehe [README_ORIGINAL.md](README_ORIGINAL.md) für die originale TrossenRobotics Dokumentation.

---
```

### 6.2 CHANGELOG.md erstellen

```bash
cd ~/openpi
nano CHANGELOG.md
```

**Inhalt:**

```markdown
# Changelog - Sourteig Fork

Alle signifikanten Änderungen an diesem Fork werden hier dokumentiert.

## [Unreleased]

### Added
- Komplette Dokumentations-Restrukturierung (v2.0)
- RAG-optimierte Metadaten
- Git-Workflow Guide für Fork-Management

### Changed
- Dokumentation in thematische Ordner aufgeteilt
- Redundanzen entfernt

## [1.0.0] - 2025-01-07

### Added
- Initiale Pi0 Training Dokumentation (17 Dateien)
- Camera Config für Trossen AI Kit
- WandB Optimierungsguide

### Based On
- TrossenRobotics/openpi @ commit 5f6f593a
```

---

## Best Practices

### DO ✅

- **Committen Sie oft** - Kleine, fokussierte Commits
- **Nutzen Sie Feature Branches** - Niemals direkt in main arbeiten
- **Schreiben Sie klare Commit Messages** - Conventional Commits nutzen
- **Pullen Sie vor dem Pushen** - `git pull` vor jedem `git push`
- **Testen Sie vor dem Committen** - Code sollte funktionieren
- **Dokumentieren Sie Änderungen** - CHANGELOG.md aktualisieren

### DON'T ❌

- **Keine Secrets committen** - API Keys, Passwörter, etc.
- **Keine binären Daten** - Große Checkpoints, Videos (nutzen Sie Git LFS)
- **Kein Force Push auf main** - Nur auf Feature Branches
- **Keine ungetesteten Changes** - Immer erst lokal testen
- **Kein Rebase von bereits gepushten Commits** - Außer auf Feature Branches

---

## Häufige Szenarien

### Szenario 1: Neue Dokumentation hinzufügen

```bash
cd ~/openpi
git checkout -b docs/add-inference-guide

# Datei erstellen
nano docs/inference/34_advanced_topics.md

# Committen
git add docs/inference/34_advanced_topics.md
git commit -m "docs: add advanced inference topics guide"

# Pushen
git push -u origin docs/add-inference-guide

# Auf GitHub: Pull Request erstellen
```

### Szenario 2: Training Config anpassen

```bash
cd ~/openpi
git checkout -b config/optimize-batch-size

# Config ändern
nano src/openpi/training/config.py

# Testen
uv run python -c "from openpi.training import config; config.get_config('pi0_lighter_cup_trossen')"

# Committen
git add src/openpi/training/config.py
git commit -m "config: increase batch size to 64 for RTX 6000 Pro"

git push -u origin config/optimize-batch-size
```

### Szenario 3: Upstream Updates holen

```bash
cd ~/openpi
git checkout main
git fetch upstream

# Check was neu ist
git log main..upstream/main --oneline

# Mergen
git merge upstream/main

# Konflikte lösen falls nötig
# ...

# Pushen
git push origin main
```

### Szenario 4: Fehler rückgängig machen

**Letzter Commit rückgängig (lokal):**
```bash
git reset --soft HEAD~1  # Commit entfernen, Änderungen behalten
# oder
git reset --hard HEAD~1  # Commit UND Änderungen entfernen
```

**Bereits gepushter Commit rückgängig:**
```bash
git revert <commit-hash>
git push origin main
```

### Szenario 5: Checkpoint speichern

```bash
cd ~/openpi

# Stash für temporäres Speichern
git stash save "WIP: training config experiments"

# Andere Arbeit...
git checkout main
# ...

# Zurück zum Experiment
git checkout experiment/config
git stash pop
```

---

## Git LFS für große Dateien

### Wann Git LFS nutzen?

**Nutzen Sie Git LFS für:**
- Checkpoints (`.ckpt`, `.pth`)
- Datasets (`.zarr`, `.hdf5`)
- Videos (`.mp4`, `.avi`)
- Große Bilder

**Setup:**

```bash
cd ~/openpi

# Git LFS installieren (falls noch nicht)
sudo apt install git-lfs
git lfs install

# Track große Dateitypen
git lfs track "*.ckpt"
git lfs track "*.pth"
git lfs track "checkpoints/**/*.npz"

# .gitattributes wird automatisch erstellt
git add .gitattributes
git commit -m "chore: configure git lfs for checkpoints"
```

---

## Nächste Schritte

1. **Fork erstellen** - Auf GitHub fork button klicken
2. **Remote umkonfigurieren** - `git remote set-url origin ...`
3. **Änderungen pushen** - `git push -u origin main`
4. **.gitignore anpassen** - Persönliche Dateien ignorieren
5. **README anpassen** - Fork-spezifische Info hinzufügen

**Dann:**
- [01_installation.md](01_installation.md) - Environment Setup
- [../training/21_configuration.md](../training/21_configuration.md) - Training Config

---

## Siehe auch

- [GitHub Docs: About Forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git LFS Tutorial](https://git-lfs.github.com/)
- [../reference/41_troubleshooting.md](../reference/41_troubleshooting.md#git-issues) - Git Probleme lösen

---

## Changelog

- **2025-01-08:** Initial Version mit Fork-Setup
- **2025-01-08:** Git LFS Sektion hinzugefügt
- **2025-01-08:** Best Practices und Szenarien erweitert
