# OpenPI Dokumentation - Master Index

**Projekt:** Pi0 Training für Trossen AI Roboterarme  
**Version:** 2.0 (Restrukturiert: 08.01.2025)  
**Original Hersteller:** [TrossenRobotics/openpi](https://github.com/TrossenRobotics/openpi)  
**Ihr Fork:** [Sourteig/openpi](https://github.com/Sourteig/openpi)

---

## 🎯 Schnellstart

- **Neu hier?** → [setup/01_installation.md](setup/01_installation.md)
- **Training starten?** → [training/22_training_execution.md](training/22_training_execution.md)
- **Probleme?** → [reference/41_troubleshooting.md](reference/41_troubleshooting.md)

---

## 📚 Dokumentationsstruktur

### 🔧 Setup & Installation
Installation, Hardware-Anforderungen und Repository-Setup

- [01_installation.md](setup/01_installation.md) - UV Environment, Dependencies, Grundsetup
- [02_hardware_requirements.md](setup/02_hardware_requirements.md) - GPU, RAM, Storage Anforderungen
- [03_git_workflow.md](setup/03_git_workflow.md) - Fork-Setup und Workflow mit Ihrem Repository

### 💡 Konzepte & Grundlagen
Verständnis der Pi0 Architektur und wichtiger Konzepte

- [10_pi0_architecture.md](concepts/10_pi0_architecture.md) - Was ist Pi0? Flow-Matching, VLA Models
- [11_uv_vs_conda.md](concepts/11_uv_vs_conda.md) - UV Package Manager vs. Conda
- [12_adapt_to_pi.md](concepts/12_adapt_to_pi.md) - adapt_to_pi Parameter erklärt

### 🎓 Training
Datenaufbereitung, Konfiguration und Training-Durchführung

- [20_data_preparation.md](training/20_data_preparation.md) - Dataset, Camera Mapping, Normalisierung
- [21_configuration.md](training/21_configuration.md) - Training Config erstellen und anpassen
- [22_training_execution.md](training/22_training_execution.md) - Training starten und überwachen
- [23_monitoring.md](training/23_monitoring.md) - WandB Monitoring und Optimierung

### 🚀 Inference & Deployment
Policy Server, Robot Client und Hardware-Integration

- [30_server_setup.md](inference/30_server_setup.md) - Policy Server auf Training-Maschine
- [31_client_setup.md](inference/31_client_setup.md) - Robot Client auf Inference-PC
- [32_network.md](inference/32_network.md) - Netzwerk-Konfiguration und Optimierung
- [33_hardware_integration.md](inference/33_hardware_integration.md) - Hardware-Tests und Deployment

### 📖 Referenz
Detaillierte Referenzen, API-Docs und Troubleshooting

- [40_camera_config.md](reference/40_camera_config.md) - Camera Mapping und Konfiguration
- [41_troubleshooting.md](reference/41_troubleshooting.md) - Fehlerbehebung und FAQ
- [42_api_reference.md](reference/42_api_reference.md) - API Dokumentation und Code-Referenzen

### 📦 Archiv
Historische und temporäre Dokumentation

- [archive/legacy/](archive/legacy/) - Alte Versionen und Chat-Kontexte

---

## 🗺️ Dokumentations-Roadmap

### Für Einsteiger (Erste Schritte)
1. [setup/01_installation.md](setup/01_installation.md)
2. [concepts/10_pi0_architecture.md](concepts/10_pi0_architecture.md)
3. [training/20_data_preparation.md](training/20_data_preparation.md)
4. [training/21_configuration.md](training/21_configuration.md)
5. [training/22_training_execution.md](training/22_training_execution.md)

### Für Training (Vollständiger Workflow)
1. [setup/01_installation.md](setup/01_installation.md) - Environment Setup
2. [training/20_data_preparation.md](training/20_data_preparation.md) - Daten vorbereiten
3. [training/21_configuration.md](training/21_configuration.md) - Config erstellen
4. [training/22_training_execution.md](training/22_training_execution.md) - Training starten
5. [training/23_monitoring.md](training/23_monitoring.md) - Überwachen und optimieren

### Für Inference (Nach Training)
1. [inference/30_server_setup.md](inference/30_server_setup.md) - Server Setup
2. [inference/31_client_setup.md](inference/31_client_setup.md) - Client Setup
3. [inference/32_network.md](inference/32_network.md) - Netzwerk konfigurieren
4. [inference/33_hardware_integration.md](inference/33_hardware_integration.md) - Hardware testen

### Für Troubleshooting
- Problem während Training? → [reference/41_troubleshooting.md](reference/41_troubleshooting.md#training-issues)
- Problem bei Inference? → [reference/41_troubleshooting.md](reference/41_troubleshooting.md#inference-issues)
- Config-Fehler? → [reference/41_troubleshooting.md](reference/41_troubleshooting.md#configuration-errors)

---

## 🏷️ Thematischer Index

### Nach Hardware
- **RTX 6000 Pro 96GB** - [setup/02_hardware_requirements.md](setup/02_hardware_requirements.md#rtx-6000-pro)
- **RTX 4080 16GB** - [setup/02_hardware_requirements.md](setup/02_hardware_requirements.md#rtx-4080)
- **Trossen AI Kit** - [reference/40_camera_config.md](reference/40_camera_config.md)

### Nach Thema
- **UV vs Conda** - [concepts/11_uv_vs_conda.md](concepts/11_uv_vs_conda.md)
- **Camera Mapping** - [reference/40_camera_config.md](reference/40_camera_config.md)
- **WandB Monitoring** - [training/23_monitoring.md](training/23_monitoring.md)
- **Remote Inference** - [inference/32_network.md](inference/32_network.md#remote-setup)
- **Normalization Stats** - [training/20_data_preparation.md](training/20_data_preparation.md#normalization)

### Nach Schwierigkeitsgrad
- **Beginner** - Setup, Installation, Grundkonzepte
- **Intermediate** - Training, Konfiguration
- **Advanced** - Inference, Netzwerk, Optimierung

---

## 📝 Dokumentations-Konventionen

### Dateinamen
- Format: `NN_beschreibung.md` (NN = Nummer für Sortierung)
- Kleinbuchstaben mit Unterstrichen
- Sprechende Namen

### Metadaten (YAML Frontmatter)
Jedes Dokument hat standardisierte Metadaten:
```yaml
---
title: "Dokument-Titel"
category: setup|concepts|training|inference|reference
tags: [tag1, tag2, tag3]
difficulty: beginner|intermediate|advanced
last_updated: YYYY-MM-DD
status: stable|draft|deprecated
related_docs:
  - ../path/to/related.md
---
```

### Struktur
Jedes Dokument folgt dieser Struktur:
1. **Zusammenfassung (TL;DR)** - 2-3 Sätze Überblick
2. **Voraussetzungen** - Was sollte bekannt sein
3. **Hauptinhalt** - Strukturiert mit Überschriften
4. **Nächste Schritte** - Was kommt als nächstes
5. **Siehe auch** - Verwandte Dokumentation

---

## 🔄 Dokumentations-Wartung

### Aktualisierung
- Jede Änderung aktualisiert `last_updated` im Frontmatter
- Breaking Changes werden im Dokument prominent markiert
- Alte Versionen werden nach `archive/legacy/` verschoben

### Versionierung
- Alle Änderungen werden via Git getrackt
- Wichtige Meilensteine werden mit Git-Tags markiert
- Changelog in jeder Datei (am Ende)

### Feedback
- Fehler oder Unklarheiten? → GitHub Issues
- Verbesserungsvorschläge? → Pull Requests
- Fragen? → Siehe [reference/41_troubleshooting.md](reference/41_troubleshooting.md)

---

## 🔗 Externe Ressourcen

### Offizielle Dokumentation
- [π₀ Paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi0)
- [TrossenRobotics OpenPI Tutorial](https://docs.trossenrobotics.com/trossen_arm/v1.9/tutorials/openpi.html)

### Community
- [OpenPI GitHub (Original)](https://github.com/Physical-Intelligence/openpi)
- [TrossenRobotics Fork](https://github.com/TrossenRobotics/openpi)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)

### Verwandte Projekte
- [ALOHA](https://github.com/Physical-Intelligence/aloha)
- [DROID Dataset](https://droid-dataset.github.io/)
- [LeRobot](https://github.com/huggingface/lerobot)

---

## 📊 Dokumentations-Statistik

- **Gesamt-Dokumente:** 16 (+ 1 Index)
- **Setup:** 3 Dokumente
- **Konzepte:** 3 Dokumente
- **Training:** 4 Dokumente
- **Inference:** 4 Dokumente
- **Referenz:** 3 Dokumente

**Geschätzte Lesezeit:**
- Schnellstart: 30 Minuten
- Vollständig: 4-6 Stunden
- Mit praktischer Umsetzung: 2-3 Tage

---

## ✨ Was ist neu in v2.0?

**Verbesserungen gegenüber v1.0 (fragmentierte Docs):**
- ✅ Klare hierarchische Struktur
- ✅ Keine Redundanzen mehr
- ✅ RAG-optimierte Metadaten
- ✅ Konsistente Formatierung
- ✅ Thematische Gruppierung
- ✅ Bessere Navigation
- ✅ Aktualisierte Git-Workflow für Ihren Fork

---

**Dokumentation Version:** 2.0  
**Letzte Aktualisierung:** 08.01.2025  
**Status:** ✅ Production Ready

---

**Nächster Schritt:** [setup/01_installation.md](setup/01_installation.md) - Beginnen Sie hier!
