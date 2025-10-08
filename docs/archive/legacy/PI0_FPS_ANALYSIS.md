# Pi0 FPS Analysis - 30 FPS vs. 50Hz Confusion

**Kritische Frage:** Dataset 30 FPS, aber 50Hz Control Loop erwähnt?  
**Erstellt:** 07.01.2025, 14:35 Uhr  
**Status:** 🔍 GENAUE ANALYSE

---

## 📊 FAKTEN

### 1. Dataset FPS (Verifiziert)

```json
// lighter_cup_v2episodes/meta/info.json
"fps": 30,
"video.fps": 30
```

**Bestätigt:** Dataset wurde mit **30 FPS** aufgenommen ✅

### 2. Pi0 Model Defaults

```python
// src/openpi/models/pi0.py
action_horizon: int = 50  # Default
```

### 3. Unsere Training Config

```python
// src/openpi/training/config.py
model=pi0.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora"
    # action_horizon NICHT gesetzt → nutzt Default = 50
),
```

---

## 🧩 BEGRIFFE KLÄREN

### Wichtige Unterscheidung!

**1. FPS (Frames Per Second) - Training Data**
- Wie oft wurden Daten GESAMMELT
- Dein Dataset: **30 FPS**
- Bedeutung: 1 Frame alle 33.3ms

**2. action_horizon - Model Architecture**
- Wie viele ACTION STEPS das Model voraussagt
- Pi0 Default: **50 steps**
- Bedeutung: Model sagt 50 zukünftige Actions voraus
- **NICHT FPS!** Nur Sequence Length!

**3. control_freq - Inference Execution**
- Wie oft werden Actions AUSGEFÜHRT bei Inference
- Empfohlen: Match mit Training FPS
- Für dich: **30 Hz** (nicht 50Hz!)

---

## ✅ IST UNSER TRAINING OK?

### Kurze Antwort: JA! ✅

**Warum:**

1. **action_horizon = 50 ist UNABHÄNGIG von FPS**
   - Es ist nur die Länge der Action-Sequenz
   - Bei 30 FPS Training: 50 steps = 50/30 = 1.67 Sekunden Vorhersage
   - Das ist NORMAL und korrekt

2. **Pi0 Model ist Frequenz-Agnostisch beim Training**
   - Lernt die Sequenz-Muster, nicht absolute Zeitskalen
   - 30 FPS Daten trainieren genauso wie 50 FPS Daten
   - action_horizon=50 funktioniert mit jedem FPS

3. **Deine Config nutzt Defaults korrekt**
   - `action_horizon=50` (Default) ✅
   - Dataset 30 FPS ✅
   - **KEIN Problem!**

### Training ist NICHT betroffen! ✅

---

## ⚠️ WAS ANZUPASSEN IST

### Nur Dokumentation & Inference!

**Problem:** Ich habe in Dokumentation "50Hz Control" erwähnt.

**Korrekt für dich:** 30Hz Control (match mit Training FPS)

**Wo zu korrigieren:**

1. **Dokumentation:**
   - Bandbreiten-Berechnungen (basierend auf 50Hz)
   - Inference Client Beispiele (control_freq=50)
   - → Sollten 30Hz nutzen!

2. **Inference Config** (später):
   - `control_freq=30` statt 50
   - Anpassung nur bei Inference, nicht Training!

---

## 📐 KORREKTE BERECHNUNGEN

### Bandbreite @ 30 FPS (nicht 50Hz)

**Remote Inference über 6 Mbit LTE:**

```
Pro Frame: 3.5 MB (4 Cameras, unkomprimiert)
Bei 30 FPS: 3.5 MB × 30 = 105 MB/s = 840 Mbit/s

Mit JPEG Compression (80% Reduktion):
105 MB/s × 0.2 = 21 MB/s = 168 Mbit/s

Verfügbar: 6 Mbit/s = 0.75 MB/s

→ Immer noch 28x zu langsam! ❌
```

**Fazit bleibt:** Remote Inference unmöglich, Local Inference empfohlen

### Latency Budget @ 30 FPS

```
Period: 1/30 = 33.3ms

Verfügbare Zeit pro Cycle:
- Observation Capture: 5ms
- Inference (local): 30-40ms
- Action Execution: 5ms
Total: 40-50ms

→ Passt in 33.3ms? NEIN, zu eng!

Empfehlung: 20Hz Control für Safety Margin
Period: 50ms
- Observation: 5ms
- Inference: 40ms
- Execution: 5ms
Total: 50ms → Passt! ✅
```

---

## 🎯 HANDLUNGSEMPFEHLUNG

### Für aktuelles Training: ✅ NICHTS TUN!

**Training ist KORREKT:**
- action_horizon=50 ist OK für 30 FPS Data
- Model lernt korrekt
- Keine Änderung nötig
- **Lass Training weiterlaufen!**

### Für Inference (später):

**1. Control Frequency:**
```python
# Nicht:
control_freq=50  # Zu schnell für Daten

# Sondern:
control_freq=30  # Match mit Training FPS
# Oder besser:
control_freq=20  # Safety Margin für Inference Latency
```

**2. Dokumentation Update:**
- Alle "50Hz" Erwähnungen → "30Hz" oder "20-30Hz"
- Bandbreiten-Berechnungen korrigieren
- Latency Budgets anpassen

---

## 📝 WO IN DOKUMENTATION ZU KORRIGIEREN

### Dateien mit "50Hz" Erwähnungen:

1. **PI0_INFERENCE_COMPLETE_GUIDE.md**
   - `control_freq=50` → `control_freq=30`
   - Latency Berechnungen

2. **PI0_NETWORK_ANALYSIS.md**
   - Bandbreiten-Berechnungen @ 50Hz → @ 30Hz

3. **PI0_NEW_CHAT_CONTEXT.md**
   - Control Loop Erwähnungen

4. **PI0_MIGRATION_PLAN.md**
   - Verschiedene 50Hz Referenzen

**ABER:** Dokumentation korrigieren NACH Training, nicht jetzt!

---

## ✅ ZUSAMMENFASSUNG

### Training:

**Status:** ✅ KORREKT, weiter laufen lassen!

**Warum OK:**
- action_horizon=50 ≠ 50 FPS
- action_horizon ist nur Sequence Length
- Pi0 ist Frequenz-agnostisch
- 30 FPS Daten funktionieren perfekt mit action_horizon=50

### Inference:

**Anpassung nötig:** control_freq=30 (oder 20 für Safety)

**Nicht:** control_freq=50 (zu schnell, passt nicht zu Daten)

### Dokumentation:

**Korrigieren:** Später, nach Training abgeschlossen

**Nicht kritisch:** Training ist nicht betroffen

---

## 🎓 WICHTIGE KONZEPTE

### action_horizon vs. FPS

**action_horizon=50:**
- Model sagt 50 zukünftige Actions voraus
- Bei 30 FPS: 50 steps = 1.67 Sekunden Zukunft
- Bei 50 FPS: 50 steps = 1.0 Sekunden Zukunft
- **Zeitdauer unterschiedlich, aber Sequence Length gleich**

**FPS=30:**
- Daten-Sampling Rate
- 1 Frame alle 33.3ms
- Bestimmt Zeitskala der Daten

**control_freq (Inference):**
- Wie oft führen wir Actions aus
- Sollte mit Training FPS matchen (30Hz)
- Kann niedriger sein für Latency (20Hz)
- Sollte NICHT höher sein (50Hz) - passt nicht zu Daten

---

## ✅ DEIN TRAINING IST SICHER!

**Keine Änderung nötig an:**
- ✅ Training Config
- ✅ Model Config
- ✅ Dataset
- ✅ Laufendem Training

**Später anpassen:**
- ⏳ Dokumentation (50Hz → 30Hz Referenzen)
- ⏳ Inference Client (control_freq=30)

**Lass Training weiterlaufen! 🚀**
