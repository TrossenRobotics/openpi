---
title: "adapt_to_pi Parameter erklärt"
category: concepts
tags: [adapt_to_pi, configuration, trossen, aloha]
difficulty: intermediate
last_updated: 2025-01-08
status: stable
related_docs:
  - ../training/21_configuration.md
  - 10_pi0_architecture.md
---

# `adapt_to_pi` Parameter - Detaillierte Erklärung

**Frage:** Warum ist `adapt_to_pi=False` in der empfohlenen Config?  
**Erstellt:** 07.01.2025

---

## 🔍 Was macht `adapt_to_pi`?

### Code-Analyse

Wenn `adapt_to_pi=True`, werden **zwei Transformationen** angewendet:

#### 1. Joint Flipping (Koordinatensystem-Konversion)

```python
def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
    
# Angewendet auf State und Actions:
state = _joint_flip_mask() * state
```

**Was passiert:**
- Joints 1, 2, 8, 9 werden invertiert (Vorzeichen flip)
- Andere Joints bleiben unverändert
- **Warum?** Pi's eigene Roboter haben ein anderes Koordinatensystem

#### 2. Gripper Space Transformation

```python
def _gripper_to_angular(value):
    # Konvertiert von Aloha "linear space" zu Pi "angular space"
    
    # 1. Denormalisierung (Aloha's linear gripper range)
    value = unnormalize(value, min=0.01844, max=0.05800)
    
    # 2. Linear → Angular Konversion
    # (Inverse der Interbotix Transformation)
    
    # 3. Ergebnis: Gripper in "echten" Radians
```

**Was passiert:**
- Aloha transformiert Gripper-Positionen in einen "linear space" (0.01844 - 0.05800)
- Pi0 Base Model wurde mit "angular space" (echte Radians) trainiert
- Diese Funktion konvertiert zurück zu Radians

---

## 🤔 Warum `False` für Trossen?

### Der Grund (laut Trossen Tutorial)

```python
adapt_to_pi=False,  # Trossen != Pi internal runtime
```

### Erklärung in 3 Szenarien

#### Szenario A: Pi's eigene Roboter (Original Training)

**Pi0 Base Model wurde trainiert auf:**
- Pi's eigene Roboter (nicht Aloha/Trossen!)
- Mit `adapt_to_pi=True` transformierten Daten
- Spezifisches Joint-Koordinatensystem
- Spezifische Gripper-Konventionen

#### Szenario B: Standard Aloha (Legacy)

**Standard Aloha Roboter:**
- Brauchen `adapt_to_pi=True` für Kompatibilität mit Pi0 Base Model
- Weil sie ähnlich zu Pi's original robots sind
- Joint flipping stimmt
- Gripper transformation stimmt

#### Szenario C: Trossen AI Stationary Kit (DU!)

**Trossen Hardware:**
- **Mechanisch ähnlich zu Aloha Legacy** (laut Trossen: "fully compatible")
- **ABER: Anderes Runtime/Controller System**
- **Gripper-Implementierung könnte anders sein**
- **Joint-Konventionen könnten leicht abweichen**

**Daher `adapt_to_pi=False`:**
- Trainiere im **nativen Trossen Format**
- Lasse das Model die Trossen-spezifischen Werte lernen
- **Konsistenz:** Training und Inference nutzen gleiches Format

---

## 📊 Vergleich: True vs. False

| Parameter | Joint Transform | Gripper Transform | Wann nutzen? |
|-----------|----------------|-------------------|--------------|
| `adapt_to_pi=True` | Flip Joints 1,2,8,9 | Linear → Angular | Pi's original robots, Standard Aloha Legacy |
| `adapt_to_pi=False` | Keine Transformation | Keine Transformation | **Trossen AI**, Custom Hardware |

---

## 💡 Warum False die richtige Wahl ist

### Grund 1: Trossen Runtime Unterschiede

Trossen AI nutzt **eigene Firmware/Controller:**
- Möglicherweise andere Gripper-Kalibrierung
- Möglicherweise andere Joint-Offsets
- **Besser:** Rohe Werte verwenden, Model lernt die Mappings

### Grund 2: Konsistenz Training ↔ Inference

```python
# Training
adapt_to_pi=False  # Lerne mit rohen Trossen Daten

# Inference (später)
adapt_to_pi=False  # Inference mit gleichen Konventionen

# ✅ Konsistent!
```

Falls du `adapt_to_pi=True` beim Training nutzt:
- Musst du auch `adapt_to_pi=True` bei Inference nutzen
- **Problem:** Transformationen könnten nicht exakt für Trossen passen
- **Risiko:** Fehlerhafte Actions auf echter Hardware

### Grund 3: Empirische Evidenz (Trossen)

Trossen hat in ihrem Tutorial **explizit** `adapt_to_pi=False` verwendet:
- Mit ihrem eigenen Hardware getest
- Funktioniert nachweislich
- **Folge ihrer Empfehlung!**

---

## 🧪 Was wäre mit `True`?

### Wenn du `adapt_to_pi=True` nutzen würdest:

**Vorteile:**
- Theoretisch näher am Pi0 Base Model Pre-training
- Könnte besseren Transfer Learning ermöglichen

**Nachteile:**
- ❌ Joint Flipping könnte falsch sein für Trossen
- ❌ Gripper Transformation könnte nicht exakt passen
- ❌ Inkonsistenz wenn Hardware anders kalibriert ist
- ❌ Schwer zu debuggen wenn Actions falsch sind

**Fazit:** Risiko > Nutzen für Trossen Hardware

---

## 🎯 Empfehlung

### Für dein Setup: `adapt_to_pi=False` ✅

**Warum:**
1. ✅ Trossen Tutorial empfiehlt es
2. ✅ Konsistenz mit deinen gesammelten Daten
3. ✅ Einfachere Fehlersuche
4. ✅ Model lernt direkt Trossen-Konventionen
5. ✅ Empirisch getestet von Trossen

### Nur nutze `True` wenn:
- Du hast **exakt** Standard Aloha Legacy Hardware
- Du weißt dass die Transformationen zu deiner Hardware passen
- Trossen empfiehlt es explizit für dein Setup

---

## 🔬 Technische Details

### Die Transformationen im Detail

#### Joint Flip Mask
```python
[1, -1, -1, 1, 1, 1, 1,     # Left arm joints
 1, -1, -1, 1, 1, 1, 1]     # Right arm joints

# Joints 1 & 2: Invertiert (shoulder, elbow?)
# Joints 0, 3-6: Unverändert
# Gespiegelt für rechten Arm
```

**Zweck:** Konvertiert zwischen verschiedenen Koordinatensystem-Konventionen

#### Gripper Transformation
```python
# Aloha Linear Range: [0.01844, 0.05800]
# Pi Angular Range: berechnet aus Hardware-Specs

# Konversion:
# 1. Denormalize from [0,1] to linear
# 2. Apply inverse Interbotix angular→linear transform
# 3. Result: True angular gripper position
```

**Zweck:** Aloha's Runtime konvertiert Angular → Linear für einfachere Control. Pi0 will echte Angular Werte.

---

## 🧭 Decision Tree

```
Hast du Pi's original Roboter?
├─ Ja → adapt_to_pi=True
└─ Nein
    ├─ Standard Aloha Legacy?
    │   └─ Ja → adapt_to_pi=True (vermutlich)
    └─ Trossen AI Stationary?
        └─ Ja → adapt_to_pi=False ✅ (DU BIST HIER!)
```

---

## 📝 In deiner Config

```python
data=LeRobotAlohaDataConfig(
    # ... andere params ...
    
    # EMPFOHLEN für Trossen AI:
    adapt_to_pi=False,
    
    # Erklärung:
    # - Deine gesammelten Daten sind im nativen Trossen Format
    # - Model lernt direkt mit diesen Werten
    # - Konsistenz zwischen Training und Inference
    # - Trossen hat dies empirisch getestet
),
```

---

## 🔄 Wenn du experimentieren möchtest

Du kannst zwei Configs erstellen und vergleichen:

```python
# Config 1: Trossen Empfehlung
TrainConfig(
    name="pi0_lighter_cup_no_adapt",
    data=LeRobotAlohaDataConfig(
        adapt_to_pi=False,  # Trossen Empfehlung
        # ...
    ),
)

# Config 2: Experiment
TrainConfig(
    name="pi0_lighter_cup_with_adapt",
    data=LeRobotAlohaDataConfig(
        adapt_to_pi=True,   # Experiment: Pi Format
        # ...
    ),
)
```

**Dann vergleiche:**
- Training Loss Kurven
- Validation Performance
- **Hardware Tests:** Actions korrekt?

**Erwartung:** `adapt_to_pi=False` sollte besser funktionieren für Trossen!

---

## ✅ Zusammenfassung

**`adapt_to_pi=False` bedeutet:**
- ❌ Keine Joint Flipping
- ❌ Keine Gripper Space Transformation
- ✅ Daten bleiben im nativen Trossen/Aloha Format
- ✅ Model lernt direkt mit deinen Hardware-Konventionen

**Warum False für dich:**
1. Trossen AI != Pi's original robots
2. Deine Daten sind bereits im Trossen Format
3. Trossen Tutorial empfiehlt es
4. Konsistenz Training ↔ Inference
5. Einfacher zu debuggen

**Nutze `adapt_to_pi=False` wie im Template! ✅**

---

## 🔗 Referenzen

**Code Location:**
- `~/openpi/src/openpi/policies/aloha_policy.py`
- Funktionen: `_decode_state()`, `_encode_actions()`, `_joint_flip_mask()`, `_gripper_to_angular()`

**Trossen Tutorial:**
- https://docs.trossenrobotics.com/trossen_arm/v1.9/tutorials/openpi.html
- Config Beispiel mit `adapt_to_pi=False`
