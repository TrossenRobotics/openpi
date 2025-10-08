# Python Syntax Notes - Underscore in Zahlen

## ✅ `save_interval=5_000` ist korrekt!

### Python Numeric Literals (PEP 515)

Seit Python 3.6 sind Underscores in Zahlen-Literalen erlaubt:

```python
# Alle diese sind IDENTISCH:
save_interval = 5000
save_interval = 5_000
save_interval = 5_0_0_0  # Auch OK, aber unüblich

# Verifikation:
>>> 5_000 == 5000
True
```

### Warum Underscores nutzen?

**Bessere Lesbarkeit bei großen Zahlen:**
```python
# Schwer zu lesen:
num_train_steps = 20000
batch_size = 1000000

# Leichter zu lesen:
num_train_steps = 20_000   # 20 Tausend
batch_size = 1_000_000     # 1 Million
```

### In openpi Code

Aus der openpi Codebase:
```python
# Sie nutzen BEIDE Stile:
num_train_steps: int = 30_000     # Mit Underscore
save_interval: int = 1000         # Ohne Underscore
save_interval=5000                # Ohne Underscore

# Alle sind korrekt!
```

### Empfehlung für deine Config

**Konsistent bleiben:**
```python
# Style 1: Mit Underscores (empfohlen für Zahlen ≥ 10.000)
num_train_steps=20_000,
save_interval=5_000,
batch_size=32,         # Klein genug ohne Underscore

# Style 2: Ohne Underscores (auch OK)
num_train_steps=20000,
save_interval=5000,
batch_size=32,
```

**Mein Tipp:** Nutze Underscores für Zahlen ≥ 10.000 → bessere Lesbarkeit!

---

## Andere Parameter-Syntax

### Alle korrekt in deiner Config:

```python
TrainConfig(
    name="pi0_lighter_cup_trossen",       # ✅ String
    num_train_steps=20_000,               # ✅ Int mit Underscore
    batch_size=32,                        # ✅ Int ohne Underscore
    ema_decay=None,                       # ✅ None
    
    model=pi0.Pi0Config(                  # ✅ Nested Object
        paligemma_variant="gemma_2b_lora", # ✅ String
    ),
    
    data=LeRobotAlohaDataConfig(          # ✅ Nested Object
        use_delta_joint_actions=False,    # ✅ Boolean
        adapt_to_pi=False,                # ✅ Boolean
    ),
)
```

**Alles syntaktisch korrekt!** ✅
