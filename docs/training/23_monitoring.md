# OpenPI WandB Monitoring & Optimierung - Komplettguide

**Erstellt:** 07.01.2025  
**Projekt:** Pi0 Training für Trossen AI Roboterarme  
**Training Config:** `pi0_lighter_cup_trossen`

---

## 📊 Übersicht: Verfügbare Metriken

Im Gegensatz zum ACT-Training mit LeRobot ist **OpenPI deutlich minimalistischer**. Das Framework loggt standardmäßig nur **3 Kern-Metriken**:

### Standard-Metriken (aus `train.py`)

```python
info = {
    "loss": loss,                              # Diffusion MSE Loss
    "grad_norm": optax.global_norm(grads),     # L2-Norm aller Gradienten
    "param_norm": optax.global_norm(kernel_params),  # L2-Norm der Parameter
}
```

**Zusätzlich beim Training-Start:**
- `camera_views`: Concat aller 4 Kamerabilder (erste 5 Samples)

### Was die Metriken bedeuten

#### 1. **loss** - Diffusion Loss
- **Typ:** MSE zwischen vorhergesagtem Velocity Field (v_t) und Ground Truth (u_t)
- **Bereich:** Typisch 0.001 - 0.1 (je niedriger desto besser)
- **Interpretation:**
  - Sinkt kontinuierlich → Gutes Training
  - Plateau nach 10k steps → Normal
  - Steigt plötzlich → Instabilität!

#### 2. **grad_norm** - Gradient Norm
- **Typ:** L2-Norm über alle Gradienten
- **Bereich:** Typisch 0.1 - 10.0
- **Interpretation:**
  - < 0.01 → Vanishing Gradients (Problem!)
  - 0.1 - 5.0 → Gesund
  - > 10.0 → Exploding Gradients (Problem!)
  - > 100.0 → Training divergiert!

#### 3. **param_norm** - Parameter Norm
- **Typ:** L2-Norm über alle Kernel-Parameter (ohne Bias/Scale)
- **Bereich:** Wächst langsam während Training
- **Interpretation:**
  - Steigt kontinuierlich → Normal
  - Plötzlicher Sprung → Instabilität
  - Kombiniere mit grad_norm für Diagnose

---

## 🎨 WandB Dashboard Optimierung

### Dashboard-Struktur (5 Panels)

```
┌─────────────────────────────────────────────────────┐
│  Panel A: Training Progress Overview                │
│  ├─ loss (Line) | grad_norm (Line) | param_norm     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Panel B: System Performance                        │
│  ├─ GPU Usage | GPU Memory | GPU Temp               │
│  ├─ CPU Usage | RAM Usage  | Disk I/O               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Panel C: Training Efficiency                       │
│  ├─ Steps/Sec | Samples/Sec | Time per Step         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Panel D: Gradient Health Analysis                  │
│  ├─ grad_norm vs loss (Scatter)                     │
│  ├─ grad_norm histogram                             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Panel E: Visual Quality Control                    │
│  ├─ camera_views (Image Gallery)                    │
│  ├─ Checkpoint Markers (Vertical Lines)             │
└─────────────────────────────────────────────────────┘
```

### Panel A: Training Progress Overview

**Charts:**
1. **Loss Over Time**
   - Type: Line Chart
   - X-Axis: Step
   - Y-Axis: Loss (log scale)
   - Smoothing: 0.6
   - Zweck: Hauptmetrik, zeigt Lernfortschritt

2. **Gradient Norm Over Time**
   - Type: Line Chart
   - X-Axis: Step
   - Y-Axis: grad_norm (log scale)
   - Smoothing: 0.3
   - Alert: > 10.0 (rot markieren)
   - Zweck: Gradient Stability Monitoring

3. **Parameter Norm Over Time**
   - Type: Line Chart
   - X-Axis: Step
   - Y-Axis: param_norm (linear scale)
   - Smoothing: 0.8
   - Zweck: Parameter Growth Tracking

**Layout Config:**
```json
{
  "panel_type": "line",
  "title": "Training Progress Overview",
  "metrics": ["loss", "grad_norm", "param_norm"],
  "smoothing": 0.6,
  "x_axis": "step",
  "layout": "horizontal"
}
```

### Panel B: System Performance

**WandB System Metrics aktivieren:**

```python
# In WandB Settings oder beim Init:
wandb.init(
    project="openpi",
    config=config,
    settings=wandb.Settings(
        _disable_stats=False,  # System metrics aktivieren
        _stats_sample_rate_seconds=10,  # Alle 10 Sek samplen
    )
)
```

**Verfügbare System-Metriken:**
- `system.gpu.0.gpu` - GPU Auslastung (%)
- `system.gpu.0.memory` - GPU Memory Used (GB)
- `system.gpu.0.memoryAllocated` - GPU Memory Allocated (%)
- `system.gpu.0.temp` - GPU Temperatur (°C)
- `system.cpu` - CPU Auslastung (%)
- `system.memory` - RAM Nutzung (%)
- `system.disk.in` - Disk Read (MB/s)
- `system.disk.out` - Disk Write (MB/s)

**Empfohlene Charts:**
1. **GPU Health (2x2 Grid)**
   - GPU Usage + GPU Memory
   - GPU Temp + Memory Allocated

2. **CPU/RAM Health (2x1)**
   - CPU Usage + RAM Usage

**Alert-Rules:**
```yaml
- gpu.memory > 90GB → "GPU Memory kritisch!"
- gpu.temp > 85°C → "GPU überhitzt!"
- cpu > 90% → "CPU Bottleneck!"
```

### Panel C: Training Efficiency

**Berechnung der Efficiency-Metriken:**

Diese werden nicht automatisch geloggt, müssen aber aus Timestamps berechnet werden:

```python
# In train.py würde man hinzufügen:
import time

start_time = time.time()
last_step_time = start_time

for step in pbar:
    # ... training step ...
    
    if step % config.log_interval == 0:
        current_time = time.time()
        time_per_step = (current_time - last_step_time) / config.log_interval
        steps_per_sec = 1.0 / time_per_step
        samples_per_sec = config.batch_size * steps_per_sec
        
        efficiency_metrics = {
            "efficiency/steps_per_sec": steps_per_sec,
            "efficiency/samples_per_sec": samples_per_sec,
            "efficiency/time_per_step": time_per_step,
        }
        wandb.log(efficiency_metrics, step=step)
        last_step_time = current_time
```

**Charts:**
- Steps/Sec über Zeit
- Samples/Sec über Zeit
- Time/Step über Zeit

### Panel D: Gradient Health Analysis

**Custom Charts in WandB:**

1. **Grad Norm vs Loss Scatter**
   - X: loss
   - Y: grad_norm
   - Color: step (gradient)
   - Zweck: Korrelation erkennen

2. **Grad Norm Distribution**
   - Histogram über alle Steps
   - Bins: 50
   - Zweck: Typischen Bereich verstehen

### Panel E: Visual Quality Control

**Camera Views:**
- Bereits implementiert (erste 5 Samples bei Step 0)
- Empfehlung: Auch bei Steps 5000, 10000, 15000, 20000 loggen

```python
# In train.py erweitern:
if step % 5000 == 0 and step > 0:
    # Sample new batch and log images
    sample_batch = next(data_iter)
    images_to_log = [
        wandb.Image(
            np.concatenate([np.array(img[i]) for img in sample_batch[0].images.values()], axis=1)
        )
        for i in range(min(5, len(next(iter(sample_batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=step)
```

---

## 🚀 Erweiterte Metriken (Optional)

### Code-Erweiterung für train.py

Um mehr Insights zu bekommen, kann man `train_step()` erweitern:

```python
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), chunked_loss  # ← Return auch per-sample loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    
    # ===== ERWEITERTE METRIKEN =====
    info = {
        # Standard
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        
        # NEU: Loss Statistics
        "loss_std": jnp.std(chunked_loss),
        "loss_max": jnp.max(chunked_loss),
        "loss_min": jnp.min(chunked_loss),
        
        # NEU: Gradient Statistics
        "grad_max": jax.tree.reduce(jnp.maximum, jax.tree.map(lambda g: jnp.max(jnp.abs(g)), grads)),
        "grad_mean": jax.tree.reduce(jnp.add, jax.tree.map(lambda g: jnp.mean(jnp.abs(g)), grads)) / 
                     jax.tree.reduce(jnp.add, jax.tree.map(lambda g: 1, grads)),
        
        # NEU: Update Statistics
        "update_norm": optax.global_norm(updates),
        "update_ratio": optax.global_norm(updates) / (optax.global_norm(params) + 1e-8),
        
        # NEU: EMA Distance (falls aktiv)
        "ema_distance": optax.global_norm(
            jax.tree.map(lambda x, y: x.value - y.value, new_params, state.ema_params)
        ) if state.ema_decay is not None else 0.0,
    }
    
    return new_state, info
```

### Learning Rate Tracking

```python
# In der Hauptschleife:
for step in pbar:
    with sharding.set_mesh(mesh):
        train_state, info = ptrain_step(train_rng, train_state, batch)
    
    # Learning Rate hinzufügen
    if hasattr(config.lr_schedule, 'get_lr'):
        info['lr'] = config.lr_schedule.get_lr(step)
    
    infos.append(info)
    # ... rest bleibt gleich
```

---

## ⚙️ num_workers Optimierung

### Aktuelle Systemauslastung (aus `top`):

```
CPU: 8.5% Gesamt (sehr niedrig!)
RAM: 15GB / 128GB verwendet (12% - viel Headroom)
GPU: RTX 6000 Pro 96GB (hauptsächlich belegt)
DataLoader Worker: 2x bei 100% CPU ← BOTTLENECK!
```

### Problem-Diagnose

**Die 2 DataLoader-Worker sind der Engpass:**
- Worker 1: 100.7% CPU (voll ausgelastet)
- Worker 2: 100.3% CPU (voll ausgelastet)
- Restliches System: < 10% CPU

**Das bedeutet:**
→ GPU wartet auf Daten!
→ Training läuft langsamer als möglich
→ Mehr Worker = höherer Durchsatz

### Empfehlung: num_workers=4

**Begründung:**

1. **CPU Kapazität:**
   - Aktuell: 8.5% CPU-Last bei 2 Workern
   - Mit 4 Workern: ~17% CPU-Last (immer noch sehr niedrig)
   - Mit 6 Workern: ~25% CPU-Last (auch ok)
   - System hat 24+ Cores → viel Reserve!

2. **RAM Verfügbarkeit:**
   - Pro Worker: ~2-3 GB
   - 4 Worker: ~12 GB zusätzlich
   - Verfügbar: 113 GB
   - Kein Problem!

3. **Daten-Pipeline:**
   - 4 Kameras @ 30 FPS
   - Batch Size 32
   - 4 Worker können Pipeline füllen

4. **Empirische Werte:**
   - 2 Worker: Baseline
   - 4 Worker: ~1.8x Speedup erwartet
   - 6 Worker: ~2.2x Speedup erwartet
   - 8+ Worker: Diminishing returns

### Konkrete Konfiguration

```python
# In ../openpi/src/openpi/training/config.py:

TrainConfig(
    name="pi0_lighter_cup_trossen",
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),
    data=LeRobotAlohaDataConfig(
        repo_id="MaxFridge/lighter_cup_v2",
        # ... rest bleibt gleich
    ),
    num_workers=4,  # ← VON 2 AUF 4 ERHÖHT
    batch_size=32,
    num_train_steps=20_000,
    # ... rest bleibt gleich
)
```

### Testing-Strategie

**Phase 1: Baseline messen**
```bash
cd ~/openpi
# Mit num_workers=2 (aktuell)
uv run scripts/train.py pi0_lighter_cup_trossen --exp-name=baseline_2workers
# Notiere: Steps/Sec, GPU Utilization
```

**Phase 2: Mit 4 Workern testen**
```bash
# Config anpassen auf num_workers=4
uv run scripts/train.py pi0_lighter_cup_trossen --exp-name=test_4workers
# Vergleiche: Steps/Sec sollte ~1.8x sein
```

**Phase 3: Optional 6 Worker**
```bash
# Falls 4 Worker gut funktioniert, teste 6
# Config anpassen auf num_workers=6
uv run scripts/train.py pi0_lighter_cup_trossen --exp-name=test_6workers
# Wenn Speedup < 2.0x → bleibe bei 4
```

**Erfolgskriterien:**
- ✅ GPU Utilization > 95%
- ✅ CPU < 50%
- ✅ Steps/Sec erhöht sich
- ✅ Keine Out-of-Memory Errors

---

## 🎯 WandB Best Practices

### 1. Run Organization

```python
# Beim Training-Start:
wandb.init(
    project="openpi",
    name=f"{config.name}_{config.exp_name}",
    tags=[
        "pi0",
        "lora",
        "trossen",
        "lighter_cup",
        f"workers_{config.num_workers}",
        f"batch_{config.batch_size}",
    ],
    group="pi0_lighter_cup_experiments",
    job_type="finetune",
    config=dataclasses.asdict(config),
)
```

### 2. Alert Rules

**In WandB UI einrichten:**

```yaml
Alerts:
  - name: "Gradient Explosion"
    metric: grad_norm
    condition: > 10.0
    action: email + slack
    
  - name: "Loss NaN"
    metric: loss
    condition: isNaN
    action: email + slack + stop_run
    
  - name: "GPU Memory Critical"
    metric: system.gpu.0.memory
    condition: > 90
    action: email
    
  - name: "Training Stalled"
    metric: loss
    condition: no_change_for_1000_steps
    action: email
```

### 3. Run Comparison

**Wichtige Metriken für Vergleich:**
- Final Loss (bei Step 20k)
- Durchschnittliche grad_norm
- Training Time (Dauer)
- Steps/Sec (Effizienz)
- GPU Memory Peak

**In WandB:**
1. Selektiere mehrere Runs
2. → "Compare"
3. → Parallel Coordinates Plot
4. → Table View mit Custom Columns

### 4. Checkpoint Tracking

```python
# Bei jedem Checkpoint:
if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
    checkpoint_path = _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    
    # WandB Artifact erstellen
    artifact = wandb.Artifact(
        name=f"checkpoint_{config.name}_{config.exp_name}",
        type="model",
        description=f"Checkpoint at step {step}",
        metadata={
            "step": step,
            "loss": float(reduced_info["loss"]),
            "grad_norm": float(reduced_info["grad_norm"]),
        }
    )
    artifact.add_dir(str(checkpoint_path))
    wandb.log_artifact(artifact)
```

---

## 📝 Zusammenfassung & Quick Actions

### Was du JETZT tun kannst (keine Code-Änderung):

1. **WandB Dashboard erstellen:**
   - Login auf wandb.ai
   - Gehe zu deinem Projekt "openpi"
   - Erstelle 5 Panels wie oben beschrieben
   - System Metrics aktivieren

2. **num_workers erhöhen:**
   ```bash
   cd ~/openpi
   # Editiere src/openpi/training/config.py
   # Ändere num_workers=2 zu num_workers=4
   ```

3. **Alerts einrichten:**
   - WandB UI → Settings → Alerts
   - Füge Regeln für grad_norm und loss hinzu

4. **Tags & Organization:**
   - Checke ob dein aktueller Run richtig getaggt ist
   - Falls nicht: Runs umbenennen/neu taggen

### Was du SPÄTER tun kannst (mit Code-Änderung):

1. **Erweiterte Metriken:**
   - train.py anpassen (siehe Code oben)
   - Mehr Insights in Loss & Gradients

2. **Efficiency Tracking:**
   - Steps/Sec und Samples/Sec loggen
   - Training-Effizienz messen

3. **Visual Quality Control:**
   - Camera Views bei Steps 5k/10k/15k/20k
   - Action Trajectory Plots

4. **Checkpoint Artifacts:**
   - Automatisches Upload zu WandB
   - Versionierung & Vergleich

---

## 🔗 Nächste Schritte

1. **Stelle num_workers auf 4** → Teste 1000 Steps → Vergleiche Speedup
2. **Erstelle WandB Dashboard** → 5 Panels wie beschrieben
3. **Aktiviere System Metrics** → Monitoring von GPU/CPU/RAM
4. **Richte Alerts ein** → Frühwarnung bei Problemen

Bei Fragen oder wenn du Hilfe bei der Implementierung brauchst, sag Bescheid!

---

**Dokument-Version:** 1.0  
**Letzte Aktualisierung:** 07.01.2025  
**Status:** ✅ Production Ready
