# Pi0 Camera Configuration - Dein Setup

**Dataset analysiert:** `lighter_cup_v2episodes`  
**Robot Type:** `bi_widowxai_follower`  
**FPS:** 30  
**Total Episodes:** 92

---

## 🎥 Deine Camera-Namen

### Aktuelles LeRobot Format:
```json
{
  "observation.images.right_wrist": (480, 640, 3),
  "observation.images.left_wrist": (480, 640, 3),
  "observation.images.cam_low": (480, 640, 3),
  "observation.images.cam_high": (480, 640, 3)
}
```

### Pi0 Format (Ziel):
```json
{
  "cam_right_wrist": (480, 640, 3),
  "cam_left_wrist": (480, 640, 3),
  "cam_low": (480, 640, 3),
  "cam_high": (480, 640, 3)
}
```

---

## 🔄 Camera Mapping für Pi0

### Mapping Configuration

```python
# In openpi Training Config:
repack_transforms=_transforms.Group(
    inputs=[
        _transforms.RepackTransform({
            "images": {
                # Dein Name → Pi0 Name
                "cam_high": "observation.images.cam_high",           # ✅ Perfekt, kein Mapping nötig
                "cam_low": "observation.images.cam_low",             # ✅ Perfekt, kein Mapping nötig
                "cam_left_wrist": "observation.images.left_wrist",   # 🔄 Mapping: left_wrist → cam_left_wrist
                "cam_right_wrist": "observation.images.right_wrist", # 🔄 Mapping: right_wrist → cam_right_wrist
            },
            "state": "observation.state",
            "actions": "action",
        })
    ]
),
```

---

## ✅ Kompatibilität Check

| Dein Name | Pi0 Name | Status | Anmerkung |
|-----------|----------|--------|-----------|
| `observation.images.cam_high` | `cam_high` | ✅ Kompatibel | Direktes Mapping |
| `observation.images.cam_low` | `cam_low` | ✅ Kompatibel | Direktes Mapping |
| `observation.images.left_wrist` | `cam_left_wrist` | ✅ Kompatibel | Prefix ergänzt |
| `observation.images.right_wrist` | `cam_right_wrist` | ✅ Kompatibel | Prefix ergänzt |

**Fazit:** ✅ Alle 4 Cameras sind kompatibel! Nur kleine Namens-Anpassungen im Mapping nötig.

---

## 📊 Camera Specifications

**Alle Cameras:**
- Resolution: 480 x 640 (Height x Width)
- Channels: 3 (RGB)
- FPS: 30
- Codec: avc1
- Format: yuv420p
- Audio: Nein

**Gesamt Video-Streams:** 4 Cameras = 4 Video-Streams pro Timestep

---

## 🎯 Camera Layout (Vermutung)

Basierend auf Standard Trossen AI Setup:

```
         [cam_high]
            (Top)
              |
    
    [left_wrist]  🤖  [right_wrist]
         (L)             (R)
              |
              
         [cam_low]
          (Bottom)
```

**Positionen:**
- `cam_high`: Übersicht von oben
- `cam_low`: Tischperspektive von unten/vorne
- `left_wrist`: Wrist camera am linken Arm
- `right_wrist`: Wrist camera am rechten Arm

---

## 💡 Wichtig für Pi0 Training

### State & Action Dimensions

**State:** 14 Dimensionen
```python
[
    # Left arm (7 dims)
    left_joint_0.pos,
    left_joint_1.pos,
    left_joint_2.pos,
    left_joint_3.pos,
    left_joint_4.pos,
    left_joint_5.pos,
    left_left_carriage_joint.pos,  # Gripper
    
    # Right arm (7 dims)
    right_joint_0.pos,
    right_joint_1.pos,
    right_joint_2.pos,
    right_joint_3.pos,
    right_joint_4.pos,
    right_joint_5.pos,
    right_left_carriage_joint.pos,  # Gripper
]
```

**Action:** 14 Dimensionen (identisch zu State)

### Pi0 Config Anpassungen

```python
# In TrainConfig:
max_state_dim=32,   # Pi0 default, deine 14 werden gepaddet
max_action_dim=32,  # Pi0 default, deine 14 werden gepaddet
```

**Padding:** Pi0 padded automatisch auf max_dim (14 → 32)

---

## 🔍 Validation

### Prüfe dein Setup:

```bash
# In deinem lerobot conda environment
conda activate lerobot
cd ~/lerobot

python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lighter_cup_v2episodes')
print('✅ Dataset geladen')
print(f'Episodes: {ds.num_episodes}')
print(f'Camera keys: {list(ds.camera_keys)}')
print(f'State shape: {ds.meta[\"observation.state\"][\"shape\"]}')
print(f'Action shape: {ds.meta[\"action\"][\"shape\"]}')
"
```

**Erwartete Output:**
```
✅ Dataset geladen
Episodes: 92
Camera keys: ['cam_high', 'cam_low', 'left_wrist', 'right_wrist']
State shape: [14]
Action shape: [14]
```

---

## 📝 Nächste Schritte

1. **✅ Camera Mapping dokumentiert**
2. **⏳ Training Config mit diesem Mapping erstellen**
3. **⏳ Normalization Stats berechnen**
4. **⏳ Test Training starten**

Siehe `docs/PI0_TRAINING_CONFIG_TEMPLATE.md` für die vollständige Config.
