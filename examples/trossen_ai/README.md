# Collect episodes using LeRobot

[Recorded Dataset](https://huggingface.co/datasets/TrossenRoboticsCommunity/bimanual-widowxai-handover-cube)

[Visualize using this](https://huggingface.co/spaces/lerobot/visualize_dataset)


# Training

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero --exp-name=my_experiment --overwrite
```

To add custom trian config edit `openpi/src/training/config.py`


```python
TrainConfig(
        name="pi0_trossen_transfer_block",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repo_id="TrossenRoboticsCommunity/bimanual-widowxai-handover-cube",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="grab and handover the red cube",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.top",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        batch_size=2,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
```

We trained on a RTX5090 and fine tuned using LoRA

# Checkpoints
The checkpoints are stored in the checkpoints folder in the root of the directory.

Uncompress this file in your checkpoint directory if you want to use this. The policy was trained for ALoha Layout 14 input actions
[Hugging Face Model Open Pi Fine Tuned](https://huggingface.co/shantanu-tr/open_pi_finetune_checkpoint)


# Start a policy server

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_trossen_transfer_block --policy.dir=checkpoints/pi0_trossen_transfer_block/test_pi0_finetuning/19999
```

# Start the Client

```bash

 uv run examples/trossen_ai/main.py --mode autonomous  --task_prompt "grab red cube"
```


# Troubleshooting

In case you get dependency issue while running uv remove all numpy>2.0 bindings, also remove rlds dependency.

