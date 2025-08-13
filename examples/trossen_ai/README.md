
# OpenPi – Training & Evaluating a Policy with LeRobot

This guide walks you through collecting episodes, training with OpenPi, fine-tuning using LoRA, evaluating, and running inference.

## Collect episodes using LeRobot


We collect episodes using ``Interbotix/lerobot`` for more information on installation and recording episodes check the following:
1. [Installation](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot/setup.html)
2. [Recording Episode](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot/record_episode.html)

Here is a recorded dataset using the above instructions:

[Recorded Dataset](https://huggingface.co/datasets/TrossenRoboticsCommunity/bimanual-widowxai-handover-cube)


You can also visualize the dataset using the following link, just paste the dataset name here:

[Visualize using this](https://huggingface.co/spaces/lerobot/visualize_dataset)


## Install UV

Follow the [UV installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up.

## OpenPi Setup

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## Training

Once you have recorded your dataset, you can begin training using the command below. We provide a custom training configuration for the Trossen AI dataset. Since the Aloha Legacy and Trossen AI Stationary share the same joint layout, this configuration is compatible. Explicit support for Trossen AI will be added in the future.


```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_trossen_transfer_block --exp-name=my_experiment --overwrite
```
## Custom Training Configuration

To add a custom training configuration, edit the `openpi/src/training/config.py` file. You can define your own `TrainConfig` with specific model parameters, dataset sources, prompts, and training options. After updating the configuration, reference your new config name in the training command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <your_custom_config_name> --exp-name=my_experiment --overwrite
```

This allows you to tailor the training process to your dataset and requirements.


Here is an example configuration for training on the Trossen AI dataset:

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

## Checkpoints

Checkpoints are stored in the `checkpoints` folder at the root of your project directory.

To use a pretrained policy, download and extract the following checkpoint into your `checkpoints` directory. This policy was trained for the Trossen AI Stationary Layout with 14 input actions:

- [OpenPi Fine-Tuned Checkpoint on Hugging Face](https://huggingface.co/shantanu-tr/open_pi_finetune_checkpoint)

After extraction, you can reference this checkpoint when starting the policy server.


## Running Inference with Your Trained Policy

Once training is complete and your checkpoint is ready, you can start the policy server and run the client to perform autonomous tasks.

### Start the Policy Server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/test_pi0_finetuning/19999
```

This command serves the trained policy, making it available for inference.

Launch the policy server using your trained checkpoint and configuration:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/test_pi0_finetuning/19999
```

This command serves the trained policy, making it available for inference.

### Start the Client

Run the client to interact with the policy server and execute tasks autonomously. Specify the desired task prompt:

```bash
uv run examples/trossen_ai/main.py --mode autonomous --task_prompt "grab red cube"
```

The client will connect to the policy server and perform the specified task using the trained model.


