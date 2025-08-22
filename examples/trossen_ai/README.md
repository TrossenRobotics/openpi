
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

As of now we use 2 different versions of ``lerobot`` for training and evalaution. So, we need to adjust the dependencies accordingly.
Run this command to use ``lerobot==0.1.0`` for training. To have a clean dependency environment, it's recommended to use a virtual environment.

```bash
python -m venv .venv_train
source .venv_train/bin/activate
```

```bash
uv pip install ".[train_trossen_ai]"
```

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_trossen_transfer_block --exp-name=my_experiment --overwrite
```
## Custom Training Configuration

To add a custom training configuration, edit the `openpi/src/training/config.py` file. You can define your own `TrainConfig` with specific model parameters, dataset sources, prompts, and training options. After updating the configuration, reference your new config name in the training command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py <your_custom_config_name> --exp-name=my_experiment --overwrite
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
As we use 2 different versions of ``lerobot`` for training and evaluation, we need to adjust the dependencies accordingly.
Run this command before starting the evaluation to use ``lerobot==0.3.2``:

We will use the venv_train virtual environment for running the policy server

```bash
source .venv_train/bin/activate
```

### Start the Policy Server

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/test_pi0_finetuning/19999
```

This command serves the trained policy, making it available for inference.

Launch the policy server using your trained checkpoint and configuration:

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/test_pi0_finetuning/19999
```
```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/block_transfer_training_100k/99999
```
This command serves the trained policy, making it available for inference.

### Start the Client

Run the client to interact with the policy server and execute tasks autonomously. Specify the desired task prompt:
We will use a new virtual environment for running the client. This is required as the client uses a different version of lerobot (0.3.2) than the training environment.

```bash
uv venv .venv_eval
source .venv_eval/bin/activate
uv pip install -e ".[eval_trossen_ai]"
```

```bash
python examples/trossen_ai/main.py --mode autonomous --task_prompt "grab red cube"
```

The client will connect to the policy server and perform the specified task using the trained model.


You can change the cameras and arm ip address in the script by editing

```python
bi_widowx_ai_config = BiWidowXAIFollowerConfig(
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=4.0,
            id="bimanual_follower",
            cameras={
                "top": RealSenseCameraConfig(
                    serial_number_or_name="218622270304",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "bottom": RealSenseCameraConfig(
                    serial_number_or_name="130322272628",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "right": RealSenseCameraConfig(
                    serial_number_or_name="128422271347",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "left": RealSenseCameraConfig(
                    serial_number_or_name="218622274938",
                    width=640, height=480, fps=30, use_depth=False
                ),
            }
        )
```


You can also change the inference rate by modifying the `rate_of_inference` attribute. The policy will be queried for new actions every `rate_of_inference` control steps. The value 50 is in accordance with the paper for Pi-0.

```python
self.rate_of_inference = 50  # Number of control steps per policy inference
```

We have also implemented logic for temporal ensembling, which can be controlled via the `m` attribute. Setting `m` to a value between 0 and 1 enables exponential moving average of the action predictions, potentially smoothing out the control commands.

```python
self.m = None  # Temporal ensembling weight (can be set to None for no ensembling)
```
The paper, however, suggests not to use temporal ensembling for the Pi-0 policy. So, by default this value will be None.

## Results

Here are some preliminary results from our experiments with the Pi-0 policy on the bimanual WidowX setup.
Note that the Pi-0 base checkpoint has no episodes collected using Trossen-AI arms, so fine tuning is absolutely necessary for optimal performance. We collected a small dataset of 50 episodes for this purpose (whihc is very small in comparison to other robot modalities) zero shot inference using this checkpoint might be difficult as any changes in the environmnart, color of the blocks, shape of the objects can affect the performance.
The dataset collected was in an extrememly controlled enviromnet with pick and plcaing the a red color block from same position and dropping it in the same position, this reduces the variablity and helps us verify the training and evalaution pipeline. 

Check the results out here:
[Google Drive Folder](https://drive.google.com/drive/folders/1waFcKihP8uAHSsV8VM-S7eBLDdTW7jfw?usp=sharing)

1. ``openpi_trossen_ai_red_block[success]`` : The robot is able to pickup and transfer the red block successfully in the second try.
2. ``openpi_trossen_ai_blue_lego[fail]`` : The robot fails to pick up the blue Lego block, likely due to differences in block size and color affecting the model's performance.
3. ``openpi_trossen_ai_environment_disturbances[fail]`` : The robot struggles to complete the task when the environment is disturbed, indicating sensitivity to changes in the setup.
4. ``openpi_trossen_ai_wooden_block[fail]`` : The robot fails to pick up the wooden block, suggesting that the model may not generalize well to different object types without further training.

We run this exact same command for testing each of these scenarios. The command is:

```bash
python examples/trossen_ai/main.py --mode autonomous --task_prompt "grab red cube"
```

The task prompt remains the same for all tests, as we haven't collected any data for other object types or scenarios.