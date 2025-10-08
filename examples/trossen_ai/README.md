
# OpenPi – Training & Evaluating a Policy with LeRobot

This guide walks you through collecting episodes, training with OpenPi, fine-tuning using LoRA, evaluating, and running inference.

> **Note:**
> - This example uses two different versions of LeRobot:
>   - **LeRobot V0.1.0** for training and dependency management.
>   - **LeRobot V0.3.2** for running the client and inference.
> - The custom LeRobot V0.3.2 (with BiWidowXAIFollower support) is available on GitHub:
>   [Interbotix/lerobot – `trossen_ai_open_pi` branch](https://github.com/Interbotix/lerobot/tree/trossen_ai_open_pi)
> - **LeRobot V0.1.0** is installed at `.venv/lib/python3.11/site-packages/lerobot`.
> - **LeRobot V0.3.2** is installed at `examples/trossen_ai/.venv/lib/python3.11/site-packages/lerobot`.
> - **Training commands** should be run from the project root to use LeRobot V0.1.0.
> - **Client commands** should be run from the `examples/trossen_ai` directory to use LeRobot V0.3.2.
> - This setup works because `uv` manages dependencies in isolated virtual environments for each project.

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
git clone --recurse-submodules git@github.com:TrossenRobotics/openpi.git

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

Run this command from the project root. This is for dependency management. We have to use LeRobot V0.1.0.

```bash
cd openpi
```


```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_trossen_organize_tools --exp-name=pi0_trossen_organize_tools_01 --overwrite
```
## Custom Training Configuration

To add a custom training configuration, edit the `openpi/src/training/config.py` file. You can define your own `TrainConfig` with specific model parameters, dataset sources, prompts, and training options. After updating the configuration, reference your new config name in the training command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <your_custom_config_name> --exp-name=my_experiment --overwrite
```

This allows you to tailor the training process to your dataset and requirements.


Here is an example configuration for training on the Trossen AI dataset:


The camera mapping are used to map the camera names in the dataset to the expected input names for the Pi-0 model.
In this example the dataset has 4 cameras: top, bottom, left and right. We map them to the expected input names of the model: cam_high, cam_low, cam_left_wrist and cam_right_wrist.

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
                                "cam_low": "observation.images.bottom",
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

We have successfully trained models on an RTX5090 and fine-tuned using LoRA.


## Checkpoints

Checkpoints are stored in the `checkpoints` folder at the root of your project directory.

To use a pretrained policy, download and extract the following checkpoint into your `checkpoints` directory. This policy was trained for the Trossen AI Stationary Layout with 14 input actions:

- [OpenPi Fine-Tuned Checkpoint on Hugging Face](https://huggingface.co/shantanu-tr/open_pi_finetune_checkpoint)

After extraction, you can reference this checkpoint when starting the policy server.


## Running Inference with Your Trained Policy

Once training is complete and your checkpoint is ready, you can start the policy server and run the client to perform autonomous tasks.

### Start the Policy Server


This command serves the trained policy, making it available for inference.

Launch the policy server using your trained checkpoint and configuration:

Make sure to run this from project root. This allows us to use LeRobot V0.1.0
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/test_pi0_finetuning/19999
```

This command serves the trained policy, making it available for inference.

### Start the Client


Before starting the client we need to built the `LeRobot V0.3.2` package it needs.

```bash
cd examples/trossen_ai
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The client script requires the latest version of `LeRobot V0.3.2`, while the OpenPi repository depends on an older version `LeRobot V0.1.0` for data loading. To prevent version conflicts, the ``trossen_ai`` package uses the ``Interbotix/lerobot`` repository as its dependency. When using ``uv`` for package management, this setup creates a **separate virtual environment** for ``trossen_ai``. If you need to modify any LeRobot packages, ensure you are editing them in the **correct environment**.

Run the client to interact with the policy server and execute tasks autonomously.
We will use a the `examples/trossen_ai` as root directory for running the client. This is required as the client uses a different version of `LeRobot V0.3.2` than the training environment.


```bash
cd examples/trossen_ai
uv run main.py --mode autonomous --task_prompt "grab red cube"
```

The client will connect to the policy server and perform the specified task using the trained model.


You can change the cameras and arm ip address in the script `examples/trossen_ai/main.py` by editing

```python
bi_widowx_ai_config = BiWidowXAIFollowerConfig(
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=4.0,
            id="bimanual_follower",
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name="218622270304",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_low": RealSenseCameraConfig(
                    serial_number_or_name="130322272628",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="128422271347",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="218622274938",
                    width=640, height=480, fps=30, use_depth=False
                ),
            }
        )
```

The client script provides parameters to control both the **rate of inference** and **temporal ensembling**.

The **rate of inference** determines how often the policy is queried for new actions. Since each query is computationally expensive, frequent queries reduce the control frequency to around **10 Hz**, which can lead to jerky motions. To avoid this, you should choose a rate that balances **smoothness** and **responsiveness**.

- According to the Pi-0 paper, the control loop runs at **50 Hz**, with inference every **0.5 s** (after 25 actions).
- In our case, the control loop runs at **30 Hz** to align with the camera frame rate.  

Practical trade-offs:

- **Rate = 50** → smoother motion, less responsive to environment changes.  
- **Rate = 25** → more responsive, but noticeably jerky motion.  

Depending on your setup, you may need to adjust this parameter for optimal performance.

```python
self.rate_of_inference = 50  # Number of control steps per policy inference
```


**Temporal ensembling** is a technique for smoothing the actions generated by the policy.  
It was originally introduced in the [ACT paper](https://arxiv.org/abs/2304.13705), and later mentioned in the Pi-0 paper.

While simple to implement, the **Pi-0 appendix notes that temporal ensembling can actually hurt performance**. Our own experiments confirmed this — we observed no benefit, so by default the temporal ensembling weight is set to ``None``. That said, we have included an implementation of temporal ensembling in the client script for users who wish to experiment with it.

```python
self.temporal_ensemble_coefficient = None  # Temporal ensembling weight (can be set to None for no ensembling)
```
The paper, however, suggests not to use temporal ensembling for the Pi-0 policy. So, by default this value will be None.

## Results

Here are some preliminary results from our experiments with the Pi-0 policy on the bimanual WidowX setup.
Note that the Pi-0 base checkpoint has no episodes collected using Trossen-AI arms, so fine tuning is absolutely necessary for optimal performance. We collected a small dataset of 50 episodes for this purpose (which is very small in comparison to other robot modalities) zero shot inference using this checkpoint might be difficult as any changes in the environment, color of the blocks, shape of the objects can affect the performance.
The dataset collected was in an extremely controlled environment with pick and placing the a red color block from same position and dropping it in the same position, this reduces the variability and helps us verify the training and evaluation pipeline. 

Check the results out here:
[Google Drive Folder](https://drive.google.com/drive/folders/1waFcKihP8uAHSsV8VM-S7eBLDdTW7jfw?usp=sharing)

1. ``openpi_trossen_ai_red_block[success]`` : The robot is able to pickup and transfer the red block successfully in the second try.
2. ``openpi_trossen_ai_blue_lego[fail]`` : The robot fails to pick up the blue Lego block, likely due to differences in block size and color affecting the model's performance.
3. ``openpi_trossen_ai_environment_disturbances[fail]`` : The robot struggles to complete the task when the environment is disturbed, indicating sensitivity to changes in the setup.
4. ``openpi_trossen_ai_wooden_block[fail]`` : The robot fails to pick up the wooden block, suggesting that the model may not generalize well to different object types without further training.

We run this exact same command for testing each of these scenarios. The command is:

```bash
uv run main.py --mode autonomous --task_prompt "grab red cube"
```

The task prompt remains the same for all tests, as we haven't collected any data for other object types or scenarios.


If you want to run the client in test mode (no movement, just logs the actions that would be taken), you can use the following command:

```bash
uv run main.py --mode test --task_prompt "grab red cube"
```