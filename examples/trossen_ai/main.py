#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version)

Bridge between a bimanual widowx and the OpenPI policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm

Usage:
    python main.py --mode autonomous --task_prompt "grab and handover red cube"
"""

import argparse
import logging
import time
import sys
import os
import numpy as np
from openpi_client import websocket_client_policy
import cv2
from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_widowxai_follower.config_bi_widowxai_follower import BiWidowXAIFollowerConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from collections import defaultdict
from scipy.interpolate import PchipInterpolator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lerobot'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrossenOpenPIBridge:
    """Bridge between a single Trossen arm and OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: bool = False,
    ):
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host,
            port=policy_server_port
        )

        bi_widowx_ai_config = BiWidowXAIFollowerConfig(
            left_arm_ip_address="192.168.1.121",
            right_arm_ip_address="192.168.1.111",
            min_time_to_move_multiplier=4.0,
            id="bimanual_follower",
            cameras={
                "top": RealSenseCameraConfig(
                    serial_number_or_name="218622271179",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "bottom": RealSenseCameraConfig(
                    serial_number_or_name="218622271198",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "right": RealSenseCameraConfig(
                    serial_number_or_name="218622278132",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "left": RealSenseCameraConfig(
                    serial_number_or_name="218622276227",
                    width=640, height=480, fps=30, use_depth=False
                ),
            }
        )
        self.robot = make_robot_from_config(bi_widowx_ai_config)
        self.robot.connect()

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 25  # Number of control steps per policy inference

        self.m = 0.01  # Temporal ensembling weight (can be set to None for no ensembling)

        # FIFO Buffer for actions
        self.buffer = defaultdict(list)
        self.action_buffer_size = 30 * 60  # Predefined buffer size (Depends on max episode length)

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm."""
        full_action = action.copy()
        joint_limits = np.array([
            [-np.pi, np.pi],           # left_joint_0.pos
            [-np.pi / 2, np.pi / 2],  # left_joint_1.pos
            [-np.pi / 2, np.pi / 2],  # left_joint_2.pos
            [-np.pi, np.pi],           # left_joint_3.pos
            [-np.pi / 2, np.pi / 2],  # left_joint_4.pos
            [-np.pi, np.pi],           # left_joint_5.pos
            [0.0, 0.04],               # left_left_carriage_joint.pos
            [-np.pi, np.pi],           # right_joint_0.pos
            [-np.pi / 2, np.pi / 2],  # right_joint_1.pos
            [-np.pi / 2, np.pi / 2],  # right_joint_2.pos
            [-np.pi, np.pi],           # right_joint_3.pos
            [-np.pi / 2, np.pi / 2],  # right_joint_4.pos
            [-np.pi, np.pi],           # right_joint_5.pos
            [0.0, 0.04],               # right_left_carriage_joint.pos
        ])
        for i in range(len(full_action)):
            full_action[i] = np.clip(full_action[i], joint_limits[i][0], joint_limits[i][1])

        if self.test_mode:
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return

        left_arm_action = full_action[:7]
        right_arm_action = full_action[7:]
        action_dict = {}
        for i in range(7):
            if i == 6:
                action_dict["left_left_carriage_joint.pos"] = left_arm_action[i]
                action_dict["right_left_carriage_joint.pos"] = right_arm_action[i]
            else:
                action_dict[f"left_joint_{i}.pos"] = left_arm_action[i]
                action_dict[f"right_joint_{i}.pos"] = right_arm_action[i]
        self.robot.send_action(action_dict)

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):

        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent 
        jumps and triggering safety stops (velocity limits)."""

        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith('.pos')]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        # stage_pose = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0])
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])  # Use the provided duration
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def run_episode(self, max_steps: int = 1000, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        # self.robot.configure()
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        while self.is_running and self.episode_step < max_steps:
            start_loop_time = time.perf_counter()
            observation_dict = self.robot.get_observation()

            # Extract joint positions from observation
            joint_pos_keys = [k for k in observation_dict.keys() if k.endswith('.pos')]
            joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])


            # Transform and resize images from all cameras
            cameras = ['top', 'bottom', 'right', 'left']
            for cam in cameras:
                image_hwc = observation_dict[cam]
                image_resized = cv2.resize(image_hwc, (224, 224))
                image_chw = np.transpose(image_resized, (2, 0, 1))
                observation_dict[cam] = image_chw

            # Create observation for policy to follow the ALOHA format
            observation = {
                "state": joint_positions,
                "images": {
                    "cam_high": observation_dict['top'],
                    "cam_low": observation_dict['bottom'],
                    "cam_right_wrist": observation_dict['right'],
                    "cam_left_wrist": observation_dict['left'],
                },
                "prompt": task_prompt
            }

            # Request new action chunk after consuming the previous one (Can implement temporal ensembling here)
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]

                for k in range(50):
                    future_t = self.episode_step + k
                    if future_t < 30*60:
                        self.buffer[future_t].append(self.current_action_chunk[k])

                self.action_chunk_idx = 0
                logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

            if self.m is not None:
                if len(self.buffer[self.episode_step]) == 0:
                    a_t = np.zeros(self.action_dim)
                else:
                    candidates = np.array(self.buffer[self.episode_step])  # shape: (N, 14)
                    weights = self._get_weights(len(candidates))  # shape: (N,)
                    a_t = np.average(candidates, axis=0, weights=weights)  # shape: (14,)
            else:
                a_t = self.current_action_chunk[self.action_chunk_idx]
            # Execute the current action
            if is_first_step:
                logger.info("Moving to start position to avoid large jumps...")
                self.move_to_start_position(a_t, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(a_t)

            self.action_chunk_idx += 1
            self.episode_step += 1


            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.m * np.arange(num_preds))
        return weights / weights.sum()


    def autonomous_mode(self, max_steps: int = 1000, task_prompt: str = "look down"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        if self.test_mode:
            logger.info("TEST MODE: Simulating autonomous mode without robot movement")
        else:
            logger.info("The arm will execute actions predicted by the Pi0 model.")
        self.run_episode(max_steps=max_steps, task_prompt=task_prompt)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen Single Arm <-> OpenPI Policy Server Bridge")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--mode", choices=["autonomous", "test_real_robot"], required=True,
                        help="Operation mode: autonomous (execute) or test_real_robot (use real robot data, no movement)")
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode == "test_real_robot",
    )

    if args.mode == "autonomous":
        bridge.autonomous_mode(max_steps=4000, task_prompt=args.task_prompt)
    else:
        logger.info("TEST MODE: Will connect to robot, move to home position, and read real data but NOT execute policy actions!")
        # If you want to implement test_real_robot_mode, add the method here.
        logger.info("test_real_robot_mode is not implemented.")

    bridge.cleanup()
