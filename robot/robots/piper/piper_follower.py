#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots.robot import Robot

from robot.motors.piper.piper_motor import (
    PIPERMotorsBusConfig,
    PIPERMotorsBus,
)

from .config_piper_follower import PIPERFollowerConfig

logger = logging.getLogger(__name__)


class PIPERFollower(Robot):
    """
    PIPER using piper sdk
    """

    config_class = PIPERFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PIPERFollowerConfig):
        super().__init__(config)
        self.config = config
        bus_config = PIPERMotorsBusConfig(
            can_name="can_follower",
            motors={
                "joint_1": (1, "agilex_piper"),
                "joint_2": (2, "agilex_piper"),
                "joint_3": (3, "agilex_piper"),
                "joint_4": (4, "agilex_piper"),
                "joint_5": (5, "agilex_piper"),
                "joint_6": (6, "agilex_piper"),
                "X_axis": (7, "agilex_piper"),
                "Y_axis": (8, "agilex_piper"),
                "Z_axis": (9, "agilex_piper"),
                "RX_axis": (10, "agilex_piper"),
                "RY_axis": (11, "agilex_piper"),
                "RZ_axis": (12, "agilex_piper"),
                "gripper": (13, "agilex_piper"),
            }
        )
        self.bus = PIPERMotorsBus(config=bus_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        # if self.is_connected:
        #     raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def setup_motors(self) -> None:
        return

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        # Read arm position
        obs_dict = self.bus.read()
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any], move_mode: int = 0x01) -> dict[str, Any]:
        """
        - move_mode (int): MOVE mode.
            0x00: MOVE P (Position/EndPose)
            0x01: MOVE J (Joint) default
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        target_state = [action[f"{motor}.pos"] for motor in self.bus.motors]

        if move_mode == 0x00:
            self.bus.write_endpose(target_state)
        elif move_mode == 0x01:
            self.bus.write_joint(target_state)
        else:
            raise ValueError(f"Unsupported move_mode: {move_mode}")
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
