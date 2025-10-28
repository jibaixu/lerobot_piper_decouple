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

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.teleoperators.teleoperator import Teleoperator

from robot.motors.piper.piper_motor import (
    PIPERMotorsBusConfig,
    PIPERMotorsBus,
)

from .config_piper_leader import PIPERLeaderConfig

logger = logging.getLogger(__name__)


class PIPERLeader(Teleoperator):
    """
    PIPER Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = PIPERLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PIPERLeaderConfig):
        super().__init__(config)
        self.config = config
        bus_config = PIPERMotorsBusConfig(
            can_name="can_leader",
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

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        # if self.is_connected:
        #     raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

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

    def get_action(self) -> dict[str, float]:
        action_raw = self.bus.read()  # 原始单位 0.001°
        # action 字段去掉比例缩放
        # joint_factor = 57295.7795  # 度转弧度比例因子（可调）
        # action = {
            # f"{motor}.pos": val / joint_factor if motor != "gripper" and motor[0] not in ['X', 'Y', 'Z'] else val / 1_000_000
            # for motor, val in action_raw.items()
        # }
        # action 字段保留原始比例缩放
        action = {
            f"{motor}.pos": val for motor, val in action_raw.items()
        }
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
