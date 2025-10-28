import time

from dataclasses import dataclass
from piper_sdk import C_PiperInterface_V2


@dataclass
class PIPERMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]


class PIPERMotorsBus():
    def __init__(
        self,
        config: PIPERMotorsBusConfig
    ):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.factor = 57295.7795  # 1000*180/3.1415926
        self._is_enable = False

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.piper.get_connect_status()

    def connect(self):
        while (not self.piper.EnablePiper()):
            time.sleep(0.01)
        self._is_enable = True

    def disconnect(self):
        self.piper.DisconnectPort()

    def read(self):
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        end_pose_msg = self.piper.GetArmEndPoseMsgs()
        end_pose = end_pose_msg.end_pose

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        return {
            "joint_1": joint_state.joint_1,
            "joint_2": joint_state.joint_2,
            "joint_3": joint_state.joint_3,
            "joint_4": joint_state.joint_4,
            "joint_5": joint_state.joint_5,
            "joint_6": joint_state.joint_6,
            "X_axis": end_pose.X_axis,
            "Y_axis": end_pose.Y_axis,
            "Z_axis": end_pose.Z_axis,
            "RX_axis": end_pose.RX_axis,
            "RY_axis": end_pose.RY_axis,
            "RZ_axis": end_pose.RZ_axis,
            "gripper": gripper_state.grippers_angle
        }

    def write_joint(self, target_state:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 -92000 ~ 92000 / 57324.840764
                joint_2 (float): 关节2角度 -2400 ~ 120000 / 57324.840764
                joint_3 (float): 关节3角度 3000 ~ -110000 / 57324.840764
                joint_4 (float): 关节4角度 -90000 ~ 90000 / 57324.840764
                joint_5 (float): 关节5角度 80000 ~ -80000 / 57324.840764
                joint_6 (float): 关节6角度 -90000 ~ 90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """
        # joint_0 = round(target_state[0]*self.factor)
        # joint_1 = round(target_state[1]*self.factor)
        # joint_2 = round(target_state[2]*self.factor)
        # joint_3 = round(target_state[3]*self.factor)
        # joint_4 = round(target_state[4]*self.factor)
        # joint_5 = round(target_state[5]*self.factor)
        # gripper_range = round(target_state[-1]*1000*1000)

        joint_0 = round(target_state[0])
        joint_1 = round(target_state[1])
        joint_2 = round(target_state[2])
        joint_3 = round(target_state[3])
        joint_4 = round(target_state[4])
        joint_5 = round(target_state[5])
        gripper_range = round(target_state[-1])

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0)  # 单位 0.001°

    def write_endpose(self, target_state:list):
        """
            End pose control
            - target end pose: in mm and radians
                X_axis (float): X轴位置 (0.001mm)
                Y_axis (float): Y轴位置 (0.001mm)
                Z_axis (float): Z轴位置 (0.001mm)
                RX_axis (float): 绕X轴旋转 (0.01 degrees)
                RY_axis (float): 绕Y轴旋转 (0.01 degrees)
                RZ_axis (float): 绕Z轴旋转 (0.01 degrees)
                gripper_range: 夹爪角度 0~0.08
        """
        # X_axis = round(target_state[6]*1000*1000)
        # Y_axis = round(target_state[7]*1000*1000)
        # Z_axis = round(target_state[8]*1000*1000)
        # RX_axis = round(target_state[9]*self.factor)
        # RY_axis = round(target_state[10]*self.factor)
        # RZ_axis = round(target_state[11]*self.factor)
        # gripper_range = round(target_state[-1]*1000*1000)

        X_axis = round(target_state[6])
        Y_axis = round(target_state[7])
        Z_axis = round(target_state[8])
        RX_axis = round(target_state[9])
        RY_axis = round(target_state[10])
        RZ_axis = round(target_state[11])
        gripper_range = round(target_state[-1])

        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(X_axis, Y_axis, Z_axis, RX_axis, RY_axis, RZ_axis)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0)  # 单位 0.001°
