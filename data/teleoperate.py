import time

from lerobot.utils.utils import move_cursor_up
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from robot.robots.piper.config_piper_follower import PIPERFollowerConfig
from robot.robots.piper.piper_follower import PIPERFollower
from robot.teleoperators.piper.config_piper_leader import PIPERLeaderConfig
from robot.teleoperators.piper.piper_leader import PIPERLeader


FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ["image", "wrist_image"]
CAMERA_NAME_TO_SERIAL = {
    "image": "317422074519",
    "wrist_image": "317422075321",
}

camera_configs = []
for camera_name in CAMERA_NAMES:
    camera_configs.append(
        RealSenseCameraConfig(
            serial_number_or_name=CAMERA_NAME_TO_SERIAL[camera_name],
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=FPS,
        )
    )

# Create the robot and teleoperator configurations
robot_config = PIPERFollowerConfig(
    cameras={
        CAMERA_NAMES[0]: camera_configs[0],
        CAMERA_NAMES[1]: camera_configs[1],
    }
)
teleop_config = PIPERLeaderConfig()

robot = PIPERFollower(robot_config)
teleop = PIPERLeader(teleop_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
teleop.connect()

_init_rerun(session_name="piper_teleop_session")

if not robot.is_connected or not teleop.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

display_len = max(len(key) for key in robot.action_features)
start = time.perf_counter()
while True:
    loop_start = time.perf_counter()
    action = teleop.get_action()

    observation = robot.get_observation()
    log_rerun_data(observation, action)

    robot.send_action(action)
    dt_s = time.perf_counter() - loop_start
    busy_wait(1 / FPS - dt_s)

    loop_s = time.perf_counter() - loop_start

    print("\n" + "-" * (display_len + 10))
    print(f"{'NAME':<{display_len}} | {'NORM':>7}")
    for motor, value in action.items():
        print(f"{motor:<{display_len}} | {value:>7.2f}")
    print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

    move_cursor_up(len(action) + 5)
