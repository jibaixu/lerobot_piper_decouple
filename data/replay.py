import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

from robot.robots.piper.config_piper_follower import PIPERFollowerConfig
from robot.robots.piper.piper_follower import PIPERFollower


REPO_ID = "test2/piper_test81"
EPISODE_IDX = 0

# Create the robot and teleoperator configurations
robot_config = PIPERFollowerConfig()

robot = PIPERFollower(robot_config)

dataset = LeRobotDataset(REPO_ID, episodes=[EPISODE_IDX])
actions = dataset.hf_dataset.select_columns("action")

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()
