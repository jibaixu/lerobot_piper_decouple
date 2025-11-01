from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import (
    log_say,
    init_logging,
)
from lerobot.utils.visualization_utils import _init_rerun

from robot.robots.piper.config_piper_follower import PIPERFollowerConfig
from robot.robots.piper.piper_follower import PIPERFollower
from robot.teleoperators.piper.config_piper_leader import PIPERLeaderConfig
from robot.teleoperators.piper.piper_leader import PIPERLeader


# --------- Configuration for dataset ---------
REPO_ID = "test2/piper_test81"
NUM_EPISODES = 2
# Number of seconds for data recording for each episode.
EPISODE_TIME_SEC = 3600
# Number of seconds for resetting the environment after each episode.
RESET_TIME_SEC = 3600
TASK_DESCRIPTION = "My task description"

# --------- Configuration for camera ---------
FPS = 30
NUM_IMAGE_WRITER_PROCESSES = 0
NUM_IMAGE_WRITER_THREADS_PER_CAMERA = 4

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ["image", "wrist_image"]
CAMERA_NAME_TO_SERIAL = {
    "image": "317422074519",
    "wrist_image": "317422075321",
}

init_logging()
_init_rerun(session_name="piper_record_session")

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
    cameras=CAMERA_NAME_TO_SERIAL
)
teleop_config = PIPERLeaderConfig()

robot = PIPERFollower(robot_config)
teleop = PIPERLeader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_processes=NUM_IMAGE_WRITER_PROCESSES,
    image_writer_threads=NUM_IMAGE_WRITER_THREADS_PER_CAMERA * len(CAMERA_NAMES),
)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
teleop.connect()

listener, events = init_keyboard_listener()

if not robot.is_connected or not teleop.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Run the record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=teleop,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Logic for reset env
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
# dataset.push_to_hub()

robot.disconnect()
teleop.disconnect()
listener.stop()
