import time

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (Robot)
from lerobot.teleoperators import Teleoperator
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import (
    get_safe_torch_device,
    log_say,
    init_logging,
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

from robot.robots.piper.config_piper_follower import PIPERFollowerConfig
from robot.robots.piper.piper_follower import PIPERFollower
from robot.teleoperators.piper.config_piper_leader import PIPERLeaderConfig
from robot.teleoperators.piper.piper_leader import PIPERLeader



USE_TELEOPERATOR = False
# --------- Configuration for dataset ---------
REPO_ID = "test2/piper_test86"
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
CAMERA_NAMES = ["image", "wrist_image_left", "wrist_image_right"]
CAMERA_NAME_TO_SERIAL = {
    "image": "317422074519",
    "wrist_image_left": "317422074290",
    "wrist_image_right": "317422075321",
}

init_logging()
_init_rerun(session_name="piper_record_session")

camera_configs = {}
for camera_name in CAMERA_NAMES:
    camera_configs[camera_name] = RealSenseCameraConfig(
            serial_number_or_name=CAMERA_NAME_TO_SERIAL[camera_name],
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=FPS,
        )

# Create the robot and teleoperator configurations
robot_config = PIPERFollowerConfig(
    cameras=camera_configs
)
robot = PIPERFollower(robot_config)
robot.connect()

if USE_TELEOPERATOR:
    teleop_config = PIPERLeaderConfig()
    teleop = PIPERLeader(teleop_config)
    teleop.connect()

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

listener, events = init_keyboard_listener()

if not robot.is_connected or (USE_TELEOPERATOR and not teleop.is_connected):
    raise ValueError("Robot, leader arm of keyboard is not connected!")


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        if policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None and isinstance(teleop, Teleoperator):
            action = teleop.get_action()
        else:
            action = {k: v for k, v in observation.items() if k.endswith(".pos")}

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        if policy is None and teleop is None:
            sent_action = action
        else:
            sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)

        if display_data:
            log_rerun_data(observation, action)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Run the record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=teleop if USE_TELEOPERATOR else None,
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
            teleop=teleop if USE_TELEOPERATOR else None,
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
if USE_TELEOPERATOR:
    teleop.disconnect()
listener.stop()
