import random
import argparse
from typing import Any, Dict

import torch
import zmq
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from deploy.web_utils import TorchSerializer


class BaseInferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for the server.",
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the server.",
        default=5555
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on.",
        default="cuda"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the model.",
        default=7
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset.",
        default="/home/zhiheng/cache/huggingface/lerobot/test/piper-pickcube-jointctrl1"
    )

    # client mode
    args = parser.parse_args()

    device = args.device
    np.random.seed(args.seed)
    policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)

    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.image": [-0.1, 0.0],
        "observation.images.wristimage": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(args.dataset_path, delta_timestamps=delta_timestamps)
    item_idx = random.randint(0, len(dataset) - 1)
    data = dataset[item_idx]

    state = data["observation.state"][1]       # tensor, /255
    image = data['observation.images.image'][1] 
    wrist_image = data['observation.images.wristimage'][1] 
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)
    wrist_image = wrist_image.to(device, non_blocking=True)
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)
    wrist_image = wrist_image.unsqueeze(0)

    element = {
            "observation.images.image": image.detach().cpu().numpy(),
            "observation.images.wristimage": wrist_image.detach().cpu().numpy(),
            "observation.state": state.detach().cpu().numpy(),
    }
    element = {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in element.items()}
    action_chunk = policy_client.get_action(element)
    # plot_prediction_vs_groundtruth(data['action.actions'], action_chunk, save_path="/home/zhiheng/project/Isaac-GR00T/compare.png")
    print(action_chunk)
