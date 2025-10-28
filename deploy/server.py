import argparse

import zmq

from dataclasses import dataclass
from typing import Callable

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from deploy.web_utils import TorchSerializer

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555):
        super().__init__(host, port)
        self.register_endpoint("get_action", model.select_action)

    @staticmethod
    def start_server(policy , port: int):
        server = RobotInferenceServer(policy, port=port)
        server.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the server.",
        default=5555
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for the server.",
        default="localhost"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="/home/zhiheng/data/dp_output/jointctrl1"
    )
    # server mode
    args = parser.parse_args()

    # Create a policy
    # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
    # the model path, transform name, embodiment tag, and denoising steps for the robot
    # inference system. This policy object is then utilized in the server mode to start
    # the Robot Inference Server for making predictions based on the specified model and
    # configuration.

    # we will use an existing data config to create the modality config and transform
    # if a new data config is specified, this expect user to
    # construct your own modality config and transform
    # see gr00t/utils/data.py for more details

    policy = DiffusionPolicy.from_pretrained(args.model_path)
    # Start the server
    server = RobotInferenceServer(policy, port=args.port)
    server.run()
