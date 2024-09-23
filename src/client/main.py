import argparse
import asyncio
import logging
import multiprocessing as mp

from src.client.api_client import run_api_client
from src.client.camera import run_camera
from src.common.utils import get_path, setup_logging
from src.train.model import get_model_from_path, load_model_from_path

setup_logging("logs/client.log")
_log = logging.getLogger(__name__)


def get_model_and_run_camera(model_path: str, cam_idx: int | str = 0) -> None:
    """Load the model and run the camera."""
    _log.info("Camera PID: %d", mp.current_process().pid)
    try:
        cam_idx = int(cam_idx)
    except ValueError:
        cam_idx = get_path(cam_idx)
    model = get_model_from_path(model_path)
    load_model_from_path(model_path, model)
    model.share_memory()
    run_camera(model, cam_idx)


def run_client():
    _log.info("API client PID: %d", mp.current_process().pid)
    asyncio.run(run_api_client())


def main() -> None:
    """Train the model or classify images and output scores to a CSV file."""
    model_path = get_path(args.model_path)
    # Run camera in seperate thread/process, aggregate results and sent to API
    camera_proc = mp.Process(target=get_model_and_run_camera, args=(model_path, args.cam))
    api_client_proc = mp.Process(target=run_client)
    # Start the processes
    camera_proc.start()
    api_client_proc.start()
    # Wait for the processes to finish
    api_client_proc.join()
    camera_proc.join()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the expression recognition client.")
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Specify path to save or load the trained model. Include the model name in the filename.",
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        type=str,
        required=False,
        default=0,
        help="Specify the camera index or path to the video file. Default is 0 for the primary camera.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    setup_logging("logs/client.log", log_level=logging.DEBUG if args.debug else logging.INFO)
    main()
