import logging
from collections import deque
from datetime import UTC, datetime

import cv2
import torch.nn as nn

from src.client.buffer import CAN_PRODUCE_EVENT, PREDICTIONS_QUEUE, STOP_EVENT
from src.client.capture import CameraError, ERVideoCapture
from src.common.frame import FrameError
from src.common.models import FramePredictions

_log = logging.getLogger(__name__)

# Constants for smoothing
EXPRESSION_HISTORY_LEN = 10  # To stabilize expressions


def open_camera(cam_idx: int | str = 0) -> ERVideoCapture:
    """Open the camera device."""
    while True:
        try:
            return ERVideoCapture(cam_idx)
        except CameraError as e:
            if isinstance(cam_idx, str):
                raise e
            cam_idx = int(str(cam_idx))
            cam_idx += 1
            if cam_idx >= 10:
                raise CameraError(cam_idx) from e


def most_common_expression(history: deque[str]) -> str:
    """Get the most frequent expression from history."""
    return max(set(history), key=history.count)


def run_camera(model: nn.Module, cam_idx: int | str) -> None:
    """Run the camera and detect faces and expressions."""
    cap = open_camera(cam_idx)
    _log.info("Starting video stream... Press 'q' to quit.")

    expression_history: deque[str] = deque(maxlen=EXPRESSION_HISTORY_LEN)

    try:
        while True:
            er_frame = cap.get_frame()
            face_rois = er_frame.detect_faces()
            _log.debug("Detected %d faces", len(face_rois))

            # Detect rectangle of each face in the frame
            for roi in face_rois:
                predicted_expression, predictions_with_probabilities = er_frame.predict_expression(model, rectangle=roi)
                expression_history.append(predicted_expression)
                # Use the most frequent expression in the history for stability
                stable_expression = most_common_expression(expression_history)
                er_frame.modify_with_rect(stable_expression, rectangle=roi)

            er_frame.display(face_rois)

            if len(face_rois) > 0:
                frame_predictions = FramePredictions(
                    num_faces=len(face_rois), predictions=predictions_with_probabilities, created_at=datetime.now(UTC)
                )
                # Wait for the event to be set before producing more
                CAN_PRODUCE_EVENT.wait()
                PREDICTIONS_QUEUE.put(frame_predictions)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                STOP_EVENT.set()
                _log.info("Exit triggered. Stopping video...")
                break
    except FrameError:
        _log.info("Cannot read frame. Stopping video.")
    finally:
        cap.release()
