import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchcam.methods import SmoothGradCAMpp

from src.common.const import CONFIG
from src.common.models import Prediction
from src.train.model import ERBaseModel

_log = logging.getLogger(__name__)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
SMOOTHING_FACTOR = 0.5  # For bounding box smoothing

Rect = tuple[int, int, int, int]


class FrameError(Exception):
    def __init__(self, device: int):
        super().__init__(f"Failed to read the frame from camera device {device}")


def smooth_bounding_box(current_box: Rect, previous_box: Rect, alpha=SMOOTHING_FACTOR) -> Rect:
    """Apply a linear interpolation to smooth bounding box coordinates."""
    return [int(alpha * c + (1 - alpha) * p) for c, p in zip(current_box, previous_box, strict=False)]


class ERFrame:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.previous_face_rois: list[Rect] = []

    def __call__(self) -> np.ndarray:
        return self.frame

    def detect_faces(
        self, scale_factor: float = 1.05, min_neighbors: int = 5, min_size: tuple[int, int] = (36, 36)
    ) -> list[Rect]:
        """Detect faces in the frame and return the rectangles of the faces."""
        # Use the opencv face cascade to detect faces
        face_rois = FACE_CASCADE.detectMultiScale(
            self.frame,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Smooth the bounding box by interpolating with the previous frame's bounding box
        if len(self.previous_face_rois) != len(face_rois):
            self.previous_face_rois = face_rois
        new_face_rois: list[Rect] = []
        for roi_idx, roi in enumerate(face_rois):
            if roi_idx < len(self.previous_face_rois):
                roi = smooth_bounding_box(roi, self.previous_face_rois[roi_idx])
            new_face_rois.append(roi)
        return new_face_rois

    def predict_expression(self, model: ERBaseModel, rectangle: Rect | None = None) -> tuple[str, list[Prediction]]:
        # Convert to grayscale
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        if rectangle is not None:
            # Unpack rectangle and crop the face
            (x, y, w, h) = rectangle
            face_img = gray[y : y + h, x : x + w]
        else:
            face_img = gray

        # Transform image
        face_img = cv2.resize(face_img, CONFIG.image_size)
        face_img = np.expand_dims(face_img, axis=(0, 1))
        # Normalize image
        face_img = face_img / 255.0

        # To torch tensor
        face_tensor = torch.from_numpy(face_img).type(torch.FloatTensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_tensor = face_tensor.to(device)
        if CONFIG.use_gradcam:
            predicted_class, probabilities = self._add_gradcam(model, face_tensor, rectangle)
        else:
            with torch.no_grad():
                out = model(face_tensor)
                probabilities = F.softmax(out, dim=1)
                predicted = torch.argmax(out, dim=1)
                predicted_class = CONFIG.labels[predicted.item()]

        probs = probabilities[0].cpu().detach().numpy()
        predictions_probability = self._get_probability_predictions(probs)
        return predicted_class, predictions_probability

    def modify_with_rect(self, expression: str, rectangle: Rect) -> np.ndarray:
        """Modify the frame to highlight the face rectangle."""
        # unpack rectangle
        (x, y, w, h) = rectangle
        cv2.rectangle(self.frame, (x, y), (x + w, y + h), CONFIG.rectangle_color, 2)
        cv2.putText(
            self.frame,
            f"Expression: {expression}",
            org=(x, y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=CONFIG.text_color,
            thickness=2,
        )
        return self.frame

    def display(self, face_rois: list[Rect]) -> np.ndarray:
        """Modify the frame to include metadata."""
        # Metadata on the frame
        cv2.putText(
            self.frame,
            f"Number of faces: {len(face_rois)}",
            org=(10, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=CONFIG.text_color,
            thickness=1,
        )
        # Display the frame
        cv2.imshow(f"Cam - {CONFIG.app_name}", self.frame)
        return self.frame

    def _add_gradcam(self, model: ERBaseModel, face_tensor: torch.Tensor, rectangle: Rect) -> tuple[str, Any]:
        """Add GradCAM to the frame."""
        (x, y, w, h) = rectangle
        with SmoothGradCAMpp(model, model.last_conv_layer, input_shape=(512, 6, 6)) as cam_extractor:
            out = model(face_tensor)
            probabilities = F.softmax(out, dim=1)
            predicted = torch.argmax(out, dim=1)
            predicted_class = CONFIG.labels[predicted.item()]

            # Get the CAM which will map the input image to the output class
            activation_map = cam_extractor(predicted.item(), out)
            # Merge frame's rectangle with gradcam's activation heatmap
            heatmap = cv2.resize(activation_map[0].squeeze(0).cpu().numpy(), (w, h))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_VIRIDIS)
            self.frame[y : y + h, x : x + w] = cv2.addWeighted(
                self.frame[y : y + h, x : x + w], 1, heatmap, 0.5, gamma=0
            )
        return predicted_class, probabilities

    def _get_probability_predictions(self, probabilities: np.ndarray) -> list[Prediction]:
        labeled_probabilities = list(zip(CONFIG.labels, probabilities, strict=False))
        _log.debug("Probabilities: %s", labeled_probabilities)
        return [Prediction(label=label, probability=round(prob, 2)) for label, prob in labeled_probabilities]
