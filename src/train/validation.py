import csv
import logging
from pathlib import Path

import cv2
import torch.nn as nn

from src.common.const import CONFIG
from src.common.frame import ERFrame
from src.common.utils import get_all_files, get_path

_log = logging.getLogger(__name__)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def validate_to_csv(images_path: Path, model: nn.Module) -> None:
    """
    Validate the dataset on a directory of images.
    Save the classification scores to a CSV file.
    """
    csv_rows = [["filepath", *CONFIG.labels, "predicted_label"]]

    correct = 0
    for image_name in get_all_files(images_path, exp=r".*\.(jpg|jpeg|png|bmp|tiff)$"):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # Load image
            image_path = get_path(images_path, image_name)
            image = cv2.imread(image_path)
            frame = ERFrame(image)
            # Detect the faces using built-in CV2 function
            face_rois = FACE_CASCADE.detectMultiScale(image, scaleFactor=1.05, minNeighbors=4, minSize=(36, 36))
            if len(face_rois) != 1:
                predicted_class, rounded_probabilities = frame.predict_expression(model, use_gradcam=False)
            else:
                # If more than 1 face are detected in the image, use the first one
                rect = face_rois[0]
                predicted_class, rounded_probabilities = frame.predict_expression(
                    model, use_gradcam=False, rectangle=rect
                )
            if predicted_class.lower() in image_name.lower():
                correct += 1
            csv_rows.append([str(image_path), *rounded_probabilities, predicted_class])

    csv_file_path = get_path(images_path, "classification_scores.csv")
    with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)
    accuracy = correct / (len(csv_rows) - 1) * 100
    _log.info("Accuracy %.2f%%. Classification scores have been saved to %s", accuracy, csv_file_path)
