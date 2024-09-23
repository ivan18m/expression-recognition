from datetime import UTC, datetime

from src.common.models import FramePredictions, Prediction

CONSUMER_CLIENT_SLEEP_TIME_S = 10.0


class BaseERClient:
    """Base class for emotion recognition clients."""


def average_predictions(predictions: list[FramePredictions]) -> FramePredictions:
    """Calculate the average of predictions."""
    if len(predictions) == 0:
        raise ValueError("No predictions to average.")

    num_faces_sum = 0
    # Sum probabilities and count for each label
    predictions_sum: dict[str, tuple[float, int]] = {}
    for frame_pred in predictions:
        num_faces_sum += frame_pred.num_faces
        for p in frame_pred.predictions:
            p_sum, p_count = predictions_sum.get(p.label, (0, 0))
            predictions_sum[p.label] = (p_sum + p.probability, p_count + 1)

    # Calculate avg from sum and count
    num_faces = round(num_faces_sum / len(predictions))
    avg_predictions = []
    for label, (p_sum, p_count) in predictions_sum.items():
        avg_predictions.append(Prediction(label=label, probability=round(p_sum / p_count, 4)))
    return FramePredictions(num_faces=num_faces, predictions=avg_predictions, created_at=datetime.now(UTC))
