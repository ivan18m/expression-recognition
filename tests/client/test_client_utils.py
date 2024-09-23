from datetime import UTC, datetime

import pytest
from freezegun import freeze_time

from src.client.client_utils import average_predictions
from src.common.models import FramePredictions, Prediction

NOW = datetime(2024, 9, 23, 15, 0, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "predictions, expected",  # noqa: PT006
    [
        (
            [
                FramePredictions(
                    num_faces=2,
                    predictions=[
                        Prediction(label="happiness", probability=0.8),
                        Prediction(label="surprise", probability=0.2),
                    ],
                    created_at=NOW,
                ),
                FramePredictions(
                    num_faces=1,
                    predictions=[
                        Prediction(label="sadness", probability=0.6),
                        Prediction(label="surprise", probability=0.4),
                    ],
                    created_at=NOW,
                ),
            ],
            FramePredictions(
                num_faces=2,
                predictions=[
                    Prediction(label="happiness", probability=0.8),
                    Prediction(label="surprise", probability=0.3),  # averaged probability
                    Prediction(label="sadness", probability=0.6),
                ],
                created_at=NOW,
            ),
        ),
        (
            [
                FramePredictions(
                    num_faces=1, predictions=[Prediction(label="happiness", probability=0.9)], created_at=NOW
                )
            ],
            FramePredictions(num_faces=1, predictions=[Prediction(label="happiness", probability=0.9)], created_at=NOW),
        ),
    ],
)
@freeze_time(NOW)
def test_average_predictions(predictions, expected):
    result = average_predictions(predictions)

    assert result.num_faces == expected.num_faces
    assert len(result.predictions) == len(expected.predictions)
    for i in range(len(result.predictions)):
        assert result.predictions[i].label == expected.predictions[i].label
        assert round(result.predictions[i].probability, 4) == round(expected.predictions[i].probability, 4)
        assert result.created_at == expected.created_at


def test_average_predictions_empty():
    with pytest.raises(ValueError, match="No predictions to average."):
        average_predictions([])
