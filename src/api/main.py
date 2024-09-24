import logging
from asyncio import Lock

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.common.models import ExpressionModelError, FramePredictions
from src.common.utils import setup_logging

setup_logging("logs/api.log", log_level=logging.DEBUG)
_log = logging.getLogger(__name__)

LATEST_PREDICTIONS: FramePredictions = None
LATEST_PREDICTIONS_LOCK = Lock()

app = FastAPI(title="Emotion Recognition API")


class MoodResponse(BaseModel):
    mood: str


@app.post("/predictions", status_code=status.HTTP_201_CREATED)
async def get_predictions(frame_predictions: FramePredictions) -> FramePredictions:
    try:
        # The data will already be validated by Pydantic if the types and values match the schema
        _log.info("Received predictions: %s", frame_predictions)
        async with LATEST_PREDICTIONS_LOCK:
            global LATEST_PREDICTIONS
            LATEST_PREDICTIONS = frame_predictions
        _log.info("Stored predictions in the queue %s.", LATEST_PREDICTIONS)
        return frame_predictions
    except ExpressionModelError as exc:
        # Return a 400 error if any validation fails in the model
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/mood/latest")
async def get_mood() -> MoodResponse:
    global LATEST_PREDICTIONS
    if LATEST_PREDICTIONS is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No mood data available.")

    async with LATEST_PREDICTIONS_LOCK:
        # Find the maximum probability from latest_predictions.predictions
        max_prob = max(LATEST_PREDICTIONS.predictions, key=lambda x: x.probability)
    return MoodResponse(mood=max_prob.label)


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
