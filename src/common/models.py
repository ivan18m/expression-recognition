from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from src.common.const import CONFIG


class ExpressionModelError(Exception):
    def __init__(self, message):
        super().__init__(f"{message}")


class BaseExpressionModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class Prediction(BaseExpressionModel):
    label: str
    probability: float

    @field_validator("probability")
    @classmethod
    def _validate_between_0_and_1(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ExpressionModelError("Must be between 0 and 1.")
        return value

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        if value not in CONFIG.labels:
            raise ExpressionModelError(f"Must be one of {CONFIG.labels}.")
        return value


class FramePredictions(BaseExpressionModel):
    num_faces: int
    predictions: list[Prediction]
    created_at: datetime

    @field_validator("num_faces")
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ExpressionModelError("Must be non-negative.")
        return value
