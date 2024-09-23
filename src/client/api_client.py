import asyncio
import logging

import httpx

from src.client.buffer import CAN_PRODUCE_EVENT, PREDICTIONS_QUEUE, STOP_EVENT
from src.client.client_utils import CONSUMER_CLIENT_SLEEP_TIME_S, BaseERClient, average_predictions
from src.common.models import FramePredictions

_log = logging.getLogger(__name__)
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT_S = 5.0


class ERApiError(Exception):
    """Custom exception for API errors."""


class ERApiTimeoutError(ERApiError):
    """Exception for API timeout errors."""

    def __init__(self, req_method: str, url: str, timeout: int, data: dict | str):
        super().__init__(f"API {req_method} request to {url} timed out after {timeout} seconds.\nPayload: {data}")


class ERApiClient(BaseERClient):
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = API_TIMEOUT_S):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=base_url, http2=True, timeout=timeout)
        self.headers = {"Content-Type": "application/json", "Accept": "*/*", "Connection": "keep-alive"}
        _log.info("API client initialized with base URL %s and timeout %s", base_url, timeout)

    async def _post(self, url: str, data: dict | str):
        try:
            if isinstance(data, dict):
                response = await self.client.post(url, json=data)
            elif isinstance(data, str):
                response = await self.client.post(url, data=data)
            else:
                raise ValueError("`data` to POST must be a dictionary or string")
        except httpx.TimeoutException as exc:
            raise ERApiTimeoutError("POST", url, self.timeout, data) from exc
        response.raise_for_status()
        return response

    async def send_predictions(self, frame_predictions: FramePredictions):
        """Asynchronously sends a message to the server."""
        payload = frame_predictions.model_dump_json()
        url = f"{self.base_url}/predictions"
        response = await self._post(url, payload)
        _log.debug(
            "Sent prediction payload %s to %s. Received %d %s", payload, url, response.status_code, response.json()
        )

    async def close(self):
        """Gracefully close the API client."""
        await self.client.aclose()
        _log.info("API client closed.")


async def run_api_client():
    _log.info("Starting API client...")
    client = ERApiClient()
    try:
        while not STOP_EVENT.is_set():
            if not PREDICTIONS_QUEUE.empty():
                predictions: list[FramePredictions] = []

                CAN_PRODUCE_EVENT.clear()  # Prevent producing while consuming
                # Drain the queue
                while not PREDICTIONS_QUEUE.empty():
                    frame_predictions = PREDICTIONS_QUEUE.get()
                    predictions.append(frame_predictions)
                CAN_PRODUCE_EVENT.set()  # Allow producer to continue

                if predictions:
                    try:
                        avg_predictions = average_predictions(predictions)
                        await client.send_predictions(avg_predictions)
                    except httpx.HTTPError as exc:
                        _log.error("API error, Data not sent %s", exc.__qualname__)

            await asyncio.sleep(CONSUMER_CLIENT_SLEEP_TIME_S)
    finally:
        await client.close()
