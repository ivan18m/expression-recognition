import asyncio
import logging

import zmq
import zmq.asyncio

from src.client.buffer import CAN_PRODUCE_EVENT, PREDICTIONS_QUEUE, STOP_EVENT
from src.client.client_utils import CONSUMER_CLIENT_SLEEP_TIME_S, BaseERClient, average_predictions
from src.common.models import FramePredictions

_log = logging.getLogger(__name__)
CONNECTION_URL = "tcp://localhost:5555"


class ZmqClient(BaseERClient):
    def __init__(self, connection_url: str = CONNECTION_URL):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(connection_url)

    async def send_predictions(self, frame_predictions: FramePredictions):
        """Asynchronously sends a message to the server."""
        payload = frame_predictions.model_dump()
        await self.socket.send_json(payload)
        _log.debug("Sent prediction %s", payload)

    def close(self):
        """Closes the ZMQ socket."""
        self.socket.close()
        self.context.term()

    def __del__(self):
        """Automatically called when the object is deleted."""
        asyncio.run(self.close())


async def run_zmq_client():
    client = ZmqClient()
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
                    avg_predictions = average_predictions(predictions)
                    await client.send_predictions(avg_predictions)
            await asyncio.sleep(CONSUMER_CLIENT_SLEEP_TIME_S)
    finally:
        await client.close()
