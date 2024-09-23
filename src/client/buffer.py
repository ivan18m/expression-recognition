from multiprocessing import Event, Queue

MAX_BUFFER_SIZE = 256

# Buffers
PREDICTIONS_QUEUE = Queue(maxsize=MAX_BUFFER_SIZE)  # Queue for FramePredictions

# Events
CAN_PRODUCE_EVENT = Event()
CAN_PRODUCE_EVENT.set()  # Initially allow the producer to work
STOP_EVENT = Event()

"""Note:
If there's any thread that holds a lock or imports a module, and fork is called, it's very likely that the subprocess
will be in a corrupted state and will deadlock or fail in a different way. Note that even if you don't, Python built
in libraries do - no need to look further than multiprocessing. multiprocessing.Queue is actually a very complex
class, that spawns multiple threads used to serialize, send and receive objects, and they can cause aforementioned
problems too. If you find yourself in such situation try using a SimpleQueue, that doesn't use any additional threads.
"""
